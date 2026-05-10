"""EigenTruth Intervention — 动态探针与 Hook 系统 / Dynamic Probe and Hook System.

通过 PyTorch forward_hook 实现对 Transformer 隐状态的 / Using PyTorch forward_hook to achieve the following on Transformer hidden states:
1. 实时拦截与马氏距离监测 / Real-time interception and Mahalanobis distance monitoring
2. 激活引导向量 (Steering) 注入 / Activation steering vector injection
3. 庞加莱映射与双曲语义熵 (HSE) 跟踪 / Poincaré mapping and Hyperbolic Semantic Entropy (HSE) tracking

所有 Hook 内部操作均在 torch.no_grad() 下执行，
截获的张量通过 .detach() 脱离计算图，严格防止显存泄漏。
All internal Hook operations are executed under torch.no_grad().
Intercepted tensors are detached from the compute graph to strictly prevent memory leaks.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from eigentruth.core.math_engine import (
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
)

logger = logging.getLogger("eigentruth")


class TruthProbe:
    """真值探针：在指定 Transformer 层上挂载 forward_hook，
    实时监测隐状态偏移并注入引导修正。
    Truth Probe: Attaches a forward_hook to a specified Transformer layer
    to monitor hidden state deviations in real-time and inject steering corrections.

    Attributes:
        manifold: 真值流形（由 warmup 阶段构建） / Truth manifold (built during warmup).
        steering_lambda: 引导向量注入强度 (0 = 不干预) / Steering vector injection strength (0 = no intervention).
        threshold: 马氏距离阈值，超出时触发引导干预 / Mahalanobis distance threshold; exceeds trigger steering intervention.
        last_distance: 最近一次 hook 触发时的马氏距离 / The Mahalanobis distance from the last triggered hook.
        last_hse: 最近一次计算的双曲语义熵 / The most recently computed Hyperbolic Semantic Entropy.
        is_active: hook 是否已挂载且激活 / Whether the hook is mounted and active.
    """

    def __init__(
        self,
        manifold: TruthManifold,
        steering_lambda: float = 0.1,
        threshold: float = 15.0,
        hse_window_size: int = 20,
    ) -> None:
        self.manifold = manifold
        self.steering_lambda = steering_lambda
        self.threshold = threshold
        self.hse_window_size = hse_window_size

        # 运行时状态 (取 Batch 最大值供快速查询)
        self.last_distance: float = 0.0
        self.last_hse: float = 0.0
        self.is_active: bool = False

        # HSE 追踪：收集生成过程中的庞加莱点 [W, B, D]
        self._poincare_history: Deque[Tensor] = deque(maxlen=hse_window_size)

        # hook 句柄
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def register(self, model: nn.Module, layer_idx: int) -> None:
        """在指定 Transformer 层注册 forward_hook。
        Register a forward_hook on the specified Transformer layer.

        支持负索引（如 -10 表示倒数第 10 层）。
        Supports negative indexing (e.g., -10 means the 10th layer from the end).

        Args:
            model: HuggingFace CausalLM 模型（或任意含 `.model.layers` 的模型）/ HuggingFace CausalLM model.
            layer_idx: 目标层索引，支持负索引 / Target layer index, supports negative indexing.

        Raises:
            ValueError: 无法定位 Transformer 层列表 / Cannot locate Transformer layer list.
            IndexError: layer_idx 超出范围 / layer_idx out of bounds.
        """
        layers = self._find_layers(model)
        target = layers[layer_idx]

        # 如果已有 hook，先移除
        self.remove()

        self._hook_handle = target.register_forward_hook(self._hook_fn)
        self.is_active = True
        logger.debug(
            f"TruthProbe registered on layer {layer_idx} "
            f"(actual: {type(target).__name__})"
        )

    def remove(self) -> None:
        """安全移除 hook，释放资源。
        Safely remove the hook and release resources.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.is_active = False
        self._poincare_history.clear()

    def reset_history(self) -> None:
        """清空 HSE 历史（每次 generate 调用前重置）。
        Clear HSE history (reset before every generate call).
        """
        self._poincare_history.clear()
        self.last_hse = 0.0

    # ------------------------------------------------------------------
    # Hook 回调
    # ------------------------------------------------------------------

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """forward_hook 回调：拦截隐状态、计算距离、注入引导。
        forward_hook callback: Intercept hidden states, compute distance, and inject steering.

        工程防线 / Engineering Defenses:
        - 处理 HF 的 tuple/BaseModelOutputWithPast 输出格式 / Handle HF's tuple/BaseModelOutputWithPast output format
        - .detach() 脱离计算图 / Detach from computation graph
        - torch.no_grad() 包裹所有运算 / Wrap all operations in torch.no_grad()
        - 只处理最后一个 token (h[:, -1:, :]) / Only process the last token (h[:, -1:, :])
        """
        # 防线 3: 处理 tuple 输出（HF 模型常见）
        hidden, is_tuple, rest = self._unpack_output(output)

        if not self.manifold.is_ready():
            return output  # 流形未就绪，直接透传

        with torch.no_grad():
            # 防线 2+3: detach + 切片最后一个 token
            h_last = hidden[:, -1:, :].detach()  # [B, 1, D]
            h_vec = h_last.squeeze(1)              # [B, D]

            # 批量计算马氏距离 [B]
            dist = mahalanobis_distance(
                h_vec, self.manifold.mean, self.manifold.cov_inv
            )
            # 保存最大距离用于预警诊断
            self.last_distance = dist.max().item()

            # 庞加莱映射 + HSE 追踪
            h_poincare = poincare_map(h_vec) # [B, D]
            self._poincare_history.append(h_poincare.cpu())

            if len(self._poincare_history) >= 2:
                pts = torch.stack(list(self._poincare_history)) # [W, B, D]
                hse_batch = hyperbolic_semantic_entropy(pts) # [B]
                self.last_hse = hse_batch.max().item()

            # 引导干预
            mask = dist > self.threshold # [B]
            if mask.any() and self.steering_lambda > 0:
                steering = self._compute_steering_vector(h_vec)  # [B, D]
                # 注入：只修改最后一个 token 的激活
                correction = self.steering_lambda * steering  # [B, D]
                # 将无需干预的批次修正量清零
                correction = correction * mask.unsqueeze(1).to(correction.dtype)
                
                hidden = hidden.clone()  # 避免原地修改影响计算图
                hidden[:, -1:, :] = hidden[:, -1:, :] + correction.unsqueeze(1)

        return self._repack_output(hidden, is_tuple, rest)

    # ------------------------------------------------------------------
    # 引导向量
    # ------------------------------------------------------------------

    def _compute_steering_vector(self, h: Tensor) -> Tensor:
        """计算从当前激活到真值质心的归一化引导方向。
        Compute the normalized steering direction from the current activation to the truth centroid.

        如果有对比方向，则沿对比方向；否则方向朝向真值质心。
        If a contrastive direction exists, follow it; otherwise, direct towards the truth centroid.

        Args:
            h: 当前隐状态, 形状 [B, D] / Current hidden state, shape [B, D].

        Returns:
            引导向量, 形状 [B, D] / Steering vector, shape [B, D].
        """
        if self.manifold.contrastive_direction is not None:
            direction = self.manifold.contrastive_direction.to(h.device)
            direction = direction.unsqueeze(0).expand_as(h)
        else:
            mean = self.manifold.mean.to(h.device)
            direction = mean.to(torch.float32) - h.to(torch.float32)
            
        norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        return (direction / norm).to(h.dtype)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _find_layers(model: nn.Module) -> nn.ModuleList:
        """在 HF 模型中定位 Transformer 层列表。
        Locate the list of Transformer layers in an HF model.

        尝试常见路径 / Try common paths: model.model.layers, model.transformer.h, etc.

        Returns:
            nn.ModuleList 或类似可索引容器 / nn.ModuleList or similar indexable container.

        Raises:
            ValueError: 无法自动定位层列表 / Cannot automatically locate layer list.
        """
        # 常见 HF 架构的层路径
        candidates = [
            ("model", "layers"),        # Llama, Qwen, Mistral
            ("transformer", "h"),       # GPT-2, GPT-Neo
            ("model", "decoder", "layers"),  # OPT
            ("gpt_neox", "layers"),     # GPT-NeoX
        ]

        for path in candidates:
            obj = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    return obj
            except AttributeError:
                continue

        # 回退: 如果模型本身就有 .layers 属性（测试用简单模型）
        if hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, list)):
            return model.layers

        raise ValueError(
            "Cannot locate transformer layers. "
            "Supported patterns: model.model.layers, model.transformer.h, etc. "
            "If using a custom model, ensure it has a .layers attribute."
        )

    @staticmethod
    def _unpack_output(output: Any) -> Tuple[Tensor, bool, tuple]:
        """解包 HF 模型输出，提取 hidden_states 张量。
        Unpack HF model output to extract the hidden_states tensor.

        Returns:
            (hidden_states, is_tuple, rest_of_tuple)
        """
        if isinstance(output, tuple):
            return output[0], True, output[1:]
        if isinstance(output, Tensor):
            return output, False, ()
        # BaseModelOutputWithPast 等 dataclass 式输出
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state, False, ()
        # 回退
        raise TypeError(
            f"Unsupported output type from hooked layer: {type(output)}. "
            f"Expected Tensor, tuple, or HF BaseModelOutput."
        )

    @staticmethod
    def _repack_output(
        hidden: Tensor, is_tuple: bool, rest: tuple
    ) -> Union[Tensor, tuple]:
        """将修改后的 hidden_states 重新打包为原始格式。
        Repack the modified hidden_states into their original format.
        """
        if is_tuple:
            return (hidden,) + rest
        return hidden
