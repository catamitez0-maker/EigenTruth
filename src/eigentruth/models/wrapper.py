"""EigenTruth Models — HuggingFace 顶层封装器。

提供 EigenTruthWrapper，实现：
- warmup(): 真值流形冷启动
- generate(): 受控文本生成（自带幻觉监测与实时纠偏）
- forward(): 透传给原始模型

非侵入式设计：不修改原始模型参数，通过 hook 实现所有干预。
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from eigentruth.core.math_engine import TruthManifold
from eigentruth.intervention.hooks import TruthProbe

logger = logging.getLogger("eigentruth")

# 配置 eigentruth logger 默认输出格式
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[EigenTruth] %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class EigenTruthWrapper(nn.Module):
    """EigenTruth 主封装器：为 HuggingFace CausalLM 模型穿戴幻觉治理装甲。

    用法示例::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from eigentruth.models.wrapper import EigenTruthWrapper

        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        safe_model = EigenTruthWrapper(
            model=base_model,
            target_layer_idx=-10,
            steering_lambda=0.1,
            mahalanobis_threshold=15.0,
        )
        safe_model.warmup(["光速是每秒299792458米。"], tokenizer)
        outputs = safe_model.generate(**inputs, max_new_tokens=100)

    Args:
        model: HuggingFace CausalLM 模型实例.
        target_layer_idx: 挂载探针的 Transformer 层索引（支持负索引）.
        steering_lambda: 激活引导强度 (0 = 纯监测, 不干预).
        mahalanobis_threshold: 马氏距离阈值，超出时触发引导与预警.
        hse_warning_threshold: HSE 预警阈值.
        curvature: 庞加莱球曲率参数.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = -10,
        steering_lambda: float = 0.1,
        mahalanobis_threshold: float = 15.0,
        hse_warning_threshold: float = 5.0,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()

        self.model = model
        self.target_layer_idx = target_layer_idx
        self.steering_lambda = steering_lambda
        self.mahalanobis_threshold = mahalanobis_threshold
        self.hse_warning_threshold = hse_warning_threshold
        self.curvature = curvature

        # 内部状态
        self.manifold = TruthManifold()
        self.probe: Optional[TruthProbe] = None
        self._is_warmed_up: bool = False

    # ------------------------------------------------------------------
    # Warmup: 真值流形冷启动
    # ------------------------------------------------------------------

    @torch.no_grad()
    def warmup(
        self,
        fact_dataset: List[str],
        tokenizer: Any,
        max_length: int = 128,
        batch_size: int = 1,
    ) -> None:
        """使用事实语料构建真值流形。

        遍历 fact_dataset 中的每条事实文本，通过模型前向传播
        提取目标层的隐状态，增量构建 TruthManifold。

        Args:
            fact_dataset: 绝对正确的事实文本列表.
            tokenizer: HuggingFace tokenizer.
            max_length: tokenize 最大长度.
            batch_size: 暂时为 1（MVP）.
        """
        self.manifold = TruthManifold()
        device = self._get_device()

        # 临时 hook 用于收集隐状态
        collected_states: List[Tensor] = []

        def _collect_hook(module: nn.Module, input: Any, output: Any) -> None:
            """临时 hook：收集隐状态并更新流形。"""
            hidden = output[0] if isinstance(output, tuple) else output
            # 取所有 token 的均值作为该句的表征
            h_mean = hidden.detach().mean(dim=1).squeeze(0)  # [D]
            collected_states.append(h_mean.cpu())

        # 定位目标层并注册临时 hook
        layers = TruthProbe._find_layers(self.model)
        target_layer = layers[self.target_layer_idx]
        hook_handle = target_layer.register_forward_hook(_collect_hook)

        try:
            for i, text in enumerate(fact_dataset):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                )
                # .to(device) 返回的可能是 BatchEncoding 或 dict
                inputs = inputs.to(device)

                # 兼容 dict 和 BatchEncoding：提取 input_ids
                if isinstance(inputs, dict):
                    self.model(**inputs)
                elif hasattr(inputs, "input_ids"):
                    self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=getattr(inputs, "attention_mask", None),
                    )
                else:
                    self.model(**dict(inputs))

                logger.debug(f"Warmup: processed {i + 1}/{len(fact_dataset)}")
        finally:
            hook_handle.remove()

        # 用收集到的隐状态构建流形
        for h in collected_states:
            self.manifold.update(h)

        if not self.manifold.is_ready():
            logger.warning(
                f"⚠️ 流形构建不完整（仅 {self.manifold.n} 个样本）。"
                f"建议至少提供 2 条以上事实语料。"
            )
        else:
            logger.info(
                f"✅ 真值流形已就绪 — {self.manifold.n} 个样本, "
                f"hidden_dim={self.manifold.hidden_dim}"
            )

        # 挂载正式探针
        self._activate_probe()
        self._is_warmed_up = True

    # ------------------------------------------------------------------
    # Generate: 受控文本生成
    # ------------------------------------------------------------------

    def generate(self, **kwargs: Any) -> Any:
        """受控文本生成。

        透传参数给原始 model.generate()，在生成过程中
        TruthProbe hook 自动工作，实时监测并干预幻觉。

        生成完成后检查 HSE，如超阈值则输出预警。

        Args:
            **kwargs: 传递给 model.generate() 的所有参数.

        Returns:
            model.generate() 的原始返回值.
        """
        if not self._is_warmed_up:
            logger.warning(
                "⚠️ 模型未经 warmup，将以无防护模式生成。"
                "请先调用 safe_model.warmup(fact_dataset, tokenizer)。"
            )
            return self.model.generate(**kwargs)

        # 重置 HSE 历史
        if self.probe is not None:
            self.probe.reset_history()

        # 生成
        outputs = self.model.generate(**kwargs)

        # 生成后检查 HSE
        self._check_hse_warning()

        return outputs

    # ------------------------------------------------------------------
    # Forward: 透传
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """透传给原始模型的 forward 方法。"""
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def is_warmed_up(self) -> bool:
        """流形是否已构建完成。"""
        return self._is_warmed_up

    @property
    def last_distance(self) -> float:
        """最近一次的马氏距离。"""
        return self.probe.last_distance if self.probe else 0.0

    @property
    def last_hse(self) -> float:
        """最近一次的双曲语义熵。"""
        return self.probe.last_hse if self.probe else 0.0

    def get_diagnostics(self) -> dict:
        """获取诊断信息字典。"""
        return {
            "is_warmed_up": self._is_warmed_up,
            "manifold_samples": self.manifold.n,
            "hidden_dim": self.manifold.hidden_dim,
            "last_mahalanobis_distance": self.last_distance,
            "last_hse": self.last_hse,
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "hse_warning_threshold": self.hse_warning_threshold,
            "steering_lambda": self.steering_lambda,
            "probe_active": self.probe.is_active if self.probe else False,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _activate_probe(self) -> None:
        """创建并激活正式探针。"""
        if self.probe is not None:
            self.probe.remove()

        self.probe = TruthProbe(
            manifold=self.manifold,
            steering_lambda=self.steering_lambda,
            threshold=self.mahalanobis_threshold,
        )
        self.probe.register(self.model, self.target_layer_idx)

    def _check_hse_warning(self) -> None:
        """检查 HSE 是否超阈值并输出预警。"""
        if self.probe is None:
            return

        if self.probe.last_hse > self.hse_warning_threshold:
            logger.warning(
                f"⚠️ 检测到深层语义发散 (HSE={self.probe.last_hse:.2f} > "
                f"阈值 {self.hse_warning_threshold:.2f})，"
                f"系统可能正在产生幻觉！"
            )

    def _get_device(self) -> torch.device:
        """获取模型所在设备。"""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def detach_probe(self) -> None:
        """移除探针，恢复原始模型行为。"""
        if self.probe is not None:
            self.probe.remove()
            self.probe = None
        self._is_warmed_up = False
        logger.info("🔓 EigenTruth 探针已移除，模型恢复原始状态。")

    def __del__(self) -> None:
        """析构时确保 hook 被移除。"""
        try:
            self.detach_probe()
        except Exception:
            pass
