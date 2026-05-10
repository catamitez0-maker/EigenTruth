"""EigenTruth Models — HuggingFace 顶层封装器 / HuggingFace Top-level Wrapper.

提供 EigenTruthWrapper，实现 / Provides EigenTruthWrapper, implementing:
- warmup(): 真值流形冷启动 / Truth manifold cold start
- generate(): 受控文本生成（自带幻觉监测与实时纠偏） / Controlled text generation (with built-in hallucination monitoring and real-time correction)
- forward(): 透传给原始模型 / Passthrough to the original model

非侵入式设计：不修改原始模型参数，通过 hook 实现所有干预。
Non-intrusive design: Does not modify original model parameters, implements all interventions via hooks.
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
    EigenTruth Main Wrapper: Equips HuggingFace CausalLM models with hallucination governance armor.

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
        model: HuggingFace CausalLM 模型实例 / HuggingFace CausalLM model instance.
        target_layer_idx: 挂载探针的 Transformer 层索引（支持负索引） / Transformer layer index to mount the probe (supports negative indexing).
        steering_lambda: 激活引导强度 (0 = 纯监测, 不干预) / Activation steering strength (0 = monitor only, no intervention).
        mahalanobis_threshold: 马氏距离阈值，超出时触发引导与预警 / Mahalanobis distance threshold, triggers steering and warnings when exceeded.
        hse_warning_threshold: HSE 预警阈值 / HSE warning threshold.
        curvature: 庞加莱球曲率参数 / Poincaré ball curvature parameter.
        hse_window_size: HSE 滑动窗口大小 / HSE sliding window size.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int = -10,
        steering_lambda: float = 0.1,
        mahalanobis_threshold: float = 15.0,
        hse_warning_threshold: float = 5.0,
        curvature: float = 1.0,
        hse_window_size: int = 20,
    ) -> None:
        super().__init__()

        self.model = model
        self.target_layer_idx = target_layer_idx
        self.steering_lambda = steering_lambda
        self.mahalanobis_threshold = mahalanobis_threshold
        self.hse_warning_threshold = hse_warning_threshold
        self.curvature = curvature
        self.hse_window_size = hse_window_size

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
        false_dataset: Optional[List[str]] = None,
        max_length: int = 128,
        batch_size: int = 1,
    ) -> None:
        """使用事实语料构建真值流形。
        Build the truth manifold using a factual corpus.

        提取目标层的隐状态，增量构建 TruthManifold。
        如果提供了 false_dataset，将额外构建对比方向。
        Extracts hidden states from the target layer to incrementally build the TruthManifold.
        If false_dataset is provided, it will additionally construct a contrastive direction.

        Args:
            fact_dataset: 绝对正确的事实文本列表 / List of absolutely correct factual texts.
            tokenizer: HuggingFace tokenizer.
            false_dataset: 绝对错误的对应文本列表 (用于构建对比方向) / List of corresponding completely false texts (used to build contrastive direction).
            max_length: tokenize 最大长度 / max length for tokenization.
            batch_size: 暂时为 1（MVP） / Currently 1 (MVP).
        """
        self.manifold = TruthManifold()
        device = self._get_device()

        def _collect_states(dataset: List[str]) -> List[Tensor]:
            collected: List[Tensor] = []
            
            def _collect_hook(module: nn.Module, input: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                # 提取最后一个 token 的表征以捕捉因果结论
                h_last = hidden.detach()[:, -1, :].squeeze(0)  # [D]
                collected.append(h_last.cpu())

            layers = TruthProbe._find_layers(self.model)
            target_layer = layers[self.target_layer_idx]
            hook_handle = target_layer.register_forward_hook(_collect_hook)

            try:
                for i, text in enumerate(dataset):
                    inputs = tokenizer(
                        text, return_tensors="pt", max_length=max_length,
                        truncation=True, padding=True
                    ).to(device)
                    
                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    elif hasattr(inputs, "input_ids"):
                        self.model(
                            input_ids=inputs.input_ids,
                            attention_mask=getattr(inputs, "attention_mask", None),
                        )
                    else:
                        self.model(**dict(inputs))
            finally:
                hook_handle.remove()
                
            return collected

        logger.info("Collecting true representations...")
        fact_states = _collect_states(fact_dataset)

        # 用收集到的隐状态构建流形
        for h in fact_states:
            self.manifold.update(h)

        if false_dataset is not None:
            logger.info("Collecting false representations for contrastive direction...")
            false_states = _collect_states(false_dataset)
            if len(false_states) > 0:
                false_mean = torch.stack(false_states).mean(dim=0)
                self.manifold.false_mean = false_mean.to(torch.float32)
                # 构建 Truth Direction = True Mean - False Mean
                self.manifold.contrastive_direction = self.manifold.mean - self.manifold.false_mean
                logger.info("Contrastive direction successfully established.")

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
        Controlled text generation.

        透传参数给原始 model.generate()，在生成过程中
        TruthProbe hook 自动工作，实时监测并干预幻觉。
        Passes parameters through to the original model.generate(). During generation,
        the TruthProbe hook automatically operates, monitoring and intervening in real-time.

        生成完成后检查 HSE，如超阈值则输出预警。
        After generation, checks HSE and outputs a warning if the threshold is exceeded.

        Args:
            **kwargs: 传递给 model.generate() 的所有参数 / All parameters passed to model.generate().

        Returns:
            model.generate() 的原始返回值 / Original return value of model.generate().
        """
        if not self._is_warmed_up:
            logger.warning(
                "⚠️ 模型未经 warmup，将以无防护模式生成。"
                "请先调用 safe_model.warmup(fact_dataset, tokenizer)。\n"
                "⚠️ Model not warmed up, generating in unprotected mode. "
                "Please call safe_model.warmup(fact_dataset, tokenizer) first."
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
            hse_window_size=self.hse_window_size,
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
                f"系统可能正在产生幻觉！\n"
                f"⚠️ Deep semantic divergence detected (HSE={self.probe.last_hse:.2f} > "
                f"Threshold {self.hse_warning_threshold:.2f}). "
                f"The system might be hallucinating!"
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
        logger.info("🔓 EigenTruth 探针已移除，模型恢复原始状态。 / EigenTruth probe removed, model restored to original state.")

    def __del__(self) -> None:
        """析构时确保 hook 被移除。"""
        try:
            self.detach_probe()
        except Exception:
            pass
