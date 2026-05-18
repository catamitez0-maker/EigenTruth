"""EigenTruth Core — 防崩溃数学引擎 / Crash-proof Math Engine.

基于几何动力学的底层数学原语，包括 / Core mathematical primitives based on geometric dynamics, including:
- Sherman-Morrison 在线 precision proxy 更新 / Online precision-proxy update via Sherman-Morrison
- 马氏距离计算 / Mahalanobis distance computation
- 庞加莱球模型映射 / Poincaré ball model mapping
- 双曲语义熵 (HSE) / Hyperbolic Semantic Entropy (HSE)

所有浮点密集运算在内部强制使用 FP32 以确保数值稳定性。
All float-intensive computations are forced to FP32 internally to ensure numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TruthManifold:
    """真值流形：存储事实语料的统计特征。
    Truth Manifold: Stores statistical features of the factual corpus.

    `cov_inv` 是用于快速马氏距离监测的正则化在线 precision proxy，
    不是严格的样本协方差矩阵逆。
    `cov_inv` is a regularized online precision proxy for fast Mahalanobis
    monitoring, not an exact inverse of the empirical sample covariance.

    Attributes:
        mean: 隐状态质心向量 / Hidden state centroid vector, shape [hidden_dim].
        cov_inv: 协方差矩阵的逆 / Inverse covariance matrix, shape [hidden_dim, hidden_dim].
        n: 已累积的样本数量 / Accumulated sample count.
        hidden_dim: 隐状态维度 (由首次更新时自动推断) / Hidden state dimension (inferred automatically on first update).
    """

    mean: Optional[Tensor] = None
    cov_inv: Optional[Tensor] = None
    n: int = 0
    hidden_dim: int = 0

    # 扩展：支持方案 B (对比流形)
    false_mean: Optional[Tensor] = None
    contrastive_direction: Optional[Tensor] = None

    # 运行时设备跟踪（不参与序列化）
    _device: torch.device = field(default_factory=lambda: torch.device("cpu"), repr=False)

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def update(self, h: Tensor, epsilon: float = 1e-6) -> None:
        """用单个隐状态向量增量更新流形。
        Incrementally update the manifold with a single hidden state vector.

        首次调用时自动初始化 mean 和 cov_inv。
        Automatically initializes mean and cov_inv on first call.

        Args:
            h: 隐状态向量 / Hidden state vector, shape [hidden_dim].
            epsilon: Sherman-Morrison 分母正则项 / Denominator regularization term.
        """
        h = h.detach()
        if h.ndim != 1:
            raise ValueError(
                f"TruthManifold.update() expects a 1D hidden state vector, got shape {tuple(h.shape)}."
            )

        if self.mean is None:
            # 首次初始化
            self.hidden_dim = h.shape[-1]
            self._device = h.device
            self.mean = h.clone().to(torch.float32)
            self.cov_inv = torch.eye(
                self.hidden_dim, device=self._device, dtype=torch.float32
            )
            self.n = 1
            return

        if h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self.hidden_dim}, got {h.shape[-1]}."
            )

        self.n += 1
        # 增量协方差逆更新（必须在均值更新前计算 delta）
        # Compute delta BEFORE updating the mean (critical for correct covariance estimation)
        old_mean = self.mean.clone()
        self.mean = self.mean + (h.to(torch.float32) - self.mean) / self.n
        delta = h.to(torch.float32) - old_mean
        self.cov_inv = sherman_morrison_update(self.cov_inv, delta, epsilon)

    def to(self, device: Union[str, torch.device]) -> "TruthManifold":
        """Move manifold tensors to a device in-place and return self."""
        device = torch.device(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.cov_inv is not None:
            self.cov_inv = self.cov_inv.to(device)
        if self.false_mean is not None:
            self.false_mean = self.false_mean.to(device)
        if self.contrastive_direction is not None:
            self.contrastive_direction = self.contrastive_direction.to(device)
        self._device = device
        return self

    def is_ready(self) -> bool:
        """流形至少经过 2 个样本后方可使用。"""
        return self.n >= 2 and self.mean is not None and self.cov_inv is not None

    def save(self, path: Union[str, Path]) -> None:
        """将流形序列化到磁盘。

        Args:
            path: 保存路径 (建议后缀 .pt 或 .bin).
        """
        state = {
            "mean": self.mean,
            "cov_inv": self.cov_inv,
            "n": self.n,
            "hidden_dim": self.hidden_dim,
            "false_mean": self.false_mean,
            "contrastive_direction": self.contrastive_direction,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TruthManifold":
        """从磁盘加载流形。

        Args:
            path: 之前由 save() 保存的文件路径.

        Returns:
            恢复后的 TruthManifold 实例.
        """
        state = torch.load(path, weights_only=True)
        manifold = cls()
        manifold.mean = state["mean"]
        manifold.cov_inv = state["cov_inv"]
        manifold.n = state["n"]
        manifold.hidden_dim = state["hidden_dim"]
        manifold.false_mean = state.get("false_mean", None)
        manifold.contrastive_direction = state.get("contrastive_direction", None)
        if manifold.mean is not None:
            manifold._device = manifold.mean.device
        return manifold


# ---------------------------------------------------------------------------
# 核心数学函数
# ---------------------------------------------------------------------------

def sherman_morrison_update(
    cov_inv: Tensor,
    x: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Sherman-Morrison 秩-1 在线更新协方差逆矩阵。
    Sherman-Morrison rank-1 online update for the inverse covariance matrix.

    公式 / Formula: A⁻¹_new = A⁻¹ - (A⁻¹ x xᵀ A⁻¹) / (1 + xᵀ A⁻¹ x + ε)

    内部所有运算强制使用 FP32 以防止 FP16 数值崩溃。
    All internal computations are forced to FP32 to prevent FP16 numerical collapse.

    Args:
        cov_inv: 当前协方差逆矩阵 / Current inverse covariance matrix, shape [d, d].
        x: 增量向量 / Incremental vector (h - mean), shape [d].
        epsilon: 分母正则项, 防止除零 / Denominator regularization term to prevent division by zero.

    Returns:
        更新后的协方差逆矩阵 / Updated inverse covariance matrix, shape [d, d], with the same dtype as input.
    """
    orig_dtype = cov_inv.dtype

    # 防线 1: 强制 FP32 内部计算
    cov_inv_f32 = cov_inv.to(torch.float32)
    x_f32 = x.to(torch.float32)

    Ax = cov_inv_f32 @ x_f32                           # [d]
    denom = 1.0 + x_f32 @ Ax + epsilon                 # 标量
    cov_inv_f32 = cov_inv_f32 - torch.outer(Ax, Ax) / denom

    return cov_inv_f32.to(orig_dtype)


def mahalanobis_distance(
    h: Tensor,
    mean: Tensor,
    cov_inv: Tensor,
) -> Tensor:
    """计算隐状态到真值质心的马氏距离。
    Compute the Mahalanobis distance from a hidden state to the truth centroid.

    D_M(h) = sqrt( (h - μ)ᵀ Σ⁻¹ (h - μ) )

    Args:
        h: 隐状态向量 / Hidden state vector, shape [..., hidden_dim].
        mean: 质心向量 / Centroid vector, shape [hidden_dim].
        cov_inv: 协方差逆矩阵 / Inverse covariance matrix, shape [hidden_dim, hidden_dim].

    Returns:
        马氏距离张量 / Mahalanobis distance tensor, shape matching batch dimension of h (>=0).
    """
    # 强制 FP32
    delta = (h - mean).to(torch.float32)
    cov_inv_f32 = cov_inv.to(torch.float32)

    # δᵀ Σ⁻¹ δ 批量化计算
    m_sq = (delta @ cov_inv_f32 * delta).sum(dim=-1)
    # clamp 防止浮点误差导致 sqrt 负数
    return torch.sqrt(torch.clamp(m_sq, min=0.0))


def poincare_map(
    h_euclidean: Tensor,
    curvature: float = 1.0,
    max_norm: float = 0.999,
) -> Tensor:
    """将欧氏空间隐状态映射到庞加莱球模型。
    Map Euclidean space hidden states to the Poincaré ball model.

    使用指数映射 / Using exponential map at origin:
        exp_0(v) = tanh(√c ‖v‖ / 2) · v / (√c ‖v‖)

    Args:
        h_euclidean: 欧氏空间向量 / Euclidean space vector, shape [..., dim].
        curvature: 负曲率参数 c (正值) / Negative curvature parameter c (positive value).
        max_norm: 钳位最大范数, 保持在开球内部 / Clamp maximum norm to stay inside the open ball (< 1).

    Returns:
        庞加莱球坐标 / Poincaré ball coordinates, shape [..., dim], norm < 1.
    """
    h = h_euclidean.to(torch.float32)
    sqrt_c = curvature ** 0.5

    norm = torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
    # tanh(√c ‖v‖ / 2)
    scale = torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm)
    result = scale * h

    # 钳位：确保结果范数严格 < 1
    result_norm = torch.norm(result, dim=-1, keepdim=True)
    result = torch.where(
        result_norm >= max_norm,
        result * (max_norm / result_norm.clamp(min=1e-8)),
        result,
    )
    return result


def _poincare_distance(u: Tensor, v: Tensor, curvature: float = 1.0) -> Tensor:
    """计算庞加莱球上两点间的测地线距离。

    d(u, v) = (2/√c) · arctanh(√c · ‖(-u) ⊕_c v‖)

    简化实现 (使用恒等式):
        d(u, v) = (1/√c) · arccosh(1 + 2c · ‖u-v‖² / ((1-c‖u‖²)(1-c‖v‖²)))

    Args:
        u, v: 庞加莱球上的点, 形状 [..., dim].
        curvature: 曲率参数 c.

    Returns:
        测地线距离标量或张量.
    """
    c = curvature
    diff_sq = torch.sum((u - v) ** 2, dim=-1)
    u_sq = torch.sum(u ** 2, dim=-1)
    v_sq = torch.sum(v ** 2, dim=-1)

    denom = (1.0 - c * u_sq) * (1.0 - c * v_sq)
    denom = denom.clamp(min=1e-8)

    arg = 1.0 + 2.0 * c * diff_sq / denom
    # arccosh(x) = log(x + sqrt(x²-1)), 钳位 arg >= 1
    arg = arg.clamp(min=1.0 + 1e-8)

    return (1.0 / (c ** 0.5)) * torch.acosh(arg)


def hyperbolic_semantic_entropy(
    points_poincare: Tensor,
    curvature: float = 1.0,
) -> Tensor:
    """计算庞加莱球上一组点的双曲语义熵 (HSE)。
    Compute the Hyperbolic Semantic Entropy (HSE) for a set of points on the Poincaré ball.

    HSE 衡量一组语义表征在双曲空间中的分散程度
    HSE measures the dispersion of semantic representations
    in hyperbolic space:
        HSE = mean( d_hyp(p_i, centroid) )

    其中 centroid 使用爱因斯坦中点的简化近似
    centroid uses a simplified approximation of the Einstein midpoint
    (Euclidean mean followed by projection).

    Args:
        points_poincare: 庞加莱球上的点集, shape [W, D] or [W, B, D]
            Points on the Poincaré ball, W is window size.
        curvature: 曲率参数 c / Curvature parameter c.

    Returns:
        HSE 值 (>=0)。有 Batch 维则返回 [B]，否则返回标量。
        Returns [B] if input has Batch dim, otherwise scalar.
    """
    if points_poincare.shape[0] <= 1:
        # 如果是 [1, B, D]
        if points_poincare.ndim == 3:
            return torch.zeros(points_poincare.shape[1], device=points_poincare.device, dtype=torch.float32)
        return torch.tensor(0.0, device=points_poincare.device, dtype=torch.float32)

    points = points_poincare.to(torch.float32)

    # 简化中心点: 欧氏均值 → 投影回庞加莱球
    centroid_euclidean = points.mean(dim=0) # [B, D] or [D]

    # 增加假维度用于 poincare_map（原函数支持任意批次维度，直接传入即可）
    centroid = poincare_map(centroid_euclidean, curvature) # [B, D] or [D]

    # 将 centroid 扩展出 W 维度: [1, B, D] or [1, D]
    centroid_expanded = centroid.unsqueeze(0)

    # 批量计算测地线距离: _poincare_distance 支持广播
    distances = _poincare_distance(points, centroid_expanded, curvature) # [W, B] or [W]

    return distances.mean(dim=0)
