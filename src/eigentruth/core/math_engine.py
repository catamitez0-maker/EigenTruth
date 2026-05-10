"""EigenTruth Core — 防崩溃数学引擎。

基于几何动力学的底层数学原语，包括：
- Sherman-Morrison 在线协方差逆更新
- 马氏距离计算
- 庞加莱球模型映射
- 双曲语义熵 (HSE)

所有浮点密集运算在内部强制使用 FP32 以确保数值稳定性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TruthManifold:
    """真值流形：存储事实语料的统计特征。

    Attributes:
        mean: 隐状态质心向量，形状 [hidden_dim].
        cov_inv: 协方差矩阵的逆，形状 [hidden_dim, hidden_dim].
        n: 已累积的样本数量.
        hidden_dim: 隐状态维度 (由首次更新时自动推断).
    """

    mean: Optional[Tensor] = None
    cov_inv: Optional[Tensor] = None
    n: int = 0
    hidden_dim: int = 0

    # 运行时设备跟踪（不参与序列化）
    _device: torch.device = field(default_factory=lambda: torch.device("cpu"), repr=False)

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def update(self, h: Tensor, epsilon: float = 1e-6) -> None:
        """用单个隐状态向量增量更新流形。

        首次调用时自动初始化 mean 和 cov_inv。

        Args:
            h: 隐状态向量, 形状 [hidden_dim].
            epsilon: Sherman-Morrison 分母正则项.
        """
        h = h.detach()

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

        self.n += 1
        # 增量均值更新 (Welford)
        self.mean = self.mean + (h.to(torch.float32) - self.mean) / self.n

        # 增量协方差逆更新
        delta = h.to(torch.float32) - self.mean
        self.cov_inv = sherman_morrison_update(self.cov_inv, delta, epsilon)

    def is_ready(self) -> bool:
        """流形至少经过 2 个样本后方可使用。"""
        return self.n >= 2 and self.mean is not None and self.cov_inv is not None


# ---------------------------------------------------------------------------
# 核心数学函数
# ---------------------------------------------------------------------------

def sherman_morrison_update(
    cov_inv: Tensor,
    x: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Sherman-Morrison 秩-1 在线更新协方差逆矩阵。

    公式: A⁻¹_new = A⁻¹ - (A⁻¹ x xᵀ A⁻¹) / (1 + xᵀ A⁻¹ x + ε)

    内部所有运算强制使用 FP32 以防止 FP16 数值崩溃。

    Args:
        cov_inv: 当前协方差逆矩阵, 形状 [d, d].
        x: 增量向量 (h - mean), 形状 [d].
        epsilon: 分母正则项, 防止除零.

    Returns:
        更新后的协方差逆矩阵, 形状 [d, d], 与输入同精度.
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

    D_M(h) = sqrt( (h - μ)ᵀ Σ⁻¹ (h - μ) )

    Args:
        h: 隐状态向量, 形状 [hidden_dim].
        mean: 质心向量, 形状 [hidden_dim].
        cov_inv: 协方差逆矩阵, 形状 [hidden_dim, hidden_dim].

    Returns:
        马氏距离标量 (>=0).
    """
    # 强制 FP32
    delta = (h - mean).to(torch.float32)
    cov_inv_f32 = cov_inv.to(torch.float32)

    # δᵀ Σ⁻¹ δ
    m_sq = delta @ cov_inv_f32 @ delta
    # clamp 防止浮点误差导致 sqrt 负数
    return torch.sqrt(torch.clamp(m_sq, min=0.0))


def poincare_map(
    h_euclidean: Tensor,
    curvature: float = 1.0,
    max_norm: float = 0.999,
) -> Tensor:
    """将欧氏空间隐状态映射到庞加莱球模型。

    使用指数映射 (exponential map at origin):
        exp_0(v) = tanh(√c ‖v‖ / 2) · v / (√c ‖v‖)

    Args:
        h_euclidean: 欧氏空间向量, 形状 [..., dim].
        curvature: 负曲率参数 c (正值).
        max_norm: 钳位最大范数, 保持在开球内部 (< 1).

    Returns:
        庞加莱球坐标, 形状 [..., dim], 范数 < 1.
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
        u, v: 庞加莱球上的点, 形状 [dim].
        curvature: 曲率参数 c.

    Returns:
        测地线距离标量.
    """
    c = curvature
    diff_sq = torch.sum((u - v) ** 2)
    u_sq = torch.sum(u ** 2)
    v_sq = torch.sum(v ** 2)

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

    HSE 衡量一组语义表征在双曲空间中的分散程度：
        HSE = mean( d_hyp(p_i, centroid) )

    其中 centroid 使用爱因斯坦中点的简化近似（欧氏均值后投影）。

    Args:
        points_poincare: 庞加莱球上的点集, 形状 [N, dim].
        curvature: 曲率参数 c.

    Returns:
        HSE 标量值 (>=0). 点越分散, HSE 越高.
    """
    if points_poincare.shape[0] <= 1:
        return torch.tensor(0.0, device=points_poincare.device, dtype=torch.float32)

    points = points_poincare.to(torch.float32)

    # 简化中心点: 欧氏均值 → 投影回庞加莱球
    centroid_euclidean = points.mean(dim=0)
    centroid = poincare_map(centroid_euclidean.unsqueeze(0), curvature).squeeze(0)

    # 计算每个点到中心点的测地线距离
    distances = torch.stack([
        _poincare_distance(p, centroid, curvature) for p in points
    ])

    return distances.mean()
