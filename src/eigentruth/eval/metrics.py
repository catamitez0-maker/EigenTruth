"""EigenTruth eval metrics — 与模型无关的评分指标 / Model-free scoring metrics.

这些是对分数/表征的纯函数（无模型、无数据集依赖），可在 CPU 上离线单元测试。
Pure functions over scores/representations (no model or dataset deps); CPU-testable offline.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor

ArrayLike = Union[Tensor, Sequence[float]]


def _average_ranks(x: Tensor) -> Tensor:
    """返回 1-based 排名，平局取平均排名（等价于 scipy.stats.rankdata 的 'average'）。
    Return 1-based ranks with ties resolved to the average rank (like scipy rankdata).
    """
    n = x.numel()
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and bool(sorted_x[j + 1] == sorted_x[i]):
            j += 1
        # i..j 是一组平局，取 1-based 排名 (i+1)..(j+1) 的平均
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """计算 ROC 曲线下面积 (AUROC)。
    Area under the ROC curve.

    约定：label 1 = 正类（例如"假陈述/幻觉"，即希望被高分标记的对象），label 0 = 负类。
    分数越高 => 越倾向正类。平局按平均排名处理（Mann–Whitney U 等价式）。
    Convention: label 1 = positive class (e.g. the false/hallucinated item we want to
    flag with a high score), label 0 = negative. Higher score => more positive. Ties are
    handled via average ranks (equivalent to the Mann–Whitney U statistic).

    Args:
        scores: 每个样本的分数 / per-item scores, shape [N].
        labels: 0/1 标签 / binary labels in {0, 1}, shape [N].

    Returns:
        AUROC ∈ [0, 1]；若某一类缺失则返回 float('nan')。
        AUROC in [0, 1]; returns float('nan') if either class is absent.
    """
    scores_t = torch.as_tensor(scores, dtype=torch.float64).flatten()
    labels_t = torch.as_tensor(labels, dtype=torch.float64).flatten()
    if scores_t.numel() != labels_t.numel():
        raise ValueError("scores and labels must have the same length.")

    pos_mask = labels_t == 1
    neg_mask = labels_t == 0
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores_t)
    sum_ranks_pos = float(ranks[pos_mask].sum().item())
    # AUROC = (R+ - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def euclidean_dispersion(points: Tensor) -> Tensor:
    """一组点到其质心的平均欧氏距离（双曲 HSE 的欧氏对照基线）。
    Mean Euclidean distance of points to their centroid — the Euclidean counterpart
    of `hyperbolic_semantic_entropy`, used for the "does hyperbolic help?" ablation.

    Args:
        points: 点集 / point set, shape [N, D].

    Returns:
        标量张量 (>=0)；N<=1 时为 0 / scalar tensor (>=0); 0 when N <= 1.
    """
    points = points.to(torch.float32)
    if points.shape[0] <= 1:
        return torch.tensor(0.0)
    centroid = points.mean(dim=0, keepdim=True)
    return torch.norm(points - centroid, dim=-1).mean()
