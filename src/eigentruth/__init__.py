"""EigenTruth — 基于几何动力学与表征工程的大模型幻觉治理工具库。

Usage::

    from eigentruth.models.wrapper import EigenTruthWrapper

    safe_model = EigenTruthWrapper(base_model, target_layer_idx=-10)
    safe_model.warmup(fact_dataset, tokenizer)
    outputs = safe_model.generate(**inputs, max_new_tokens=100)
"""

__version__ = "0.1.0"

from eigentruth.core.math_engine import (
    TruthManifold,
    mahalanobis_distance,
    poincare_map,
    hyperbolic_semantic_entropy,
    sherman_morrison_update,
)
from eigentruth.intervention.hooks import TruthProbe
from eigentruth.models.wrapper import EigenTruthWrapper

__all__ = [
    "EigenTruthWrapper",
    "TruthProbe",
    "TruthManifold",
    "mahalanobis_distance",
    "poincare_map",
    "hyperbolic_semantic_entropy",
    "sherman_morrison_update",
    "__version__",
]
