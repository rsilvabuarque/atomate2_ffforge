"""
Light wrapper around scikit-learn's BIRCH with input validation.
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import Birch
from typing import Tuple


def birch_cluster(
    pcs: np.ndarray,
    n_clusters: int,
    threshold: float = 1.5,
    branching_factor: int = 50,
    random_state: int = 42,
) -> Tuple[Birch, np.ndarray]:
    """
    Cluster whitened PCs with BIRCH.

    Returns fitted model and integer labels of length `pcs.shape[0]`.
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be ≥1")

    model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )
    labels = model.fit_predict(pcs, y=None)
    return model, labels.astype("int32")
