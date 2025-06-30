"""
Public API: sample_direct(structures, tag, dft_budget, k=1, **kw)

Ties together encoders → PCA → BIRCH → stratified pick.
"""

from __future__ import annotations

from typing import List

import numpy as np
from pymatgen.core import Structure

from .encoders import ENCODER_REGISTRY
from .pca import whiten_pca
from .birch import birch_cluster
from .stratified import stratified_pick

# ---------------------------------------------------------------------------
# Domain-specific hyperparameters
# ---------------------------------------------------------------------------
TAG2VAR = {
    "battery": 0.95,
    "polar_perovskite": 0.98,
    "surface_cat": 0.98,
    "polymer": 0.90,
    "electrolyte": 0.97,
}
RADIUS = {
    "battery": 0.05,  # tighter to resolve many Li bcc variants
    "surface_cat": 0.8,
    "electrolyte": 1.2,
    "__default__": 1.5,
}


def sample_direct(
    structures: List[Structure],
    tag: str,
    dft_budget: int,
    k: int = 1,
    seed: int = 42,
) -> List[int]:
    """
    Return indices of structures selected by DIRECT.

    Parameters
    ----------
    structures
        List of pymatgen Structures.
    tag
        Domain tag key into `ENCODER_REGISTRY`.
    dft_budget
        Intended number of DFT single-point calculations.
        We cluster into `dft_budget // k` centroids.
    k
        Representatives per cluster.
    seed
        RNG seed (affects tie-breakers when k>1).
    """
    if tag not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown tag '{tag}'. Available: {list(ENCODER_REGISTRY)}")

    # — encode —
    feats = np.vstack([ENCODER_REGISTRY[tag](s) for s in structures])

    # — PCA + whitening —
    pcs, _ = whiten_pca(feats, variance_cut=TAG2VAR.get(tag, 0.95))

    # — cluster —
    n_clusters = max(1, dft_budget // k)
    _, labels = birch_cluster(
        pcs,
        n_clusters=n_clusters,
        threshold=RADIUS.get(tag, RADIUS["__default__"]),
    )

    # — pick representatives —
    return stratified_pick(labels, k=k, seed=seed)
