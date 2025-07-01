"""
Utility for deterministic *k* representatives per cluster.

Selection rules follow FFForge spec:
* label order from BIRCH preserves proximity to centroid.
* if k > len(cluster), fall back to sampling with replacement (warn).
"""

from __future__ import annotations
from collections import defaultdict
from typing import List
import numpy as np
import warnings


def stratified_pick(labels: np.ndarray, k: int = 1, seed: int = 42) -> List[int]:
    if k < 1:
        raise ValueError("k must be ≥1")

    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for idx, lab in enumerate(labels):
        buckets[int(lab)].append(idx)

    picks: List[int] = []
    for idxs in buckets.values():
        if k == 1:
            picks.append(idxs[0])
        else:
            if len(idxs) < k:
                warnings.warn(
                    f"Cluster size {len(idxs)} < k={k}; " "sampling with replacement."
                )
                idxs_sampled = rng.choice(idxs, size=k, replace=True)
            else:
                step = len(idxs) // k
                idxs_sampled = idxs[::step][:k]
            picks.extend(idxs_sampled)
    return sorted(picks)
