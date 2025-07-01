"""
Principal-component analysis helpers with whitening and deterministic fit.

We rely on scikit-learn's PCA but expose a thin functional interface so
the rest of FFForge doesn't depend on scikit-learn objects.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def whiten_pca(
    features: np.ndarray,
    variance_cut: float = 0.9,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Standardise `features`, perform PCA, *whiten* retained PCs.

    Parameters
    ----------
    features
        Shape `(n_samples, n_features)`.
    variance_cut
        Retain enough PCs s.t. cumulative explained variance ≥ this value.
    random_state
        Ensures reproducible SVD ordering.

    Returns
    -------
    pcs
        Whitened principal components, shape `(n_samples, n_retained)`.
    pca
        Fitted PCA object (can be serialised if needed).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA(
        n_components=variance_cut,
        svd_solver="full",
        random_state=random_state,
    ).fit(X)

    pcs = pca.transform(X) * np.sqrt(pca.explained_variance_)
    return pcs.astype("float64"), pca
