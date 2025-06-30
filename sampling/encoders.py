"""
Feature-extraction utilities for DIRECT sampling.

Every public encoder returns a **1-D NumPy array** (dtype=float64).
The registry `ENCODER_REGISTRY` maps a domain *tag* to its encoder.

Notes
-----
* If a pretrained M3GNet model (v0.1.x API) is unavailable, we fall back to a
  fast, deterministic composition+geometry vector so that unit tests and
  lightweight usage do not require TensorFlow downloads.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List

import numpy as np
from pymatgen.core import Structure

from dscribe.descriptors import SOAP
from ase import Atoms

# ---------------------------------------------------------------------------
# Optional M3GNet universal model
# ---------------------------------------------------------------------------
try:
    from m3gnet.models import M3GNet

    _M3GNET_HAS_PRETRAINED = hasattr(M3GNet, "from_pretrained")
except ModuleNotFoundError:  # m3gnet not installed
    M3GNet = None  # type: ignore
    _M3GNET_HAS_PRETRAINED = False

_M3GNET_LATENT = 128  # target length (matches legacy default)


@lru_cache(maxsize=1)
def _load_m3gnet() -> "M3GNet":
    """Load pretrained model once (if the helper is available)."""
    # mypy: ignore-next-line[attr-defined] because signature differs across versions
    return M3GNet.from_pretrained("M3GNet-MP-2023")  # type: ignore


def _composition_vector(struct: Structure, length: int = _M3GNET_LATENT) -> np.ndarray:
    """
    Deterministic fallback: element fractions + volume/atom → fixed vector.

    * First `len(ptable)` slots: composition histogram.
    * Next slot: volume per atom (Å³) / 100  — breaks ties for single-element cases.

    Parameters
    ----------
    struct
        Pymatgen Structure.
    length
        Desired vector length (pads with zeros).

    Returns
    -------
    vec
        1-D float64 array of length `length`.
    """
    ptable = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
    ]
    vec = np.zeros(length, dtype="float64")

    # --- composition histogram ------------------------------------------------
    for site in struct:
        sym = site.specie.symbol
        if sym in ptable:
            vec[ptable.index(sym)] += 1.0
    vec[: len(ptable)] /= len(struct)

    # --- geometric tie-breaker -------------------------------------------------
    v_atom = struct.volume / len(struct) / 100.0  # ≈ 0–10 range
    vec[len(ptable)] = v_atom

    return vec


def m3gnet_encoder(struct: Structure) -> np.ndarray:
    """
    Return a 128-D vector describing `struct`.

    * Uses the pretrained M3GNet latent if available.
    * Otherwise falls back to the composition+geometry vector.

    Parameters
    ----------
    struct
        Periodic Structure.

    Returns
    -------
    vec
        1-D float64 array (length 128).
    """
    if M3GNet is not None and _M3GNET_HAS_PRETRAINED:
        try:
            _, _, _, latent = _load_m3gnet().predict_structure(
                struct, return_latent=True
            )
            return latent.astype("float64")
        except Exception:
            pass  # runtime/TensorFlow error → fallback

    return _composition_vector(struct)


# ---------------------------------------------------------------------------
# SOAP encoder for non-periodic molecular clusters
# ---------------------------------------------------------------------------

_SOAP: SOAP | None = None


def _init_soap(species: List[str]) -> SOAP:
    global _SOAP
    if _SOAP is None:
        _SOAP = SOAP(
            species=species,
            periodic=False,
            rcut=5.0,
            nmax=8,
            lmax=6,
            sparse=False,
        )
    return _SOAP


def soap_encoder(struct: Structure) -> np.ndarray:
    """Return averaged SOAP power spectrum (≈256–512 D) for a cluster."""
    ase_obj: Atoms = struct.to(fmt="ase")
    soap_mat = _init_soap(species=list({site.specie.symbol for site in struct})).create(
        ase_obj
    )
    return soap_mat.mean(axis=0).astype("float64")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
EncoderFn = Callable[[Structure], np.ndarray]

ENCODER_REGISTRY: Dict[str, EncoderFn] = {
    "battery": m3gnet_encoder,
    "polar_perovskite": m3gnet_encoder,
    "surface_cat": m3gnet_encoder,
    "polymer": soap_encoder,  # placeholder; will switch to mol-GNN later
    "electrolyte": soap_encoder,
}
