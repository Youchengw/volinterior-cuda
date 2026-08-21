"""Atom-radius helpers.

VMD's QuickSurf receives per-atom radii from the molecule representation.  A
trajectory adapter should use topology-provided radii whenever available; the
small table here is a deterministic fallback for common protein/ion elements.
"""

from __future__ import annotations

import re

import numpy as np


VMD_ATOM_RADII_A: dict[str, float] = {
    # These are VMD's default atom radii observed for the canonical .gro
    # topology (and used by QuickSurf when no per-atom radius is supplied).
    "H": 1.00,
    "C": 1.50,
    "N": 1.40,
    "O": 1.30,
    "F": 1.47,
    "P": 1.90,
    "S": 1.90,
    "CL": 1.75,
    "NA": 2.27,
    "K": 2.75,
    "MG": 1.73,
    "CA": 2.31,
    "ZN": 1.39,
}

# Backward-compatible public name; the fallback is intentionally VMD-like.
VDW_RADII_A = VMD_ATOM_RADII_A



def infer_element_from_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", str(name)).upper()
    if cleaned.startswith("CL"):
        return "CL"
    if cleaned.startswith("NA"):
        return "NA"
    if cleaned.startswith("MG"):
        return "MG"
    # Atom names such as CA (alpha carbon) are carbon unless the topology
    # explicitly provides an element of calcium.
    return cleaned[:1] if cleaned else ""


def infer_radii_from_names(names: list[str] | np.ndarray, elements: list[str] | np.ndarray | None = None) -> np.ndarray:
    """Return VDW radii in Å, raising on unknown elements instead of guessing."""

    if elements is None:
        elems = [infer_element_from_name(name) for name in names]
    else:
        elems = [str(e).strip().upper() for e in elements]
        if len(elems) != len(names):
            raise ValueError("elements and names must have equal length")
        elems = [
            e if e in VMD_ATOM_RADII_A else infer_element_from_name(name)
            for e, name in zip(elems, names, strict=True)
        ]

    unknown = sorted({e for e in elems if e not in VDW_RADII_A})
    if unknown:
        raise ValueError(f"unknown element(s) without a radius: {unknown}")
    return np.asarray([VDW_RADII_A[e] for e in elems], dtype=np.float32)
