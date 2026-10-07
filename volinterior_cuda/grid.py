"""Grid construction shared by CPU and CUDA paths."""

from __future__ import annotations

import numpy as np

from .config import GridSpec, VolInteriorConfig


def vmd_grid_padding_A(max_radius_A: float, radius_scale_A: float) -> float:
    """Return QuickSurf's bounding-box padding in Å.

    VMD first pads by ``1.70 * radscale * maxrad`` and then applies its
    volume-based padding heuristic to avoid clipping the Gaussian surface.
    """

    maxrad = np.float32(max_radius_A)
    radscale = np.float32(radius_scale_A)
    base = np.float32(1.70) * radscale * maxrad
    padrad = np.float32(0.65) * np.sqrt(
        np.float32(4.0 / 3.0 * np.pi) * base * base * base
    )
    return float(np.maximum(base, padrad))


def make_grid(coords_A: np.ndarray, config: VolInteriorConfig,
              max_radius_A: float | None = None) -> GridSpec:
    """Enclose the selected atoms using VMD QuickSurf padding and grid rounding."""
    coords = np.asarray(coords_A, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or not np.isfinite(coords).all():
        raise ValueError("coords_A must be finite, non-empty and have shape (n_atoms, 3)")
    padding = config.padding_A
    if padding is None:
        padding = vmd_grid_padding_A(1.5 if max_radius_A is None else max_radius_A,
                                    config.radius_scale_A)
    lo = coords.min(axis=0) - float(padding)
    hi = coords.max(axis=0) + float(padding)
    origin = lo.astype(np.float32)
    extent = (hi - lo).astype(np.float32)
    # VMD uses ceil(extent / spacing), with no extra +1 voxel.
    shape = tuple(np.maximum(1, np.ceil(extent / config.spacing_A).astype(np.int64)).tolist())
    return GridSpec(origin_A=origin, shape=shape, spacing_A=config.spacing_A)
