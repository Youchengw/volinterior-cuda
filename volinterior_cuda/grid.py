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


def make_grid(
    coords_A: np.ndarray,
    config: VolInteriorConfig,
    box_A: np.ndarray | None = None,
    max_radius_A: float | None = None,
) -> GridSpec:
    """Build a grid around the selection or around an explicit simulation box.

    The VMD path uses the current molecule's bounding box.  ``domain='box'``
    therefore requires ``box_A`` and is the closest compatibility mode.  The
    default ``domain='selection'`` adds a conservative padding around the
    selected atoms, which is useful when a topology has no unit-cell metadata.
    """

    coords = np.asarray(coords_A, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError("coords_A must have shape (n_atoms, 3) and be non-empty")

    if config.domain == "box":
        if box_A is None:
            raise ValueError("box_A is required when domain='box'")
        box = np.asarray(box_A, dtype=np.float32)
        if box.shape != (3,) or np.any(box <= 0):
            raise ValueError("box_A must contain three positive lengths")
        origin = np.zeros(3, dtype=np.float32)
        extent = box
    else:
        padding = config.padding_A
        if padding is None:
            if max_radius_A is None:
                # Keep the standalone helper usable; measure_volinterior passes
                # the actual selected-atom maximum radius.
                max_radius_A = 1.5
            padding = vmd_grid_padding_A(max_radius_A, config.radius_scale_A)
        lo = coords.min(axis=0) - float(padding)
        hi = coords.max(axis=0) + float(padding)
        origin = lo.astype(np.float32)
        extent = (hi - lo).astype(np.float32)

    # VMD uses ceil(extent / spacing), with no extra ``+1`` voxel.
    shape = tuple(np.maximum(1, np.ceil(extent / config.spacing_A).astype(np.int64)).tolist())
    return GridSpec(origin_A=origin, shape=shape, spacing_A=config.spacing_A)
