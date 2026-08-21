"""Surface, connectivity, and ray-based interior classification."""

from __future__ import annotations

import numpy as np


def six_connected_structure() -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    return structure


def surface_mask(density, isovalue: float):
    return density >= np.float32(isovalue)


def fixed_connectivity_cpu(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classify free voxels by whether they connect to a grid boundary."""

    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - project dependency normally exists
        raise RuntimeError("CPU connectivity classification requires scipy") from exc
    free = ~np.asarray(surface, dtype=bool)
    labels, _ = ndimage.label(free, structure=six_connected_structure())
    boundary_labels = np.unique(
        np.concatenate(
            (
                labels[0, :, :].ravel(),
                labels[-1, :, :].ravel(),
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            )
        )
    )
    exterior = np.isin(labels, boundary_labels) & free
    interior = free & ~exterior
    return interior, exterior


def fixed_connectivity_cuda(surface):
    """GPU connected-component classification via ``cupyx.scipy.ndimage``."""

    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cndimage
    except ImportError as exc:  # pragma: no cover - depends on local CUDA env
        raise RuntimeError("CUDA connectivity requires cupy and cupyx") from exc
    free = ~surface.astype(cp.bool_)
    structure = cp.asarray(six_connected_structure())
    labels, _ = cndimage.label(free, structure=structure)
    boundary_labels = cp.unique(
        cp.concatenate(
            (
                labels[0, :, :].ravel(),
                labels[-1, :, :].ravel(),
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            )
        )
    )
    exterior = cp.isin(labels, boundary_labels) & free
    interior = free & ~exterior
    return interior, exterior
