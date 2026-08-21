"""Portable tests that do not require the CUDA runtime or project trajectories."""

from __future__ import annotations

import numpy as np

from volinterior_cuda import VolInteriorConfig, infer_radii_from_names, measure_volinterior
from volinterior_cuda.directions import make_directions
from volinterior_cuda.grid import make_grid


def test_radii_inference_and_direction_determinism():
    radii = infer_radii_from_names(np.asarray(["CA", "N", "O", "CL"]))
    np.testing.assert_array_equal(radii, np.asarray([1.5, 1.4, 1.3, 1.75], dtype=np.float32))
    first = make_directions(8, "vmd_poisson", 512346, 40)
    second = make_directions(8, "vmd_poisson", 512346, 40)
    np.testing.assert_array_equal(first, second)
    assert first is not second


def test_selection_grid_and_cpu_masks_partition_voxels():
    coords = np.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    radii = np.full(len(coords), 1.0, dtype=np.float32)
    config = VolInteriorConfig(
        resolution=5.0,
        spacing_A=1.0,
        isovalue=0.5,
        nrays=4,
        mode="fixed",
        classifier="dda",
        backend="cpu",
        ray_scheme="fibonacci",
    )
    grid = make_grid(coords, config, max_radius_A=float(radii.max()))
    result = measure_volinterior(coords, radii, config=config, grid_override=grid)
    assert result.grid.n_voxels > 0
    assert np.count_nonzero(result.surface | result.interior | result.exterior) == result.grid.n_voxels
    assert not np.any(result.interior & result.exterior)
