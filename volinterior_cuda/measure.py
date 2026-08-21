"""Top-level orchestration for the CUDA/CPU volinterior implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .classify import fixed_connectivity_cpu, fixed_connectivity_cuda, surface_mask
from .config import GridSpec, VolInteriorConfig
from .density import cupy_available, quicksurf_density_cpu, quicksurf_density_cuda
from .directions import make_directions
from .grid import make_grid
from .raycast import (
    raycast_blocked_cpu,
    raycast_blocked_cuda,
    raycast_interior_count_cuda,
    raycast_interior_cuda,
)


@dataclass
class VolInteriorResult:
    """Classification result returned on host memory for easy serialization."""

    grid: GridSpec
    surface: np.ndarray
    interior: np.ndarray
    exterior: np.ndarray
    density: np.ndarray | None
    probability: np.ndarray | None
    blocked_rays: np.ndarray | None
    backend: str
    config: dict[str, object]
    directions: np.ndarray | None

    @property
    def boundary(self) -> np.ndarray:
        return self.surface

    @property
    def interior_voxels(self) -> int:
        return int(np.count_nonzero(self.interior))

    @property
    def exterior_voxels(self) -> int:
        return int(np.count_nonzero(self.exterior))

    @property
    def interior_volume_A3(self) -> float:
        return self.interior_voxels * self.grid.voxel_volume_A3

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "grid": self.grid.as_dict(),
            "interior_voxels": self.interior_voxels,
            "exterior_voxels": self.exterior_voxels,
            "surface_voxels": int(np.count_nonzero(self.surface)),
            "interior_volume_A3": self.interior_volume_A3,
            "config": self.config,
        }



@dataclass
class VolInteriorCountsResult:
    """Scalar fixed-mode result that keeps voxel masks on the GPU."""

    grid: GridSpec
    interior_voxels: int
    surface_voxels: int
    exterior_voxels: int
    backend: str
    config: dict[str, object]

    @property
    def total_voxels(self) -> int:
        return self.grid.n_voxels

    @property
    def interior_volume_A3(self) -> float:
        return self.interior_voxels * self.grid.voxel_volume_A3

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "grid": self.grid.as_dict(),
            "total_voxels": self.total_voxels,
            "interior_voxels": self.interior_voxels,
            "surface_voxels": self.surface_voxels,
            "exterior_voxels": self.exterior_voxels,
            "interior_volume_A3": self.interior_volume_A3,
            "config": self.config,
        }

def _resolve_backend(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cupy_available():
            raise RuntimeError("backend='cuda' requested, but CuPy/CUDA is unavailable")
        return "cuda"
    if requested == "auto":
        return "cuda" if cupy_available() else "cpu"
    raise ValueError(f"unknown backend: {requested}")


def measure_volinterior(
    coords_A: np.ndarray,
    radii_A: np.ndarray,
    *,
    box_A: np.ndarray | None = None,
    config: VolInteriorConfig | None = None,
    return_density: bool = False,
    return_probability: bool = False,
    grid_override: GridSpec | None = None,
) -> VolInteriorResult:
    """Measure interior volume for one coordinate frame.

    For ``classifier='dda'`` the result follows the paper/VMD rule: a ray is
    *blocked* when it hits an isosurface voxel and *unblocked* when it reaches
    the grid boundary.  Fixed mode labels a voxel interior only when every ray
    is blocked.  Fuzzy mode returns the blocked-ray probability and applies
    ``fuzzy_cutoff`` to produce a binary interior mask.

    ``classifier='connectivity'`` is an exact grid-topology alternative for
    fixed mode and is faster for large grids; it does not depend on ``nrays``.
    Fixed mode defaults to the all-rays-blocked mask without materializing the
    full blocked-ray probability map. Set ``return_probability=True`` when the
    per-voxel ray count is needed; fuzzy mode still computes it internally.
    """

    cfg = config or VolInteriorConfig()
    backend = _resolve_backend(cfg.backend)
    coords = np.asarray(coords_A, dtype=np.float32)
    radii = np.asarray(radii_A, dtype=np.float32)
    if radii.ndim != 1 or radii.shape[0] != coords.shape[0] or np.any(radii <= 0):
        raise ValueError("radii_A must have one positive value per coordinate")
    grid = (
        grid_override
        if grid_override is not None
        else make_grid(coords, cfg, box_A=box_A, max_radius_A=float(np.max(radii)))
    )

    if backend == "cuda":
        import cupy as cp

        density_device = quicksurf_density_cuda(
            coords,
            radii,
            grid,
            cfg.radius_scale_A,
            cfg.gausslim,
            kernel_mode=cfg.density_kernel,
            accel_grid_spacing_A=cfg.accel_grid_spacing_A,
        )
        surface_device = surface_mask(density_device, cfg.isovalue)
        density = cp.asnumpy(density_device) if return_density else None
        surface = cp.asnumpy(surface_device)
    else:
        density_host = quicksurf_density_cpu(
            coords, radii, grid, cfg.radius_scale_A, cfg.gausslim
        )
        surface_host = surface_mask(density_host, cfg.isovalue)
        density = density_host if return_density else None
        surface = np.asarray(surface_host, dtype=bool)

    directions: np.ndarray | None = None
    blocked_host: np.ndarray | None = None
    probability: np.ndarray | None = None

    if cfg.mode == "fixed" and cfg.classifier == "connectivity":
        if backend == "cuda":
            interior_device, exterior_device = fixed_connectivity_cuda(surface_device)
            interior = cp.asnumpy(interior_device)
            exterior = cp.asnumpy(exterior_device)
        else:
            interior, exterior = fixed_connectivity_cpu(surface)
    else:
        directions = make_directions(cfg.nrays, cfg.ray_scheme, cfg.ray_seed, cfg.ray_candidates)
        free = ~surface
        if backend == "cuda" and cfg.mode == "fixed" and not return_probability:
            # The volume result only needs the all-rays-blocked predicate.
            # This kernel exits as soon as one ray reaches the boundary.
            interior_device = raycast_interior_cuda(surface_device, directions)
            interior = free & cp.asnumpy(interior_device).astype(bool)
            blocked_host = None
            probability = None
        else:
            if backend == "cuda":
                blocked_device = raycast_blocked_cuda(surface_device, directions)
                blocked_host = cp.asnumpy(blocked_device)
            else:
                blocked_host = raycast_blocked_cpu(surface, directions)
            probability = blocked_host.astype(np.float32) / float(cfg.nrays)
            if cfg.mode == "fixed":
                interior = free & (blocked_host == cfg.nrays)
            else:
                interior = free & (probability >= cfg.fuzzy_cutoff)
        exterior = free & ~interior

    return VolInteriorResult(
        grid=grid,
        surface=surface,
        interior=np.asarray(interior, dtype=bool),
        exterior=np.asarray(exterior, dtype=bool),
        density=density,
        probability=probability if return_probability else None,
        blocked_rays=blocked_host,
        backend=backend,
        config=cfg.as_dict(),
        directions=directions,
    )



def measure_volinterior_counts(
    coords_A: np.ndarray,
    radii_A: np.ndarray,
    *,
    box_A: np.ndarray | None = None,
    config: VolInteriorConfig | None = None,
) -> VolInteriorCountsResult:
    """Measure fixed DDA volume while returning only scalar voxel counts.

    The CUDA path leaves density/surface/interior masks on the device and
    copies back only the three scalar counts. This is intended for trajectory
    production runs that do not need diagnostic maps or ray probabilities.
    """

    cfg = config or VolInteriorConfig()
    backend = _resolve_backend(cfg.backend)
    if cfg.mode != "fixed" or cfg.classifier != "dda":
        raise ValueError("measure_volinterior_counts requires mode='fixed' and classifier='dda'")
    coords = np.asarray(coords_A, dtype=np.float32)
    radii = np.asarray(radii_A, dtype=np.float32)
    if radii.ndim != 1 or radii.shape[0] != coords.shape[0] or np.any(radii <= 0):
        raise ValueError("radii_A must have one positive value per coordinate")

    if backend != "cuda":
        result = measure_volinterior(
            coords,
            radii,
            box_A=box_A,
            config=cfg,
            return_density=False,
            return_probability=False,
        )
        return VolInteriorCountsResult(
            grid=result.grid,
            interior_voxels=result.interior_voxels,
            surface_voxels=int(np.count_nonzero(result.surface)),
            exterior_voxels=result.exterior_voxels,
            backend=result.backend,
            config=result.config,
        )

    import cupy as cp

    grid = make_grid(
        coords,
        cfg,
        box_A=box_A,
        max_radius_A=float(np.max(radii)),
    )
    density_device = quicksurf_density_cuda(
        coords,
        radii,
        grid,
        cfg.radius_scale_A,
        cfg.gausslim,
        kernel_mode=cfg.density_kernel,
        accel_grid_spacing_A=cfg.accel_grid_spacing_A,
    )
    surface_device = surface_mask(density_device, cfg.isovalue)
    directions = make_directions(
        cfg.nrays,
        cfg.ray_scheme,
        cfg.ray_seed,
        cfg.ray_candidates,
    )
    interior_voxels = raycast_interior_count_cuda(surface_device, directions)
    surface_voxels = int(cp.count_nonzero(surface_device).get())
    exterior_voxels = grid.n_voxels - interior_voxels - surface_voxels
    return VolInteriorCountsResult(
        grid=grid,
        interior_voxels=interior_voxels,
        surface_voxels=surface_voxels,
        exterior_voxels=exterior_voxels,
        backend=backend,
        config=cfg.as_dict(),
    )
