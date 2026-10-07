"""QuickSurf followed by connectivity, VMD fixed DDA, or fuzzy DDA."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import time
import numpy as np
from .classify import connectivity_partition
from .config import GridSpec, VolInteriorConfig
from .density import quicksurf_density_cpu, quicksurf_density_cuda
from .directions import make_directions
from .grid import make_grid
from .raycast import raycast_blocked_cpu, raycast_blocked_cuda, raycast_interior_cuda, raycast_partition_counts_cuda

DEFINITIONS = {"connectivity": "enclosed_free_6_material_preserving",
               "fixed": "vmd_fixed_2_0_1a1_material_overwrite",
               "fuzzy": "fuzzy_blocked_fraction_material_preserving"}


def minimum_blocked(cutoff, nrays):
    if cutoff is None or not np.isfinite(cutoff) or not 0 < cutoff <= 1:
        raise ValueError("fuzzy requires an explicit finite cutoff in (0, 1]")
    return int((Decimal(str(cutoff)) * nrays).to_integral_value(rounding=ROUND_CEILING))


@dataclass
class FrameResult:
    counts: dict
    grid: GridSpec
    labels: np.ndarray | None = None


def dda_partition(surface, directions, *, mode="fixed", cutoff=None, backend="cuda", return_labels=False):
    """Fixed DDA reproduces VMD's escaping-material overwrite.

    Fuzzy uses the explicit blocked-fraction criterion from the reanalysis;
    material stays intact. This does not claim bitwise parity with every
    version of VMD's probability-map discretizer.
    """
    if mode not in {"fixed", "fuzzy"} or backend not in {"cpu", "cuda"}:
        raise ValueError("invalid DDA mode or backend")
    rays = np.asarray(directions, dtype=np.float32)
    if rays.ndim != 2 or rays.shape[1] != 3 or not 1 <= len(rays) <= 65535:
        raise ValueError("directions must have shape (1..65535, 3)")
    if not np.isfinite(rays).all() or np.any(np.linalg.norm(rays, axis=1) == 0):
        raise ValueError("ray directions must be finite and nonzero")
    k = minimum_blocked(cutoff, len(rays)) if mode == "fuzzy" else len(rays)
    if backend == "cuda":
        import cupy as xp
        host = xp.asnumpy
    else:
        xp = np
        host = np.asarray
    material = xp.asarray(surface, dtype=bool)
    if mode == "fixed" and backend == "cuda" and not return_labels:
        interior, selection = raycast_partition_counts_cuda(material, rays, include_surface=True)
        return {"interior_voxels": interior, "selection_voxels": selection,
                "exterior_voxels": int(material.size) - interior - selection}, None
    if mode == "fixed" and backend == "cuda":
        inside = raycast_interior_cuda(material, rays, include_surface=True).astype(bool)
    else:
        raycast = raycast_blocked_cuda if backend == "cuda" else raycast_blocked_cpu
        blocked = raycast(material, rays, include_surface=mode == "fixed")
        inside = blocked >= k
    interior_mask = (~material) & inside
    selection_mask = material & inside if mode == "fixed" else material
    interior = int(xp.count_nonzero(interior_mask).item())
    selection = int(xp.count_nonzero(selection_mask).item())
    labels = host(xp.where(selection_mask, -5, xp.where(interior_mask, 0, 5)).astype(xp.int8)) if return_labels else None
    return {"interior_voxels": interior, "selection_voxels": selection,
            "exterior_voxels": int(material.size) - interior - selection}, labels


def measure_frame(coords_A, radii_A, masses_Da, *, method="connectivity", config=None,
                  cutoff=None, directions=None, return_labels=False):
    cfg = config or VolInteriorConfig(backend="cuda")
    if method not in DEFINITIONS or cfg.backend not in {"cpu", "cuda"}:
        raise ValueError("choose connectivity/fixed/fuzzy and an explicit cpu/cuda backend")
    if method == "fuzzy":
        minimum_blocked(cutoff, cfg.nrays)
    coords = np.asarray(coords_A, dtype=np.float32)
    radii = np.asarray(radii_A, dtype=np.float32)
    masses = np.asarray(masses_Da, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or not np.isfinite(coords).all():
        raise ValueError("finite coordinates with shape (n_atoms, 3) are required")
    for name, array in (("radii", radii), ("masses", masses)):
        if array.shape != (len(coords),) or not np.isfinite(array).all() or not (array > 0).all():
            raise ValueError(f"{name} must have one finite positive value per atom")
    started = time.perf_counter()
    grid = make_grid(coords, cfg, max_radius_A=float(radii.max()))
    center = np.average(coords, axis=0, weights=masses)
    seed = tuple(np.rint((center - grid.origin_A) / grid.spacing_A).astype(int))
    if cfg.backend == "cuda":
        import cupy as cp
        density = quicksurf_density_cuda(coords, radii, grid, cfg.radius_scale_A, cfg.gausslim,
                                         kernel_mode="cell_list")
        material = density >= np.float32(cfg.isovalue)
        del density
        cp.cuda.get_current_stream().synchronize()
    else:
        density = quicksurf_density_cpu(coords, radii, grid, cfg.radius_scale_A, cfg.gausslim)
        material = density >= np.float32(cfg.isovalue)
        del density
    density_done = time.perf_counter()
    raw_material = int(material.sum().item())
    if method == "connectivity":
        counts, labels = connectivity_partition(material, seed, backend=cfg.backend, return_labels=return_labels)
    else:
        if directions is None:
            directions = make_directions(cfg.nrays, cfg.ray_scheme, cfg.ray_seed, cfg.ray_candidates)
        if len(directions) != cfg.nrays:
            raise ValueError("ray count differs from config.nrays")
        counts, labels = dda_partition(material, directions, mode=method, cutoff=cutoff,
                                       backend=cfg.backend, return_labels=return_labels)
    if cfg.backend == "cuda":
        cp.cuda.get_current_stream().synchronize()
    finished = time.perf_counter()
    defaults = {"closure_pass": None, "center_in_material": None, "center_connected_to_boundary": None,
                "central_lumen_voxels": None, "other_interior_voxels": None,
                "interior_components": None, "failure_reason": ""}
    counts = {**defaults, **counts}
    counts.update(method=method, definition=DEFINITIONS[method], total_voxels=grid.n_voxels,
                  density_voxels=raw_material, cell_volume_A3=grid.voxel_volume_A3,
                  density_isovalue=cfg.isovalue, resolution=cfg.resolution, grid_spacing_A=cfg.spacing_A,
                  fuzzy_cutoff=cutoff if method == "fuzzy" else None,
                  minimum_blocked_rays=minimum_blocked(cutoff, cfg.nrays) if method == "fuzzy" else None,
                  nrays=cfg.nrays if method != "connectivity" else None,
                  grid_x=grid.shape[0], grid_y=grid.shape[1], grid_z=grid.shape[2],
                  origin_x_A=float(grid.origin_A[0]), origin_y_A=float(grid.origin_A[1]), origin_z_A=float(grid.origin_A[2]),
                  capsid_com_x_A=float(center[0]), capsid_com_y_A=float(center[1]), capsid_com_z_A=float(center[2]),
                  density_seconds=density_done-started, classification_seconds=finished-density_done,
                  frame_seconds=finished-started)
    counts["lumen_voxels"] = counts["interior_voxels"]
    counts["shell_voxels"] = counts["selection_voxels"]
    counts["outer_voxels"] = counts["lumen_voxels"] + counts["shell_voxels"]
    if counts["outer_voxels"] + counts["exterior_voxels"] != grid.n_voxels:
        raise RuntimeError("voxel partition does not conserve the grid")
    for compartment in ("lumen", "shell", "outer"):
        counts[compartment + "_volume_nm3"] = counts[compartment + "_voxels"] * grid.voxel_volume_A3 / 1000
    return FrameResult(counts, grid, labels)
