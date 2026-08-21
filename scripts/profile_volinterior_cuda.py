#!/usr/bin/env python3
"""Profile the existing Python/CuPy CUDA volinterior pipeline.

The profiler intentionally keeps the production implementation unchanged. It
times trajectory extraction, grid construction, density (including the cell
list preparation and CUDA kernel launch), surface thresholding, direction
generation, DDA classification, and the final surface reduction separately.
CUDA events report device time while ``perf_counter`` reports the end-to-end
wall time for each stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from volinterior_cuda import VolInteriorConfig, infer_radii_from_names
from volinterior_cuda.classify import surface_mask
from volinterior_cuda.density import quicksurf_density_cuda
from volinterior_cuda.directions import make_directions
from volinterior_cuda.grid import make_grid
from volinterior_cuda.raycast import raycast_interior_count_cuda


CSV_FIELDS = [
    "repeat",
    "local_frame",
    "full_frame",
    "trajectory_extract_ms",
    "grid_ms",
    "density_wall_ms",
    "density_gpu_ms",
    "surface_mask_wall_ms",
    "surface_mask_gpu_ms",
    "directions_ms",
    "dda_wall_ms",
    "dda_gpu_ms",
    "surface_reduce_wall_ms",
    "surface_reduce_gpu_ms",
    "pipeline_ms",
    "total_voxels",
    "surface_voxels",
    "interior_voxels",
    "exterior_voxels",
    "grid_x",
    "grid_y",
    "grid_z",
    "origin_x_A",
    "origin_y_A",
    "origin_z_A",
    "voxel_volume_A3",
    "density_kernel",
    "nrays",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--frames", default="0")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--selection", default="protein")
    parser.add_argument("--resolution", type=float, default=12.5)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--isovalue", type=float, default=0.5)
    parser.add_argument("--nrays", type=int, default=32)
    parser.add_argument("--ray-scheme", choices=("vmd_poisson", "poisson", "fibonacci"), default="vmd_poisson")
    parser.add_argument("--ray-seed", type=int, default=512346)
    parser.add_argument("--density-kernel", choices=("cell_list", "atom"), default="cell_list")
    parser.add_argument("--dda-layout", choices=("1d", "3d"), default="3d")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_frames(value: str) -> list[int]:
    frames = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not frames or frames[0] < 0:
        raise ValueError("--frames must contain non-negative integers")
    return frames


def _config(args: argparse.Namespace) -> VolInteriorConfig:
    return VolInteriorConfig(
        resolution=args.resolution,
        spacing_A=args.spacing,
        isovalue=args.isovalue,
        nrays=args.nrays,
        mode="fixed",
        classifier="dda",
        backend="cuda",
        ray_scheme=args.ray_scheme,
        ray_seed=args.ray_seed,
        density_kernel=args.density_kernel,
    )


def _radii_from_atoms(atoms) -> np.ndarray:
    elements = getattr(atoms, "elements", None)
    if elements is not None:
        elements = np.asarray(elements)
    return infer_radii_from_names(np.asarray(atoms.names), elements)


def _gpu_stage(cp, fn: Callable[[], Any]) -> tuple[Any, float, float]:
    """Run ``fn`` and return (value, wall_ms, CUDA-event_ms)."""

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    wall_start = time.perf_counter()
    start_event.record()
    value = fn()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    gpu_ms = float(cp.cuda.get_elapsed_time(start_event, end_event))
    return value, wall_ms, gpu_ms


def _read_reference(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["local_frame"]): row for row in csv.DictReader(handle)}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        "pipeline_ms",
        "trajectory_extract_ms",
        "grid_ms",
        "density_wall_ms",
        "density_gpu_ms",
        "surface_mask_wall_ms",
        "surface_mask_gpu_ms",
        "directions_ms",
        "dda_wall_ms",
        "dda_gpu_ms",
        "surface_reduce_wall_ms",
        "surface_reduce_gpu_ms",
    ]
    summary: dict[str, Any] = {"row_count": len(rows), "stages_ms": {}}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows]
        summary["stages_ms"][field] = {
            "median": float(statistics.median(values)),
            "mean": float(statistics.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    pipeline_median = summary["stages_ms"]["pipeline_ms"]["median"]
    summary["wall_time_fraction_of_pipeline"] = {
        field: summary["stages_ms"][field]["median"] / pipeline_median
        for field in (
            "trajectory_extract_ms",
            "grid_ms",
            "density_wall_ms",
            "surface_mask_wall_ms",
            "directions_ms",
            "dda_wall_ms",
            "surface_reduce_wall_ms",
        )
    }
    return summary


def main() -> int:
    args = _parser().parse_args()
    args.topology = args.topology.resolve()
    args.trajectory = args.trajectory.resolve()
    if args.reference_csv is not None:
        args.reference_csv = args.reference_csv.resolve()
    output = args.output.resolve()
    frames = _parse_frames(args.frames)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")
    if not args.topology.exists() or not args.trajectory.exists():
        raise FileNotFoundError("topology and trajectory must exist")
    output.mkdir(parents=True, exist_ok=True)

    import cupy as cp
    import MDAnalysis as mda

    cfg = _config(args)
    universe = mda.Universe(str(args.topology), str(args.trajectory))
    if max(frames) >= len(universe.trajectory):
        raise ValueError(f"requested frame {max(frames)} exceeds trajectory length {len(universe.trajectory)}")
    atoms = universe.select_atoms(args.selection)
    if len(atoms) == 0:
        raise ValueError(f"selection returned no atoms: {args.selection}")
    radii = _radii_from_atoms(atoms)

    # Compile/cache all RawKernels before recording profile rows.
    universe.trajectory[frames[0]]
    warmup_coords = np.asarray(atoms.positions, dtype=np.float32).copy()
    warmup_grid = make_grid(warmup_coords, cfg, max_radius_A=float(np.max(radii)))
    warmup_density = quicksurf_density_cuda(
        warmup_coords,
        radii,
        warmup_grid,
        cfg.radius_scale_A,
        cfg.gausslim,
        kernel_mode=cfg.density_kernel,
        accel_grid_spacing_A=cfg.accel_grid_spacing_A,
    )
    warmup_surface = surface_mask(warmup_density, cfg.isovalue)
    warmup_directions = make_directions(cfg.nrays, cfg.ray_scheme, cfg.ray_seed, cfg.ray_candidates)
    raycast_interior_count_cuda(warmup_surface, warmup_directions, layout=args.dda_layout)
    cp.cuda.Stream.null.synchronize()
    del warmup_density, warmup_surface

    rows: list[dict[str, Any]] = []
    for frame in frames:
        universe.trajectory[frame]
        extract_start = time.perf_counter()
        coords = np.asarray(atoms.positions, dtype=np.float32).copy()
        extract_ms = (time.perf_counter() - extract_start) * 1000.0
        for repeat in range(args.repeats):
            pipeline_start = time.perf_counter()
            grid_start = time.perf_counter()
            grid = make_grid(coords, cfg, max_radius_A=float(np.max(radii)))
            grid_ms = (time.perf_counter() - grid_start) * 1000.0

            density, density_wall_ms, density_gpu_ms = _gpu_stage(
                cp,
                lambda: quicksurf_density_cuda(
                    coords,
                    radii,
                    grid,
                    cfg.radius_scale_A,
                    cfg.gausslim,
                    kernel_mode=cfg.density_kernel,
                    accel_grid_spacing_A=cfg.accel_grid_spacing_A,
                ),
            )
            surface, surface_wall_ms, surface_gpu_ms = _gpu_stage(
                cp, lambda: surface_mask(density, cfg.isovalue)
            )
            directions_start = time.perf_counter()
            directions = make_directions(cfg.nrays, cfg.ray_scheme, cfg.ray_seed, cfg.ray_candidates)
            directions_ms = (time.perf_counter() - directions_start) * 1000.0

            interior_device_count, dda_wall_ms, dda_gpu_ms = _gpu_stage(
                cp, lambda: raycast_interior_count_cuda(
                    surface, directions, layout=args.dda_layout
                )
            )
            # Surface reduction is measured separately from the DDA count.
            surface_device_count, surface_reduce_wall_ms, surface_reduce_gpu_ms = _gpu_stage(
                cp, lambda: cp.count_nonzero(surface)
            )
            surface_voxels = int(surface_device_count.get())
            # ``raycast_interior_count_cuda`` synchronizes internally and
            # returns the scalar used by the production runner.
            interior_voxels = int(interior_device_count)
            exterior_voxels = int(grid.n_voxels - interior_voxels - surface_voxels)
            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0

            rows.append(
                {
                    "repeat": repeat,
                    "local_frame": frame,
                    "full_frame": frame + args.frame_offset,
                    "trajectory_extract_ms": f"{extract_ms:.6f}",
                    "grid_ms": f"{grid_ms:.6f}",
                    "density_wall_ms": f"{density_wall_ms:.6f}",
                    "density_gpu_ms": f"{density_gpu_ms:.6f}",
                    "surface_mask_wall_ms": f"{surface_wall_ms:.6f}",
                    "surface_mask_gpu_ms": f"{surface_gpu_ms:.6f}",
                    "directions_ms": f"{directions_ms:.6f}",
                    "dda_wall_ms": f"{dda_wall_ms:.6f}",
                    "dda_gpu_ms": f"{dda_gpu_ms:.6f}",
                    "surface_reduce_wall_ms": f"{surface_reduce_wall_ms:.6f}",
                    "surface_reduce_gpu_ms": f"{surface_reduce_gpu_ms:.6f}",
                    "pipeline_ms": f"{pipeline_ms:.6f}",
                    "total_voxels": grid.n_voxels,
                    "surface_voxels": surface_voxels,
                    "interior_voxels": interior_voxels,
                    "exterior_voxels": exterior_voxels,
                    "grid_x": grid.shape[0],
                    "grid_y": grid.shape[1],
                    "grid_z": grid.shape[2],
                    "origin_x_A": f"{grid.origin_A[0]:.9f}",
                    "origin_y_A": f"{grid.origin_A[1]:.9f}",
                    "origin_z_A": f"{grid.origin_A[2]:.9f}",
                    "voxel_volume_A3": f"{grid.voxel_volume_A3:.6f}",
                    "density_kernel": cfg.density_kernel,
                    "nrays": cfg.nrays,
                }
            )
            del density, surface, directions

    csv_path = output / "profile_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    reference = _read_reference(args.reference_csv) if args.reference_csv is not None else {}
    validation_rows = []
    for row in rows:
        frame = int(row["local_frame"])
        ref = reference.get(frame)
        if ref is None:
            validation_rows.append({"local_frame": frame, "repeat": int(row["repeat"]), "status": "not_checked"})
            continue
        checks = {
            "interior_voxels": int(row["interior_voxels"]) == int(ref["interior_voxels"]),
            "total_voxels": int(row["total_voxels"]) == int(ref["total_voxels"]),
            "grid_x": int(row["grid_x"]) == int(ref["grid_x"]),
            "grid_y": int(row["grid_y"]) == int(ref["grid_y"]),
            "grid_z": int(row["grid_z"]) == int(ref["grid_z"]),
            "origin": all(
                np.isclose(float(row[key]), float(ref[key]), rtol=0.0, atol=1e-6)
                for key in ("origin_x_A", "origin_y_A", "origin_z_A")
            ),
        }
        validation_rows.append(
            {
                "local_frame": frame,
                "repeat": int(row["repeat"]),
                "status": "pass" if all(checks.values()) else "fail",
                "checks": checks,
            }
        )
    validation = {
        "status": "pass" if all(item["status"] in {"pass", "not_checked"} for item in validation_rows) else "fail",
        "reference_csv": str(args.reference_csv) if args.reference_csv is not None else None,
        "rows": validation_rows,
    }

    metadata = {
        "schema_version": 1,
        "analysis": "volinterior_cuda_profiling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"topology": str(args.topology), "trajectory": str(args.trajectory), "selection": args.selection},
        "frames": frames,
        "repeats": args.repeats,
        "parameters": {**cfg.as_dict(), "dda_layout": args.dda_layout},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "cupy": cp.__version__, "mdanalysis": mda.__version__},
        "cuda_device": str(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]),
        "summary": _summarize(rows),
        "validation": validation,
        "outputs": {"profile_csv": csv_path.name, "summary_json": "profile_summary.json", "validation_json": "validation.json"},
    }
    (output / "profile_summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": validation["status"], "output": str(output), "summary": metadata["summary"]}, indent=2))
    return 0 if validation["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
