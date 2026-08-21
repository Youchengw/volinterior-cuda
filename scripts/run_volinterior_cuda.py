#!/usr/bin/env python3
"""Run the CUDA volinterior workflow on selected trajectory frames.

The runner is intentionally output-directory driven. It writes a resumable
CSV/checkpoint pair and a provenance-rich metadata.json. Scalar mode uses the
GPU-only count path; ``--maps`` additionally writes per-frame label DX maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from volinterior_cuda import (
    VolInteriorConfig,
    infer_radii_from_names,
    measure_volinterior,
    measure_volinterior_counts,
)

CSV_FIELDS = [
    "local_frame",
    "full_frame",
    "time_ns",
    "elapsed_ms",
    "backend",
    "total_voxels",
    "interior_voxels",
    "surface_voxels",
    "exterior_voxels",
    "grid_x",
    "grid_y",
    "grid_z",
    "origin_x_A",
    "origin_y_A",
    "origin_z_A",
    "voxel_volume_A3",
    "interior_volume_A3",
    "map_path",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run directory, or an existing directory with --resume")
    parser.add_argument("--stage", choices=("smoke", "exploration", "production"), default="exploration")
    parser.add_argument("--frames", default="0", help="comma-separated local trajectory frames")
    parser.add_argument("--first-frame", type=int, default=None)
    parser.add_argument("--last-frame", type=int, default=None)
    parser.add_argument("--frame-offset", type=int, default=0, help="full_frame = local_frame + offset")
    parser.add_argument("--dt-ns", type=float, default=1.2)
    parser.add_argument("--time-origin-ns", type=float, default=0.0, help="time of local frame 0")
    parser.add_argument("--selection", default="protein")
    parser.add_argument(
        "--radii-source",
        choices=("auto", "topology", "elements", "names"),
        default="auto",
        help="atom radii source; auto prefers topology radii, then elements/names",
    )
    parser.add_argument("--expected-atoms", type=int, default=None, help="optional selection-size guard")
    parser.add_argument("--domain", choices=("selection", "box"), default="selection")
    parser.add_argument("--resolution", type=float, default=12.5)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--isovalue", type=float, default=0.5)
    parser.add_argument("--nrays", type=int, default=32)
    parser.add_argument("--ray-scheme", choices=("vmd_poisson", "poisson", "fibonacci"), default="vmd_poisson")
    parser.add_argument("--ray-seed", type=int, default=512346)
    parser.add_argument("--density-kernel", choices=("cell_list", "atom"), default="cell_list")
    parser.add_argument("--classifier", choices=("dda", "connectivity"), default="dda")
    parser.add_argument("--maps", action="store_true", help="write per-frame label DX maps; scalar mode is faster and smaller")
    parser.add_argument("--resume", action="store_true", help="resume rows already present in --output/cuda_lumen_volume.csv")
    return parser


def _parse_frames(args: argparse.Namespace) -> list[int]:
    if args.first_frame is not None or args.last_frame is not None:
        first = 0 if args.first_frame is None else args.first_frame
        last = first if args.last_frame is None else args.last_frame
        if first < 0 or last < first:
            raise ValueError("invalid --first-frame/--last-frame range")
        return list(range(first, last + 1))
    frames = sorted({int(item.strip()) for item in args.frames.split(",") if item.strip()})
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
        classifier=args.classifier,
        backend="cuda",
        domain=args.domain,
        ray_scheme=args.ray_scheme,
        ray_seed=args.ray_seed,
        density_kernel=args.density_kernel,
    )


def _radii_from_atoms(atoms, source: str = "auto") -> np.ndarray:
    """Resolve VMD-like per-atom radii from a selected AtomGroup.

    ``auto`` uses topology radii when present and valid, then falls back to
    element/name inference. Explicit sources fail loudly when unavailable so a
    different capsid cannot silently receive inappropriate radii.
    """

    if source not in {"auto", "topology", "elements", "names"}:
        raise ValueError("source must be auto, topology, elements, or names")
    if source in {"auto", "topology"}:
        for attribute in ("radii", "vdw_radii", "vdwradii"):
            try:
                values = getattr(atoms, attribute, None)
            except Exception:
                values = None
            if values is None:
                continue
            radii = np.asarray(values, dtype=np.float32)
            if radii.shape == (len(atoms),) and np.all(np.isfinite(radii)) and np.all(radii > 0):
                return np.ascontiguousarray(radii)
        if source == "topology":
            raise ValueError("topology radii were requested but no valid radii attribute is available")

    try:
        elements = getattr(atoms, "elements", None)
    except Exception:
        elements = None
    if source == "elements" and elements is None:
        raise ValueError("element radii were requested but the topology has no elements attribute")
    if source in {"auto", "elements"} and elements is not None:
        elements = np.asarray(elements)
    else:
        elements = None
    return infer_radii_from_names(np.asarray(atoms.names), elements)


def _box_for_frame(universe, domain: str) -> np.ndarray | None:
    if domain != "box":
        return None
    dimensions = getattr(universe, "dimensions", None)
    if dimensions is None:
        raise ValueError("domain='box' requires trajectory unit-cell dimensions")
    box = np.asarray(dimensions[:3], dtype=np.float32)
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("domain='box' requires finite positive unit-cell lengths")
    return box


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_dx(path: Path, grid, values: np.ndarray, title: str) -> None:
    """Write a scalar x/y/z array in the project OpenDX reader ordering."""
    data = np.asarray(values, dtype=np.float32)
    if data.shape != grid.shape:
        raise ValueError(f"DX values shape {data.shape} does not match grid {grid.shape}")
    # The project reader reshapes the data block directly as (nx, ny, nz).
    # Keep this ordering so coordinate indexing and VMD-exported maps agree.
    flat = data.ravel(order="C")
    nx, ny, nz = grid.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# Generated by volinterior-cuda\n")
        handle.write(f"# {title}\n")
        handle.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        handle.write("origin %.9g %.9g %.9g\n" % tuple(float(v) for v in grid.origin_A))
        handle.write("delta %.9g 0 0\n" % float(grid.spacing_A))
        handle.write("delta 0 %.9g 0\n" % float(grid.spacing_A))
        handle.write("delta 0 0 %.9g\n" % float(grid.spacing_A))
        handle.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        handle.write(f"object 3 class array type float rank 0 items {flat.size} data follows\n")
        for start in range(0, flat.size, 8):
            handle.write(" ".join(f"{float(value):.8g}" for value in flat[start:start + 8]))
            handle.write("\n")
        # Keep the field object immediately after the numeric block. The
        # project's OpenDX reader uses the first object line as the data end.
        handle.write(f'object "{title}" class field\n')
        handle.write('attribute "dep" string "positions"\n')
        handle.write('component "positions" value 1\n')
        handle.write('component "connections" value 2\n')
        handle.write('component "data" value 3\n')

def _initial_metadata(args, cfg: VolInteriorConfig, frames: list[int], universe, atoms, radii) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis": "volinterior_cuda_trajectory",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": shlex.join(sys.argv),
        "stage": args.stage,
        "selection": args.selection,
        "selected_atoms": int(len(atoms)),
        "expected_selected_atoms": args.expected_atoms,
        "radii_source": args.radii_source,
        "domain": args.domain,
        "trajectory_frames_available": int(len(universe.trajectory)),
        "trajectory_atoms_available": int(universe.atoms.n_atoms),
        "radii_min_A": float(np.min(radii)),
        "radii_max_A": float(np.max(radii)),
        "frame_mapping": {
            "local_frames": frames,
            "full_frame_offset": args.frame_offset,
            "full_frames": [frame + args.frame_offset for frame in frames],
            "dt_ns": args.dt_ns,
            "time_origin_ns": args.time_origin_ns,
        },
        "parameters": cfg.as_dict(),
        "options": {"maps": bool(args.maps), "resume": bool(args.resume)},
        "inputs": {"topology": _file_record(args.topology), "trajectory": _file_record(args.trajectory)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            "csv": "cuda_lumen_volume.csv",
            "checkpoint": "checkpoint.json",
            "maps": "maps/" if args.maps else None,
        },
    }


def _read_completed(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {int(row["local_frame"]) for row in csv.DictReader(handle)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _result_row(local_frame: int, result, elapsed_ms: float, args: argparse.Namespace, map_path: str) -> dict[str, Any]:
    full_frame = local_frame + args.frame_offset
    return {
        "local_frame": local_frame,
        "full_frame": full_frame,
        "time_ns": f"{args.time_origin_ns + local_frame * args.dt_ns:.8f}",
        "elapsed_ms": f"{elapsed_ms:.3f}",
        "backend": result.backend,
        "total_voxels": result.grid.n_voxels,
        "interior_voxels": result.interior_voxels,
        "surface_voxels": result.surface_voxels if hasattr(result, "surface_voxels") else int(np.count_nonzero(result.surface)),
        "exterior_voxels": result.exterior_voxels,
        "grid_x": result.grid.shape[0],
        "grid_y": result.grid.shape[1],
        "grid_z": result.grid.shape[2],
        "origin_x_A": f"{result.grid.origin_A[0]:.9f}",
        "origin_y_A": f"{result.grid.origin_A[1]:.9f}",
        "origin_z_A": f"{result.grid.origin_A[2]:.9f}",
        "voxel_volume_A3": f"{result.grid.voxel_volume_A3:.6f}",
        "interior_volume_A3": f"{result.interior_volume_A3:.6f}",
        "map_path": map_path,
    }


def main() -> int:
    args = _parser().parse_args()
    args.topology = args.topology.resolve()
    args.trajectory = args.trajectory.resolve()
    output = args.output.resolve()
    frames = _parse_frames(args)
    cfg = _config(args)
    if cfg.classifier != "dda":
        raise ValueError("trajectory scalar runner currently requires --classifier dda")
    if not args.topology.exists() or not args.trajectory.exists():
        raise FileNotFoundError("topology and trajectory must exist")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; use --resume for an existing run")
    output.mkdir(parents=True, exist_ok=True)
    if args.maps:
        (output / "maps").mkdir(exist_ok=True)

    import cupy as cp
    import MDAnalysis as mda

    universe = mda.Universe(str(args.topology), str(args.trajectory))
    if max(frames) >= len(universe.trajectory):
        raise ValueError(f"requested frame {max(frames)} exceeds trajectory length {len(universe.trajectory)}")
    atoms = universe.select_atoms(args.selection)
    if len(atoms) == 0:
        raise ValueError(f"selection returned no atoms: {args.selection}")
    radii = _radii_from_atoms(atoms, args.radii_source)
    if args.expected_atoms is not None and len(atoms) != args.expected_atoms:
        raise ValueError(
            f"selection has {len(atoms)} atoms, expected {args.expected_atoms}"
        )

    csv_path = output / "cuda_lumen_volume.csv"
    checkpoint_path = output / "checkpoint.json"
    metadata_path = output / "metadata.json"
    completed = _read_completed(csv_path) if args.resume else set()
    pending = [frame for frame in frames if frame not in completed]

    if metadata_path.exists() and args.resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = _initial_metadata(args, cfg, frames, universe, atoms, radii)
    metadata["software"].update({"cupy": cp.__version__, "mdanalysis": mda.__version__})
    try:
        device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        metadata["software"]["cuda_device"] = str(device_name)
    except Exception:
        metadata["software"]["cuda_device"] = "unknown"
    _write_json(metadata_path, metadata)

    if not pending:
        metadata["status"] = "complete"
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(metadata_path, metadata)
        print(json.dumps({"status": "complete", "frames": len(frames), "output": str(output)}, indent=2))
        return 0

    # Compile/cache kernels before timed frames. The warmup is never written as
    # a measurement row and is recorded only in metadata.
    universe.trajectory[pending[0]]
    warmup_coords = np.asarray(atoms.positions, dtype=np.float32).copy()
    warmup_box = _box_for_frame(universe, args.domain)
    warmup_start = time.perf_counter()
    warmup_result = measure_volinterior_counts(
        warmup_coords, radii, box_A=warmup_box, config=cfg
    )
    cp.cuda.Stream.null.synchronize()
    metadata["warmup_ms"] = (time.perf_counter() - warmup_start) * 1000.0
    del warmup_result

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    elapsed: list[float] = []
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for local_frame in pending:
            universe.trajectory[local_frame]
            coords = np.asarray(atoms.positions, dtype=np.float32).copy()
            box_A = _box_for_frame(universe, args.domain)
            start = time.perf_counter()
            map_path = ""
            if args.maps:
                result = measure_volinterior(
                    coords,
                    radii,
                    box_A=box_A,
                    config=cfg,
                    return_density=False,
                    return_probability=False,
                )
                cp.cuda.Stream.null.synchronize()
                labels = np.zeros(result.grid.shape, dtype=np.float32)
                labels[result.exterior] = 0.0
                labels[result.interior] = 1.0
                labels[result.surface] = 2.0
                map_file = output / "maps" / f"frame_full{local_frame + args.frame_offset:04d}_labels.dx"
                _write_dx(map_file, result.grid, labels, f"volinterior labels frame {local_frame + args.frame_offset}")
                map_path = str(map_file.relative_to(output))
            else:
                result = measure_volinterior_counts(
                    coords, radii, box_A=box_A, config=cfg
                )
                cp.cuda.Stream.null.synchronize()
            duration = (time.perf_counter() - start) * 1000.0
            elapsed.append(duration)
            writer.writerow(_result_row(local_frame, result, duration, args, map_path))
            handle.flush()
            completed.add(local_frame)
            _write_json(checkpoint_path, {
                "status": "running",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "completed_local_frames": sorted(completed),
                "remaining_local_frames": [frame for frame in frames if frame not in completed],
            })
            del result

    metadata["status"] = "complete"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["summary"] = {
        "frame_count": len(frames),
        "elapsed_ms": {
            "median": statistics.median(elapsed),
            "mean": statistics.mean(elapsed),
            "p95": float(np.percentile(elapsed, 95)),
            "min": min(elapsed),
            "max": max(elapsed),
        },
    }
    _write_json(metadata_path, metadata)
    _write_json(checkpoint_path, {
        "status": "complete",
        "updated_utc": metadata["completed_utc"],
        "completed_local_frames": sorted(completed),
        "remaining_local_frames": [],
    })
    print(json.dumps({"status": "complete", "frames": len(frames), "output": str(output), "summary": metadata["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
