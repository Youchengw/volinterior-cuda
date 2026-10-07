"""Explicit-input command line interface for screened capsid voxel classification."""
import argparse
from pathlib import Path
import numpy as np
from .config import VolInteriorConfig
from .measure import minimum_blocked
from .trajectory import extraction_source, numpy_source, mdanalysis_source
from .workflow import run_trajectory


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="completed atomic extraction directory")
    source.add_argument("--coordinates-npy", type=Path, help="coordinates in angstrom: (frames, atoms, 3) or (atoms, 3)")
    source.add_argument("--topology", type=Path, help="MDAnalysis topology; also supply --trajectory and --dt-ns")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--selection", default="protein")
    parser.add_argument("--radii-npy", type=Path, help="one radius in angstrom per selected atom")
    parser.add_argument("--masses-npy", type=Path, help="one mass in dalton per atom; required with coordinates-npy")
    parser.add_argument("--dt-ns", type=float, help="frame spacing; required for NumPy and direct trajectory inputs")
    parser.add_argument("--frames", help="comma-separated source frame IDs; default: all")
    parser.add_argument("--method", choices=("auto", "connectivity", "fixed", "fuzzy"), default="auto")
    parser.add_argument("--check-frames", type=int, default=9, help="uniform sample count, including both endpoints")
    parser.add_argument("--check-only", action="store_true", help="screen samples and stop")
    parser.add_argument("--fallback", choices=("fixed", "fuzzy"), help="optional advance choice; default asks you to choose after a leak")
    parser.add_argument("--cutoff", type=float, help="explicit blocked-ray fraction required for fuzzy or fallback fuzzy")
    parser.add_argument("--nrays", type=int, default=32)
    parser.add_argument("--ray-directions-npy", type=Path, help="optional VMD-captured directions with shape (nrays, 3)")
    parser.add_argument("--resolution", type=float, default=12.5)
    parser.add_argument("--isovalue", type=float, default=0.5)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save-labels", action="store_true", help="save per-frame 3-D int8 voxel labels; can require substantial disk space")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory exists; choose a new --output-dir")
    if args.check_frames < 2 or not 1 <= args.nrays <= 65535 or args.device < 0:
        parser.error("check-frames >= 2, nrays 1..65535 and device >= 0 are required")
    if any(not np.isfinite(v) or v <= 0 for v in (args.resolution, args.isovalue, args.spacing)):
        parser.error("resolution, isovalue and spacing must be finite and positive")
    if args.fallback and args.method != "auto":
        parser.error("--fallback applies only to --method auto")
    if args.check_only and args.method not in {"auto", "connectivity"}:
        parser.error("--check-only applies to auto/connectivity")
    if args.method == "fuzzy" or args.fallback == "fuzzy":
        try:
            minimum_blocked(args.cutoff, args.nrays)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.cutoff is not None:
        parser.error("--cutoff applies only to --method fuzzy or --fallback fuzzy")
    if args.input:
        if args.trajectory or args.masses_npy or args.dt_ns is not None:
            parser.error("extraction supplies coordinates, masses and times; do not override them")
    else:
        if args.dt_ns is None or not np.isfinite(args.dt_ns) or args.dt_ns <= 0:
            parser.error("provide a finite positive --dt-ns for NumPy/direct trajectory input")
        if args.coordinates_npy and (not args.radii_npy or not args.masses_npy or args.trajectory):
            parser.error("coordinates-npy requires radii-npy and masses-npy, and cannot use trajectory")
        if args.topology and (not args.trajectory or args.masses_npy):
            parser.error("topology requires trajectory; masses are read from the topology")
    directions = None
    if args.ray_directions_npy:
        directions = np.load(args.ray_directions_npy, allow_pickle=False)
        if (directions.shape != (args.nrays, 3) or not np.isfinite(directions).all()
                or np.any(np.linalg.norm(directions, axis=1) == 0)):
            parser.error("ray-directions-npy must contain nrays finite nonzero 3-D vectors")
    if args.backend == "cuda":
        try:
            import cupy as cp
            cp.cuda.Device(args.device).use()
            probe = cp.asarray([0], dtype=cp.float32) + np.float32(1)
            cp.cuda.get_current_stream().synchronize()
            del probe
        except Exception as exc:
            parser.error(f"CUDA initialization failed: {exc}. Install the cuda extra; use --backend cpu explicitly for a CPU reference.")
    if args.input:
        source = extraction_source(args.input, args.radii_npy)
    elif args.coordinates_npy:
        source = numpy_source(args.coordinates_npy, args.radii_npy, args.masses_npy, args.dt_ns)
    else:
        source = mdanalysis_source(args.topology, args.trajectory, args.selection, args.radii_npy, args.dt_ns)
    try:
        frames = None if args.frames is None else [int(x) for x in args.frames.split(",")]
    except ValueError:
        parser.error("--frames must be comma-separated integer frame IDs")
    config = VolInteriorConfig(backend=args.backend, resolution=args.resolution, spacing_A=args.spacing,
                               isovalue=args.isovalue, nrays=args.nrays)
    result = run_trajectory(source, args.output_dir, config, frames=frames, method=args.method,
                            check_frames=args.check_frames, check_only=args.check_only,
                            fallback=args.fallback, cutoff=args.cutoff, save_labels=args.save_labels,
                            directions=directions)
    status = result["status"]
    if status == "needs_dda_choice":
        print(f"Connectivity is not enclosed at frame {result['fallback_frame']}. Diagnostics: {args.output_dir}")
        print("Choose a DDA method and a NEW output directory, keeping the same input/frame/surface parameters:")
        print("  --method fixed                 VMD fixed all-rays-blocked classification")
        print("  --method fuzzy --cutoff VALUE  explicit blocked-ray fraction")
        raise SystemExit(2)
    if status in {"connectivity_failed", "screen_failed"}:
        raise SystemExit(f"CONNECTIVITY FAILED: inspect {args.output_dir}/metadata.json")
    print(f"{status.upper()}: method={result['selected_method']}; {args.output_dir}")


if __name__ == "__main__":
    main()
