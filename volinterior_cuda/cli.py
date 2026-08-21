"""Command-line entry point for one-frame or trajectory volinterior runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import VolInteriorConfig
from .measure import measure_volinterior
from .trajectory import iter_mdanalysis_frames


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CUDA/CPU Python implementation of measure volinterior")
    p.add_argument("--coordinates-npy", type=Path, help="(n_atoms,3) float32 NumPy array")
    p.add_argument("--radii-npy", type=Path, help="(n_atoms,) radii in Å")
    p.add_argument("--topology", type=Path)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--selection", default="protein")
    p.add_argument("--frames", default=None, help="comma-separated frame indices for MDAnalysis input")
    p.add_argument("--box", nargs=3, type=float, metavar=("LX", "LY", "LZ"))
    p.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resolution", type=float, default=12.5)
    p.add_argument("--spacing", type=float, default=1.0)
    p.add_argument("--isovalue", type=float, default=0.5)
    p.add_argument("--nrays", type=int, default=32)
    p.add_argument("--mode", choices=("fixed", "fuzzy"), default="fixed")
    p.add_argument("--classifier", choices=("connectivity", "dda"), default="dda")
    p.add_argument("--domain", choices=("selection", "box"), default="selection")
    p.add_argument("--ray-scheme", choices=("vmd_poisson", "poisson", "fibonacci"), default="vmd_poisson")
    p.add_argument("--ray-seed", type=int, default=512346)
    p.add_argument("--fuzzy-cutoff", type=float, default=0.5)
    p.add_argument("--probability", action="store_true", help="materialize the full blocked-ray probability map")
    p.add_argument("--output", type=Path, required=True, help=".json summary; masks are written beside it as .npz")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    npy_mode = args.coordinates_npy is not None or args.radii_npy is not None
    traj_mode = args.topology is not None or args.trajectory is not None
    if npy_mode == traj_mode:
        raise SystemExit("provide either --coordinates-npy/--radii-npy or --topology/--trajectory")
    if npy_mode and (args.coordinates_npy is None or args.radii_npy is None):
        raise SystemExit("NPY mode requires both --coordinates-npy and --radii-npy")

    cfg = VolInteriorConfig(
        backend=args.backend,
        resolution=args.resolution,
        spacing_A=args.spacing,
        isovalue=args.isovalue,
        nrays=args.nrays,
        mode=args.mode,
        classifier=args.classifier,
        domain=args.domain,
        ray_scheme=args.ray_scheme,
        ray_seed=args.ray_seed,
        fuzzy_cutoff=args.fuzzy_cutoff,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if npy_mode:
        coords = np.load(args.coordinates_npy)
        radii = np.load(args.radii_npy)
        result = measure_volinterior(coords, radii, box_A=args.box, config=cfg, return_density=False, return_probability=args.probability)
        rows.append(result.summary())
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            surface=result.surface,
            interior=result.interior,
            exterior=result.exterior,
            probability=result.probability if result.probability is not None else np.array([], dtype=np.float32),
        )
    else:
        if args.topology is None or args.trajectory is None:
            raise SystemExit("trajectory mode requires --topology and --trajectory")
        frames = None if args.frames is None else [int(x) for x in args.frames.split(",") if x.strip()]
        for frame, coords, radii, box in iter_mdanalysis_frames(
            str(args.topology), str(args.trajectory), selection=args.selection, frames=frames
        ):
            result = measure_volinterior(coords, radii, box_A=box, config=cfg, return_density=False)
            row = result.summary()
            row["frame"] = frame
            rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0
