"""Optional MDAnalysis adapter; imported only when a trajectory is requested."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .radii import infer_radii_from_names


def iter_mdanalysis_frames(
    topology: str,
    trajectory: str,
    *,
    selection: str = "protein",
    frames: Iterable[int] | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(frame, coords, radii, box_lengths)`` for selected atoms."""

    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - optional adapter
        raise RuntimeError("trajectory input requires MDAnalysis") from exc
    universe = mda.Universe(topology, trajectory)
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"selection returned no atoms: {selection!r}")

    try:
        elements = getattr(atoms, "elements", None)
    except Exception:
        elements = None
    if elements is not None:
        elements = np.asarray(elements)
    names = np.asarray(atoms.names)
    try:
        radii_attr = getattr(atoms, "radii", None)
    except Exception:
        radii_attr = None
    if radii_attr is not None:
        radii_candidate = np.asarray(radii_attr, dtype=np.float32)
    else:
        radii_candidate = np.asarray([], dtype=np.float32)
    if radii_candidate.shape == (len(atoms),) and np.all(np.isfinite(radii_candidate)) and np.all(radii_candidate > 0):
        radii = np.ascontiguousarray(radii_candidate)
    else:
        radii = infer_radii_from_names(names, elements)

    selected_frames = range(len(universe.trajectory)) if frames is None else frames
    for frame in selected_frames:
        universe.trajectory[int(frame)]
        box = np.asarray(universe.dimensions[:3], dtype=np.float32)
        if np.any(box <= 0):
            box = None
        yield int(frame), np.asarray(atoms.positions, dtype=np.float32).copy(), radii.copy(), box
