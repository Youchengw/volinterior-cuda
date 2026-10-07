"""Random-access input adapters; all coordinates/radii use angstrom, masses dalton."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .radii import infer_radii_from_names


class FrameSource:
    def __init__(self, frames, times, radii, masses, reader, provenance):
        self.frames = np.asarray(frames)
        self.times = np.asarray(times, dtype=float)
        self.radii = np.asarray(radii, dtype=np.float32)
        self.masses = np.asarray(masses, dtype=np.float64)
        self.reader = reader
        self.provenance = provenance
        if (self.frames.ndim != 1 or not len(self.frames) or not np.isfinite(self.frames).all()
                or not np.array_equal(self.frames, np.floor(self.frames)) or (self.frames < 0).any()
                or (np.diff(self.frames) <= 0).any()):
            raise ValueError("frame IDs must be unique nonnegative increasing integers")
        self.frames = self.frames.astype(int)
        if (self.times.shape != self.frames.shape or not np.isfinite(self.times).all()
                or (self.times < 0).any() or (np.diff(self.times) <= 0).any()):
            raise ValueError("times must be finite, nonnegative and strictly increasing")
        if not len(self.radii) or self.masses.shape != self.radii.shape or self.radii.ndim != 1:
            raise ValueError("one radius and mass are required per selected atom")
        for values in (self.radii, self.masses):
            if not np.isfinite(values).all() or not (values > 0).all():
                raise ValueError("radii and masses must be finite and positive")
        self.lookup = {int(f): i for i, f in enumerate(self.frames)}

    def read(self, frame):
        coords = np.asarray(self.reader(self.lookup[frame]), dtype=np.float32)
        if coords.shape != (len(self.radii), 3) or not np.isfinite(coords).all():
            raise ValueError(f"invalid coordinates in frame {frame}")
        return coords

    def time(self, frame):
        return float(self.times[self.lookup[frame]])


def file_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def extraction_source(directory, radii_file=None):
    directory = Path(directory)
    if (directory / ".incomplete").exists():
        raise ValueError("atomic extraction is marked incomplete")
    params = {}
    if (directory / "metadata.json").exists():
        meta = json.loads((directory / "metadata.json").read_text())
        params = meta["parameters"]
        if meta.get("analysis") != "extract_trajectory" or meta.get("status") != "complete":
            raise ValueError("legacy extraction metadata does not report completion")
        if (params.get("coordinate_unit") != "angstrom" or params.get("mass_unit") != "dalton"
                or params.get("time_unit") != "ns" or params.get("coordinate_axes") != ["frame_index", "atom_index", "xyz"]):
            raise ValueError("unsupported extraction units or coordinate order")
    if (directory / "validation.json").exists():
        validation = json.loads((directory / "validation.json").read_text())
        if validation.get("status") != "pass":
            raise ValueError("legacy extraction validation did not pass")
    atoms = pd.read_csv(directory / "atoms.csv", keep_default_na=False, float_precision="round_trip")
    frames = pd.read_csv(directory / "frames.csv", float_precision="round_trip")
    xyz = np.load(directory / "coordinates.npy", mmap_mode="r", allow_pickle=False)
    if xyz.shape != (len(frames), len(atoms), 3) or xyz.dtype.kind != "f":
        raise ValueError("coordinate shape/dtype differs from atom and frame tables")
    if not np.array_equal(atoms.atom_index, np.arange(len(atoms))) or not np.array_equal(frames.frame_index, np.arange(len(frames))):
        raise ValueError("atom/frame table order differs from coordinate array")
    names = np.asarray(atoms.atom_name)
    elements = np.asarray(atoms.element) if "element" in atoms else None
    radii = np.load(radii_file, allow_pickle=False) if radii_file else infer_radii_from_names(names, elements)
    files = [directory / name for name in ("coordinates.npy", "atoms.csv", "frames.csv", "metadata.json", "validation.json")
             if (directory / name).is_file()]
    if radii_file:
        files.append(radii_file)
    return FrameSource(frames.full_frame, frames.time_ns, radii, atoms.mass_Da,
                       lambda index: xyz[index], {"kind": "extraction", "selection": params.get("selection"),
                       "radii_source": "explicit" if radii_file else "VMD names/elements",
                       "files": [file_info(p) for p in files]})


def numpy_source(coordinates_file, radii_file, masses_file, dt_ns):
    xyz = np.load(coordinates_file, mmap_mode="r", allow_pickle=False)
    if xyz.ndim == 2:
        xyz = xyz[None, ...]
    if xyz.ndim != 3 or xyz.shape[2] != 3 or not xyz.shape[0]:
        raise ValueError("coordinates must have shape (frames, atoms, 3) or (atoms, 3)")
    radii = np.load(radii_file, allow_pickle=False)
    masses = np.load(masses_file, allow_pickle=False)
    if xyz.shape[1] != len(radii):
        raise ValueError("coordinate atom count differs from radii")
    return FrameSource(np.arange(len(xyz)), np.arange(len(xyz)) * dt_ns, radii, masses,
                       lambda index: xyz[index], {"kind": "numpy", "radii_source": "explicit", "dt_ns": dt_ns,
                       "files": [file_info(p) for p in (coordinates_file, radii_file, masses_file)]})


def mdanalysis_source(topology, trajectory, selection, radii_file, dt_ns):
    import MDAnalysis as mda
    universe = mda.Universe(str(topology), str(trajectory))
    atoms = universe.select_atoms(selection)
    if not len(atoms):
        raise ValueError("atom selection is empty")
    try:
        masses = np.asarray(atoms.masses, dtype=float)
    except Exception as exc:
        raise ValueError("topology must provide or allow inference of positive masses for COM") from exc
    if radii_file:
        radii = np.load(radii_file, allow_pickle=False)
        radii_source = "explicit"
    else:
        try:
            radii = np.asarray(atoms.radii, dtype=np.float32)
        except Exception:
            radii = np.array([])
        if radii.shape == (len(atoms),) and np.isfinite(radii).all() and (radii > 0).all():
            radii_source = "topology"
        else:
            try:
                elements = np.asarray(atoms.elements)
            except Exception:
                elements = None
            radii = infer_radii_from_names(atoms.names, elements)
            radii_source = "VMD names/elements"
    def read(index):
        universe.trajectory[index]
        return atoms.positions.copy()
    files = [topology, trajectory] + ([radii_file] if radii_file else [])
    return FrameSource(np.arange(len(universe.trajectory)), np.arange(len(universe.trajectory)) * dt_ns,
                       radii, masses, read, {"kind": "MDAnalysis", "selection": selection,
                       "radii_source": radii_source, "dt_ns": dt_ns, "coordinates": "as read; no additional unwrap/alignment",
                       "files": [file_info(p) for p in files]})
