"""CPU-only tests for the trajectory runner's portable output writer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_volinterior_cuda import (  # noqa: E402
    _box_for_frame,
    _radii_from_atoms,
    _write_dx,
)
from volinterior_cuda.config import GridSpec  # noqa: E402


def test_write_dx_round_trips_project_dx_order(tmp_path):
    grid = GridSpec(np.array([1.0, 2.0, 3.0], dtype=np.float32), (2, 2, 2), 1.0)
    values = np.arange(8, dtype=np.float32).reshape(grid.shape)
    path = tmp_path / "labels.dx"

    _write_dx(path, grid, values, "test labels")
    lines = path.read_text(encoding="ascii").splitlines()
    data_start = next(i for i, line in enumerate(lines) if "data follows" in line) + 1
    data_end = next(i for i, line in enumerate(lines[data_start:], data_start) if line.startswith("object "))
    flat = np.fromstring(" ".join(lines[data_start:data_end]), sep=" ")

    assert flat.size == 8
    np.testing.assert_array_equal(flat, values.ravel())
    assert lines[data_end] == 'object "test labels" class field'


def test_radii_source_prefers_valid_topology_radii():
    class DummyAtoms(SimpleNamespace):
        def __len__(self):
            return len(self.names)

    atoms = DummyAtoms(
        names=np.asarray(["C1", "N1"]),
        elements=np.asarray(["C", "N"]),
        radii=np.asarray([1.7, 1.8]),
    )
    np.testing.assert_array_equal(
        _radii_from_atoms(atoms, "auto"), np.asarray([1.7, 1.8], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _radii_from_atoms(atoms, "names"), np.asarray([1.5, 1.4], dtype=np.float32)
    )


def test_box_domain_requires_finite_positive_lengths():
    universe = SimpleNamespace(dimensions=np.asarray([40.0, 41.0, 42.0, 90.0, 90.0, 90.0]))
    np.testing.assert_array_equal(_box_for_frame(universe, "box"), [40.0, 41.0, 42.0])
    assert _box_for_frame(universe, "selection") is None
    universe.dimensions = np.asarray([40.0, 0.0, 42.0])
    with pytest.raises(ValueError, match="finite positive"):
        _box_for_frame(universe, "box")
