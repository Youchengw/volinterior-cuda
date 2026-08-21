"""CUDA-capable Python implementation of the VMD ``volinterior`` workflow.

The public entry point is :func:`measure_volinterior`.  CuPy is optional: the
same code can be run with ``backend='cpu'`` for validation against a CUDA run.
"""

from .config import GridSpec, VolInteriorConfig
from .measure import (
    VolInteriorCountsResult,
    VolInteriorResult,
    measure_volinterior,
    measure_volinterior_counts,
)
from .radii import infer_radii_from_names

__all__ = [
    "GridSpec",
    "VolInteriorConfig",
    "VolInteriorCountsResult",
    "VolInteriorResult",
    "infer_radii_from_names",
    "measure_volinterior",
    "measure_volinterior_counts",
]
