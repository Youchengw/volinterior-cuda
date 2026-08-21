"""Configuration and grid objects for the Python volinterior implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


Backend = Literal["auto", "cpu", "cuda"]
Mode = Literal["fixed", "fuzzy"]
Classifier = Literal["connectivity", "dda"]
RayScheme = Literal["vmd_poisson", "poisson", "fibonacci"]
DensityKernel = Literal["cell_list", "atom"]


@dataclass(frozen=True)
class VolInteriorConfig:
    """Numerical parameters corresponding to VMD's ``measure volinterior``.

    ``resolution`` is the VMD value.  The VMD Tcl/C++ implementation converts
    it to the QuickSurf radius scale with ``radscale = 0.2 * resolution``;
    :attr:`radius_scale_A` exposes that conversion explicitly.
    """

    resolution: float = 12.5
    spacing_A: float = 1.0
    isovalue: float = 0.5
    nrays: int = 32
    mode: Mode = "fixed"
    classifier: Classifier = "dda"
    backend: Backend = "auto"
    domain: Literal["selection", "box"] = "selection"
    padding_A: float | None = None
    ray_scheme: RayScheme = "vmd_poisson"
    ray_seed: int = 512346
    ray_candidates: int = 40
    fuzzy_cutoff: float = 0.5
    quicksurf_quality: int | None = None
    density_cutoff_sigma: float | None = None
    density_kernel: DensityKernel = "cell_list"
    accel_grid_spacing_A: float | None = None
    atom_chunk: int = 4096

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.spacing_A <= 0:
            raise ValueError("spacing_A must be positive")
        if not 0 < self.isovalue:
            raise ValueError("isovalue must be positive")
        if self.nrays < 1:
            raise ValueError("nrays must be at least one")
        if self.mode not in {"fixed", "fuzzy"}:
            raise ValueError("mode must be 'fixed' or 'fuzzy'")
        if self.classifier not in {"connectivity", "dda"}:
            raise ValueError("classifier must be 'connectivity' or 'dda'")
        if self.backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be 'auto', 'cpu' or 'cuda'")
        if self.domain not in {"selection", "box"}:
            raise ValueError("domain must be 'selection' or 'box'")
        if self.padding_A is not None and self.padding_A < 0:
            raise ValueError("padding_A cannot be negative")
        if self.fuzzy_cutoff < 0 or self.fuzzy_cutoff > 1:
            raise ValueError("fuzzy_cutoff must be in [0, 1]")
        if self.ray_candidates < 1:
            raise ValueError("ray_candidates must be positive")
        if self.quicksurf_quality is not None and self.quicksurf_quality not in {0, 1, 2, 3}:
            raise ValueError("quicksurf_quality must be 0, 1, 2 or 3")
        if self.density_cutoff_sigma is not None and self.density_cutoff_sigma <= 0:
            raise ValueError("density_cutoff_sigma must be positive")
        if self.density_kernel not in {"cell_list", "atom"}:
            raise ValueError("density_kernel must be 'cell_list' or 'atom'")
        if self.accel_grid_spacing_A is not None and self.accel_grid_spacing_A <= 0:
            raise ValueError("accel_grid_spacing_A must be positive")
        if self.atom_chunk < 1:
            raise ValueError("atom_chunk must be positive")

    @property
    def radius_scale_A(self) -> float:
        """QuickSurf radius scale in Å, using VMD's ``resolution / 5`` rule."""

        return 0.2 * self.resolution

    @property
    def resolved_quicksurf_quality(self) -> int:
        """VMD's quality choice made by ``measure volinterior``."""
        if self.quicksurf_quality is not None:
            return int(self.quicksurf_quality)
        return 0 if self.resolution >= 9.0 else 3

    @property
    def gausslim(self) -> float:
        """Gaussian cutoff used by VMD QuickSurf for the selected quality."""
        if self.density_cutoff_sigma is not None:
            return float(self.density_cutoff_sigma)
        return (2.0, 2.5, 3.0, 4.0)[self.resolved_quicksurf_quality]

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["radius_scale_A"] = self.radius_scale_A
        values["resolved_quicksurf_quality"] = self.resolved_quicksurf_quality
        values["gausslim"] = self.gausslim
        return values


@dataclass(frozen=True)
class GridSpec:
    """Regular Cartesian grid; coordinates refer to VMD grid points."""

    origin_A: np.ndarray
    shape: tuple[int, int, int]
    spacing_A: float

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_A, dtype=np.float32)
        if origin.shape != (3,):
            raise ValueError("origin_A must have shape (3,)")
        if len(self.shape) != 3 or any(int(n) < 1 for n in self.shape):
            raise ValueError("shape must contain three positive dimensions")
        if self.spacing_A <= 0:
            raise ValueError("spacing_A must be positive")
        object.__setattr__(self, "origin_A", origin)
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def voxel_volume_A3(self) -> float:
        return float(self.spacing_A**3)

    @property
    def upper_edge_A(self) -> np.ndarray:
        return self.origin_A + np.asarray(self.shape, dtype=np.float32) * self.spacing_A

    def as_dict(self) -> dict[str, object]:
        return {
            "origin_A": self.origin_A.tolist(),
            "shape": list(self.shape),
            "spacing_A": float(self.spacing_A),
            "voxel_volume_A3": self.voxel_volume_A3,
        }
