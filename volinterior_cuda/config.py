"""Numerical configuration and Cartesian grid metadata."""
from dataclasses import asdict, dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class VolInteriorConfig:
    """Surface/ray parameters; choose the method separately in measure_frame."""

    resolution: float = 12.5
    spacing_A: float = 1.0
    isovalue: float = 0.5
    nrays: int = 32
    backend: Literal["cpu", "cuda"] = "cuda"
    padding_A: float | None = None
    ray_scheme: Literal["vmd_poisson", "poisson", "fibonacci"] = "vmd_poisson"
    ray_seed: int = 512346
    ray_candidates: int = 40
    quicksurf_quality: int | None = None
    density_cutoff_sigma: float | None = None

    def __post_init__(self):
        for name in ("resolution", "spacing_A", "isovalue"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.nrays, (int, np.integer)) or not 1 <= self.nrays <= 65535:
            raise ValueError("nrays must be an integer in 1..65535")
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        if self.padding_A is not None and (not np.isfinite(self.padding_A) or self.padding_A < 0):
            raise ValueError("padding_A must be finite and nonnegative")
        if self.ray_scheme not in {"vmd_poisson", "poisson", "fibonacci"}:
            raise ValueError("unknown ray scheme")
        if not isinstance(self.ray_candidates, (int, np.integer)) or self.ray_candidates < 1:
            raise ValueError("ray_candidates must be a positive integer")
        if self.quicksurf_quality is not None and self.quicksurf_quality not in {0, 1, 2, 3}:
            raise ValueError("quicksurf_quality must be 0, 1, 2 or 3")
        if self.density_cutoff_sigma is not None and (not np.isfinite(self.density_cutoff_sigma) or self.density_cutoff_sigma <= 0):
            raise ValueError("density_cutoff_sigma must be finite and positive")

    @property
    def radius_scale_A(self):
        """Dimensionless QuickSurf scale; historical property name retained."""
        return 0.2 * self.resolution

    @property
    def resolved_quicksurf_quality(self):
        if self.quicksurf_quality is not None:
            return int(self.quicksurf_quality)
        return 0 if self.resolution >= 9.0 else 3

    @property
    def gausslim(self):
        if self.density_cutoff_sigma is not None:
            return float(self.density_cutoff_sigma)
        return (2.0, 2.5, 3.0, 4.0)[self.resolved_quicksurf_quality]

    def as_dict(self):
        return {**asdict(self), "radius_scale_A": self.radius_scale_A,
                "resolved_quicksurf_quality": self.resolved_quicksurf_quality,
                "gausslim": self.gausslim, "density_kernel": "cell_list"}


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

    @property
    def axis_vectors_A(self) -> np.ndarray:
        """Return the three full VMD grid-axis vectors in Å."""

        extents = (np.asarray(self.shape, dtype=np.float32) - np.float32(1.0)) * np.float32(
            self.spacing_A
        )
        return np.diag(extents).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "origin_A": self.origin_A.tolist(),
            "shape": list(self.shape),
            "spacing_A": float(self.spacing_A),
            "axis_vectors_A": self.axis_vectors_A.tolist(),
            "voxel_volume_A3": self.voxel_volume_A3,
        }
