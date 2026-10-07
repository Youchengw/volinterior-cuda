"""Screened connectivity with user-selected DDA fallback for capsid trajectories."""
from .config import GridSpec, VolInteriorConfig
from .measure import FrameResult, measure_frame

__version__ = "0.2.0"
__all__ = ["GridSpec", "VolInteriorConfig", "FrameResult", "measure_frame"]
