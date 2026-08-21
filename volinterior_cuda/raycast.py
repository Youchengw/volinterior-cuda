"""VMD-style ray casting using a voxel DDA on CPU or CUDA."""

from __future__ import annotations

import numpy as np

from .config import GridSpec


def _ray_escapes(surface: np.ndarray, index: tuple[int, int, int], direction: np.ndarray) -> bool:
    """Return True when VMD's DDA reaches the grid boundary before a wall."""

    nx, ny, nz = surface.shape
    coord = np.asarray(index, dtype=np.int32).copy()
    ray_orig = coord.astype(np.float32) + np.float32(0.5)
    d = np.asarray(direction, dtype=np.float32)
    step = np.empty(3, dtype=np.int32)
    delta_dist = np.empty(3, dtype=np.float32)
    next_crossing = np.empty(3, dtype=np.float32)
    for axis in range(3):
        if d[axis] != 0.0:
            ratios = d / d[axis]
        else:
            ratios = d.copy()
        delta_dist[axis] = np.float32(np.sqrt(np.sum(ratios * ratios, dtype=np.float32)))
        if d[axis] < 0.0:
            step[axis] = -1
            next_crossing[axis] = (ray_orig[axis] - coord[axis]) * delta_dist[axis]
        else:
            step[axis] = 1
            next_crossing[axis] = (coord[axis] + 1.0 - ray_orig[axis]) * delta_dist[axis]

    shape = (nx, ny, nz)
    while True:
        side = 0
        for axis in (1, 2):
            if next_crossing[side] > next_crossing[axis]:
                side = axis
        next_crossing[side] += delta_dist[side]
        coord[side] += step[side]
        if coord[side] < 0 or coord[side] >= shape[side]:
            return True
        if bool(surface[tuple(int(x) for x in coord)]):
            return False


def raycast_blocked_cpu(surface: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Count blocked directions for every voxel (surface voxels are zero)."""

    surface = np.asarray(surface, dtype=bool)
    directions = np.asarray(directions, dtype=np.float32)
    blocked = np.zeros(surface.shape, dtype=np.uint16 if len(directions) <= 65535 else np.uint32)
    for index in np.ndindex(surface.shape):
        if surface[index]:
            continue
        count = 0
        for direction in directions:
            if not _ray_escapes(surface, index, direction):
                count += 1
        blocked[index] = count
    return blocked


def _precompute_dda_parameters(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute direction-independent DDA state once per ray.

    VMD's DDA uses the same ``deltaDist`` and signed step for every starting
    voxel. Moving these divisions/square-roots out of the voxel kernel removes
    roughly 1.4 billion redundant calculations for the canonical grid.
    """

    dirs = np.ascontiguousarray(directions, dtype=np.float32)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("directions must have shape (nrays, 3)")
    delta = np.empty_like(dirs, dtype=np.float32)
    for ray, direction in enumerate(dirs):
        for axis in range(3):
            if direction[axis] != 0.0:
                ratios = direction / direction[axis]
            else:
                ratios = direction.copy()
            delta[ray, axis] = np.float32(
                np.sqrt(np.sum(ratios * ratios, dtype=np.float32), dtype=np.float32)
            )
    steps = np.where(dirs < 0.0, -1, 1).astype(np.int8)
    return np.ascontiguousarray(delta), np.ascontiguousarray(steps)


_RAY_RAWKERNEL = None
_RAY_INTERIOR_RAWKERNEL = None

_RAY_KERNEL = r"""
extern "C" __global__ void raycast_blocked(
    const unsigned char* surface_bits, const float* ray_delta,
    const signed char* ray_steps, int nrays,
    int nx, int ny, int nz, unsigned short* blocked) {
  int linear = blockDim.x * blockIdx.x + threadIdx.x;
  int nvox = nx * ny * nz;
  if (linear >= nvox) return;
  if (surface_bits[linear >> 3] & (1u << (linear & 7))) {
    blocked[linear] = 0;
    return;
  }

  int ix = linear / (ny * nz);
  int rem = linear - ix * ny * nz;
  int iy = rem / nz;
  int iz = rem - iy * nz;
  unsigned int total_blocked = 0;

  for (int ray = 0; ray < nrays; ++ray) {
    float deltaDist[3] = {
      ray_delta[3 * ray + 0], ray_delta[3 * ray + 1], ray_delta[3 * ray + 2]
    };
    int step[3] = {
      (int)ray_steps[3 * ray + 0],
      (int)ray_steps[3 * ray + 1],
      (int)ray_steps[3 * ray + 2]
    };
    int coord[3] = {ix, iy, iz};
    float next[3] = {
      0.5f * deltaDist[0], 0.5f * deltaDist[1], 0.5f * deltaDist[2]
    };
    bool escaped = false;
    bool hit_surface = false;
    while (true) {
      int side = 0;
      for (int axis = 1; axis < 3; ++axis) {
        if (next[side] > next[axis]) side = axis;
      }
      next[side] += deltaDist[side];
      coord[side] += step[side];
      if (coord[0] < 0 || coord[0] >= nx ||
          coord[1] < 0 || coord[1] >= ny ||
          coord[2] < 0 || coord[2] >= nz) {
        escaped = true;
        break;
      }
      int surface_index = (coord[0] * ny + coord[1]) * nz + coord[2];
      if (surface_bits[surface_index >> 3] & (1u << (surface_index & 7))) {
        hit_surface = true;
        break;
      }
    }
    if (hit_surface && !escaped) total_blocked++;
  }
  blocked[linear] = (unsigned short)total_blocked;
}
"""




_RAY_INTERIOR_KERNEL = r"""
extern "C" __global__ void raycast_interior(
    const unsigned char* surface_bits, const float* ray_delta,
    const signed char* ray_steps, int nrays,
    int nx, int ny, int nz, unsigned char* interior) {
  int linear = blockDim.x * blockIdx.x + threadIdx.x;
  int nvox = nx * ny * nz;
  if (linear >= nvox) return;
  if (surface_bits[linear >> 3] & (1u << (linear & 7))) {
    interior[linear] = 0;
    return;
  }

  int ix = linear / (ny * nz);
  int rem = linear - ix * ny * nz;
  int iy = rem / nz;
  int iz = rem - iy * nz;
  bool all_blocked = true;

  for (int ray = 0; ray < nrays && all_blocked; ++ray) {
    float deltaDist[3] = {
      ray_delta[3 * ray + 0], ray_delta[3 * ray + 1], ray_delta[3 * ray + 2]
    };
    int step[3] = {
      (int)ray_steps[3 * ray + 0],
      (int)ray_steps[3 * ray + 1],
      (int)ray_steps[3 * ray + 2]
    };
    int coord[3] = {ix, iy, iz};
    float next[3] = {
      0.5f * deltaDist[0], 0.5f * deltaDist[1], 0.5f * deltaDist[2]
    };
    while (true) {
      int side = 0;
      for (int axis = 1; axis < 3; ++axis) {
        if (next[side] > next[axis]) side = axis;
      }
      next[side] += deltaDist[side];
      coord[side] += step[side];
      if (coord[0] < 0 || coord[0] >= nx ||
          coord[1] < 0 || coord[1] >= ny ||
          coord[2] < 0 || coord[2] >= nz) {
        all_blocked = false;
        break;
      }
      int surface_index = (coord[0] * ny + coord[1]) * nz + coord[2];
      if (surface_bits[surface_index >> 3] & (1u << (surface_index & 7))) {
        break;
      }
    }
  }
  interior[linear] = all_blocked ? 1 : 0;
}
"""

def raycast_blocked_cuda(surface, directions: np.ndarray):
    """Return blocked-ray counts as a CuPy array."""

    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on local CUDA env
        raise RuntimeError("backend='cuda' requires CuPy") from exc
    global _RAY_RAWKERNEL
    d_surface = cp.asarray(surface, dtype=cp.uint8, order="C")
    d_surface_bits = cp.packbits(d_surface.ravel(), bitorder="little")
    ray_delta, ray_steps = _precompute_dda_parameters(directions)
    d_delta = cp.asarray(ray_delta)
    d_steps = cp.asarray(ray_steps)
    if d_surface.ndim != 3:
        raise ValueError("surface must be a 3-D array")
    nx, ny, nz = d_surface.shape
    if nx * ny * nz >= 2**32:
        raise ValueError("grid is too large for the CUDA ray kernel")
    d_blocked = cp.zeros(d_surface.shape, dtype=cp.uint16)
    if _RAY_RAWKERNEL is None:
        _RAY_RAWKERNEL = cp.RawKernel(_RAY_KERNEL, "raycast_blocked")
    kernel = _RAY_RAWKERNEL
    threads = 128
    blocks = (int(nx * ny * nz) + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (d_surface_bits, d_delta, d_steps, np.int32(len(directions)),
         np.int32(nx), np.int32(ny), np.int32(nz), d_blocked),
    )
    return d_blocked



def raycast_interior_cuda(surface, directions: np.ndarray):
    """Return fixed-mode interior voxels with VMD DDA early exit.

    This is equivalent to ``blocked == nrays`` but does not compute a full
    blocked-ray count for voxels that already escaped on one direction.
    """

    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on local CUDA env
        raise RuntimeError("backend='cuda' requires CuPy") from exc
    d_surface = cp.asarray(surface, dtype=cp.uint8, order="C")
    d_surface_bits = cp.packbits(d_surface.ravel(), bitorder="little")
    ray_delta, ray_steps = _precompute_dda_parameters(directions)
    d_delta = cp.asarray(ray_delta)
    d_steps = cp.asarray(ray_steps)
    if d_surface.ndim != 3:
        raise ValueError("surface must be a 3-D array")
    nx, ny, nz = d_surface.shape
    global _RAY_INTERIOR_RAWKERNEL
    if _RAY_INTERIOR_RAWKERNEL is None:
        _RAY_INTERIOR_RAWKERNEL = cp.RawKernel(
            _RAY_INTERIOR_KERNEL, "raycast_interior"
        )
    d_interior = cp.empty(d_surface.shape, dtype=cp.uint8)
    threads = 256
    blocks = (int(nx * ny * nz) + threads - 1) // threads
    _RAY_INTERIOR_RAWKERNEL(
        (blocks,),
        (threads,),
        (
            d_surface_bits,
            d_delta,
            d_steps,
            np.int32(len(directions)),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            d_interior,
        ),
    )
    return d_interior


_RAY_INTERIOR_COUNT_KERNEL = r"""
extern "C" __global__ void raycast_interior_count(
    const unsigned char* surface_bits, const float* ray_delta,
    const signed char* ray_steps, int nrays,
    int nx, int ny, int nz, unsigned int* interior_count) {
  __shared__ unsigned int block_count;
  if (threadIdx.x == 0) block_count = 0;
  __syncthreads();

  int linear = blockDim.x * blockIdx.x + threadIdx.x;
  int nvox = nx * ny * nz;
  bool all_blocked = false;
  if (linear < nvox &&
      !(surface_bits[linear >> 3] & (1u << (linear & 7)))) {
    int ix = linear / (ny * nz);
    int rem = linear - ix * ny * nz;
    int iy = rem / nz;
    int iz = rem - iy * nz;
    all_blocked = true;

    for (int ray = 0; ray < nrays && all_blocked; ++ray) {
      float deltaDist[3] = {
        ray_delta[3 * ray + 0], ray_delta[3 * ray + 1], ray_delta[3 * ray + 2]
      };
      int step[3] = {
        (int)ray_steps[3 * ray + 0],
        (int)ray_steps[3 * ray + 1],
        (int)ray_steps[3 * ray + 2]
      };
      int coord[3] = {ix, iy, iz};
      float next[3] = {
        0.5f * deltaDist[0], 0.5f * deltaDist[1], 0.5f * deltaDist[2]
      };
      while (true) {
        int side = 0;
        for (int axis = 1; axis < 3; ++axis) {
          if (next[side] > next[axis]) side = axis;
        }
        next[side] += deltaDist[side];
        coord[side] += step[side];
        if (coord[0] < 0 || coord[0] >= nx ||
            coord[1] < 0 || coord[1] >= ny ||
            coord[2] < 0 || coord[2] >= nz) {
          all_blocked = false;
          break;
        }
        int surface_index = (coord[0] * ny + coord[1]) * nz + coord[2];
        if (surface_bits[surface_index >> 3] & (1u << (surface_index & 7))) {
          break;
        }
      }
    }
  }

  if (all_blocked) atomicAdd(&block_count, 1u);
  __syncthreads();
  if (threadIdx.x == 0 && block_count > 0) {
    atomicAdd(interior_count, block_count);
  }
}
"""

_RAY_INTERIOR_COUNT_RAWKERNEL = None
_RAY_INTERIOR_COUNT_3D_RAWKERNEL = None

_RAY_INTERIOR_COUNT_3D_KERNEL = r"""
extern "C" __global__ void raycast_interior_count_3d(
    const unsigned char* surface_bits, const float* ray_delta,
    const signed char* ray_steps, int nrays,
    int nx, int ny, int nz, unsigned int* interior_count) {
  __shared__ unsigned int block_count;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) block_count = 0;
  __syncthreads();

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  bool all_blocked = false;
  if (ix < nx && iy < ny && iz < nz) {
    int linear = (ix * ny + iy) * nz + iz;
    if (!(surface_bits[linear >> 3] & (1u << (linear & 7)))) {
      all_blocked = true;

      for (int ray = 0; ray < nrays && all_blocked; ++ray) {
        float deltaDist[3] = {
          ray_delta[3 * ray + 0],
          ray_delta[3 * ray + 1],
          ray_delta[3 * ray + 2]
        };
        int step[3] = {
          (int)ray_steps[3 * ray + 0],
          (int)ray_steps[3 * ray + 1],
          (int)ray_steps[3 * ray + 2]
        };
        int coord[3] = {ix, iy, iz};
        float next[3] = {
          0.5f * deltaDist[0],
          0.5f * deltaDist[1],
          0.5f * deltaDist[2]
        };
        while (true) {
          int side = 0;
          for (int axis = 1; axis < 3; ++axis) {
            if (next[side] > next[axis]) side = axis;
          }
          next[side] += deltaDist[side];
          coord[side] += step[side];
          if (coord[0] < 0 || coord[0] >= nx ||
              coord[1] < 0 || coord[1] >= ny ||
              coord[2] < 0 || coord[2] >= nz) {
            all_blocked = false;
            break;
          }
          int surface_index = (coord[0] * ny + coord[1]) * nz + coord[2];
          if (surface_bits[surface_index >> 3] & (1u << (surface_index & 7))) {
            break;
          }
        }
      }
    }
  }

  if (all_blocked) atomicAdd(&block_count, 1u);
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && block_count > 0) {
    atomicAdd(interior_count, block_count);
  }
}
"""


def raycast_interior_count_cuda(surface, directions: np.ndarray, *, layout: str = "3d") -> int:
    """Return the fixed-mode interior voxel count without a mask allocation."""

    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on local CUDA env
        raise RuntimeError("backend='cuda' requires CuPy") from exc
    d_surface = cp.asarray(surface, dtype=cp.uint8, order="C")
    if d_surface.ndim != 3:
        raise ValueError("surface must be a 3-D array")
    d_surface_bits = cp.packbits(d_surface.ravel(), bitorder="little")
    ray_delta, ray_steps = _precompute_dda_parameters(directions)
    d_delta = cp.asarray(ray_delta)
    d_steps = cp.asarray(ray_steps)
    nx, ny, nz = d_surface.shape
    nvox = int(nx * ny * nz)
    if nvox >= 2**32:
        raise ValueError("grid is too large for a 32-bit interior counter")

    global _RAY_INTERIOR_COUNT_RAWKERNEL, _RAY_INTERIOR_COUNT_3D_RAWKERNEL
    d_count = cp.zeros(1, dtype=cp.uint32)
    kernel_args = (
        d_surface_bits,
        d_delta,
        d_steps,
        np.int32(len(directions)),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        d_count,
    )
    if layout == "1d":
        if _RAY_INTERIOR_COUNT_RAWKERNEL is None:
            _RAY_INTERIOR_COUNT_RAWKERNEL = cp.RawKernel(
                _RAY_INTERIOR_COUNT_KERNEL, "raycast_interior_count"
            )
        threads = 256
        blocks = (nvox + threads - 1) // threads
        _RAY_INTERIOR_COUNT_RAWKERNEL((blocks,), (threads,), kernel_args)
    elif layout == "3d":
        if _RAY_INTERIOR_COUNT_3D_RAWKERNEL is None:
            _RAY_INTERIOR_COUNT_3D_RAWKERNEL = cp.RawKernel(
                _RAY_INTERIOR_COUNT_3D_KERNEL, "raycast_interior_count_3d"
            )
        # One z-column per block keeps adjacent threads contiguous in the
        # packed surface bit array while supplying x/y/z without divisions.
        # CUDA limits the z dimension of a thread block to 64.
        block = (1, 1, 64)
        blocks = (int(nx), int(ny), (int(nz) + block[2] - 1) // block[2])
        _RAY_INTERIOR_COUNT_3D_RAWKERNEL(blocks, block, kernel_args)
    else:
        raise ValueError("layout must be '1d' or '3d'")
    return int(d_count.get()[0])
