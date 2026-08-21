# volinterior-cuda method

`volinterior-cuda` follows the VMD `measure volinterior` workflow while keeping
trajectory production in Python and CUDA: Gaussian atom density, isovalue
thresholding, deterministic VMD-style rays, and fixed-mode DDA classification.

## VMD parameter mapping

VMD converts `-res` to `radscale = 0.2 * resolution`. The package exposes this
as `radius_scale_A`; `-spacing`, `-isovalue`, `-nrays`, and `-probmap` map to
`spacing_A`, `isovalue`, `nrays`, and fuzzy/probability modes. The default ray
set is the VMD-compatible Poisson sampler with seed `512346` and 40 candidates.

## CUDA implementation

The default density path uses a uniform cell list, precomputed Gaussian
coefficients, four-z-voxel CUDA evaluation, and CUDA `__expf`, following the
publicly documented VMD CUDAMDFF behavior. The scalar production DDA uses an
early exiting kernel and a three-dimensional launch so each thread receives
its voxel coordinates without integer index divisions. Deterministic ray
sets are cached by parameter tuple and returned as copies.

The package keeps density, surface, and interior masks on the GPU for scalar
trajectory production and copies back only scalar voxel counts. Diagnostic map
mode materializes masks on the host and is intentionally slower.

## Validation protocol

For each new system, record:

- topology and trajectory paths or hashes;
- atom selection and observed atom count;
- radius source and radius range;
- domain (`selection` or `box`) and periodic-boundary handling;
- resolution, spacing, isovalue, ray count, ray scheme, and seed;
- GPU model, CUDA runtime, CuPy version, and warmup policy;
- median, P95, and memory use over representative frames.

Validation should compare scalar voxel counts, grid shape/origin, and volume to
a trusted reference when available. Synthetic sphere, shell, and disconnected
components are useful topology tests. Sensitivity should cover resolution,
isovalue, spacing, ray count, and atom-radius source.

## Reference AAV8 result

The development AAV8 workflow used 467 production frames and found exact
non-timing agreement with the VMD scalar results. On the local RTX 4060 Laptop,
the optimized CUDA-DDA runner had a median of approximately `0.300 s/frame`,
while the VMD production reference had a median of approximately `7.283
s/frame`. These numbers are a reproducibility reference for the development
system, not a universal hardware guarantee.

## Provenance

The implementation is independent Python/CuPy code that reproduces the VMD
workflow semantics. VMD and CUDAMDFF remain separate projects; citations,
source provenance, and any required attribution must be retained in releases.
