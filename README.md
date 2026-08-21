# volinterior-cuda

VMD-compatible GPU acceleration for Gaussian-density interior-volume analysis
of molecular-dynamics trajectories.

The package reimplements the `measure volinterior` workflow in Python and
CuPy: VMD-style QuickSurf density, isovalue thresholding, deterministic VMD
Poisson rays, and fixed-mode DDA interior classification. The scalar CUDA path
keeps masks on the GPU and returns voxel counts, making it suitable for long
trajectories.

## Current status

This repository is being prepared from the validated AAV8 capsid workflow.
The implementation has been checked against the VMD reference for 467
production frames with identical scalar voxel results. On the development RTX
4060 Laptop benchmark, the optimized scalar path has a median time of about
0.30 s/frame versus about 7.28 s/frame for the VMD production run. These are
reference results, not a guarantee for every GPU or trajectory.

The code is intentionally separated from the AAV8 analysis project. AAV8
topologies, trajectories, production outputs, and paper-specific scripts are
not part of this repository.

## Installation

The core package requires Python 3.10+ and NumPy. CUDA and trajectory support
are optional:

```bash
python -m pip install -e ".[trajectory,cuda,dev]"
```

The `cuda` extra uses the CuPy CUDA 12 package. Choose the CuPy package matching
the CUDA runtime on the target cluster if a different runtime is required.

## Examples

For a coordinate/radius pair:

```bash
volinterior-cuda \\
  --coordinates-npy frame.coords.npy \\
  --radii-npy atom_radii.npy \\
  --backend cuda --resolution 12.5 --spacing 1.0 \\
  --isovalue 0.5 --mode fixed --classifier dda \\
  --output results/frame0.json
```

For an MDAnalysis-readable trajectory:

```bash
volinterior-cuda \\
  --topology capsid.gro --trajectory production.xtc \\
  --selection protein --backend cuda \\
  --frames 0,100,200 --output results/volinterior.json
```

For resumable scalar trajectory production with detailed metadata, use
`scripts/run_volinterior_cuda.py`. The runner supports explicit radii source
selection (`auto`, `topology`, `elements`, or `names`), optional atom-count
guards, selection or unit-cell domains, and checkpointed CSV output.

## Validation and benchmark

```bash
PYTHONPATH=. pytest -q
python scripts/profile_volinterior_cuda.py \\
  --topology capsid.gro --trajectory production.xtc \\
  --output outputs/profile --frames 0,100,200
```

The profiler reports trajectory extraction, grid construction, density,
thresholding, ray generation, DDA, and reduction timings separately. Results
and validation artifacts should be kept outside the source tree.

## Scientific and numerical scope

The default configuration follows the VMD-compatible settings used in the
reference validation: resolution 12.5, 1 Å spacing, isovalue 0.5, 32 VMD
Poisson rays, and fixed DDA classification. Resolution, spacing, isovalue,
ray count, atom radii, selection, and periodic-boundary handling should be
reported and sensitivity-tested for each new system.

For `domain=selection`, unwrap a capsid that crosses periodic boundaries before
running. `domain=box` uses the per-frame unit-cell lengths and measures the
full box domain, which is a different scientific definition.

## Provenance and licensing

The implementation is an independent Python/CuPy implementation of the
VMD-compatible workflow and documents the VMD/CUDAMDFF provenance in the
source and documentation. A final open-source license and any required VMD
attribution must be confirmed before this repository is published.

