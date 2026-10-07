# volinterior-cuda

CUDA capsid voxel classification with **screened connectivity** and a
**user-selected fixed or fuzzy DDA fallback**. No VMD executable is needed.

The default workflow samples evenly spaced trajectory frames, checks whether
the free component at the protein mass-weighted COM reaches the grid boundary,
and uses fast six-connected classification if every sample is enclosed. Every
subsequent frame is checked too. If a later frame leaks, the program stops for a
DDA choice or restarts **the entire trajectory** using a previously specified
fallback. A final table never mixes connectivity and DDA definitions.

## Install

Python 3.10+; NVIDIA GPU and a compatible CUDA 12 driver for the CUDA path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cuda,trajectory]'
volinterior-cuda --help
```

The `cuda` extra includes CUDA runtime libraries and headers. CPU reference
execution is available with `--backend cpu`; install with `pip install -e .`
when GPU/trajectory support is not needed. CUDA is the default and an unavailable
GPU causes an error instead of silently starting a slow CPU calculation.

## Start with automatic connectivity screening

An extraction directory contains `coordinates.npy`, `atoms.csv`, and
`frames.csv` (see the [input format](docs/volinterior_cuda.md#inputs)). It is read
with memory mapping, without loading the whole trajectory into RAM.

```bash
volinterior-cuda \
  --input data/protein_extraction \
  --check-frames 9 \
  --resolution 12.5 --isovalue 0.5 --spacing 1.0 \
  --output-dir results/auto_run
```

All available frames are processed by default. `--check-frames 9` includes
both endpoints and seven uniformly spaced positions in the ordered frame list.
For an 801-frame trajectory the sampled IDs are 0,100,...,800. To screen without
running the trajectory, add `--check-only`. No cutoff is needed for connectivity.

If connectivity fails, diagnostics are saved and the command exits with code 2,
showing these choices. Reuse the same input, frame selection and density
parameters, and give each attempt a new output directory:

```bash
# VMD fixed DDA: all rays must be blocked; preserves VMD material overwrite.
volinterior-cuda \
  --input data/protein_extraction \
  --method fixed --nrays 32 \
  --resolution 12.5 --isovalue 0.5 --spacing 1.0 \
  --output-dir results/dda_fixed

# Fuzzy DDA: choose an explicit blocked-ray fraction (0.9 is an example).
volinterior-cuda \
  --input data/protein_extraction \
  --method fuzzy --cutoff 0.9 --nrays 32 \
  --resolution 12.5 --isovalue 0.5 --spacing 1.0 \
  --output-dir results/dda_fuzzy_0p9
```

For unattended execution, make that choice in advance with `--fallback fixed`
or `--fallback fuzzy --cutoff 0.9` on the automatic command. If every frame is
closed, connectivity is used; otherwise the chosen DDA method is used for **all**
output frames. An unexpected error, invalid input or out-of-memory condition is
reported as an error, never interpreted as a pore leak.

## Direct trajectory or NumPy input

```bash
volinterior-cuda \
  --topology capsid.gro --trajectory capsid.dcd \
  --selection protein --dt-ns 1.2 \
  --check-frames 9 --output-dir results/from_dcd

volinterior-cuda \
  --coordinates-npy coordinates.npy \
  --radii-npy radii_A.npy --masses-npy masses_Da.npy \
  --dt-ns 1.2 --check-frames 9 \
  --output-dir results/from_numpy
```

Coordinates must already form an intact, unwrapped capsid. No additional
alignment or wrapping is applied. `--dt-ns` is explicit for these inputs and
overrides unreliable trajectory-header clocks. COM always uses positive masses,
not an unweighted atom centre. Supply `--radii-npy` when names/elements are missing
or when reproducing a particular VMD topology's radii.

Use `--frames 0,400,800` for a selected subset. Changes retain the first frame of
the input as their reference, even if it is omitted from the selected output.
`--device 0` selects the GPU. `--save-labels` additionally writes per-frame int8
arrays with `-5 = selection`, `0 = interior`, and `5 = exterior`; this can use
substantial disk space. The installed command and
`python scripts/run_volinterior_cuda.py` accept the same options.

## Results and scientific definitions

A successful run writes `counts.csv`, `data.csv`, `reference.csv`, and
`metadata.json`/`validation.json`. Screening is recorded in
`connectivity_check.csv` and `screening.json`. A failed late connectivity attempt
is retained as `connectivity_attempt.csv`; it is not a final volume series.
Input radii, masses and any used ray directions are saved. Existing directories
are never overwritten, and interrupted runs do not resume automatically.

`data.csv` has one row per output frame and compartment (`lumen`, `shell`,
`outer`), with nm³/Å³ values and changes relative to the input reference frame.
Every row records its classification method and definition. In every method,
`outer = lumen + shell`, but the definition of shell matters:

| Method | Lumen | Shell |
|---|---|---|
| connectivity | All free components disconnected from the grid boundary | Raw QuickSurf density mask |
| fixed DDA | Free voxels with every ray blocked | Material voxels with every ray blocked; escaping material becomes exterior |
| fuzzy DDA | Free voxels with blocked fraction >= explicit cutoff | Raw QuickSurf density mask |

Connectivity can be much faster but is **not generally identical to DDA**.
Closure concerns a six-connected voxel grid at a COM seed, not proof of a
continuous molecular seal. Fuzzy retains the reanalysis's explicit material
mask and integer-count threshold; it is not advertised as bitwise identical to
every VMD `-probmap -discretize` implementation. Definitions and edge cases are
in [the method guide](docs/volinterior_cuda.md).

## Validation scope

The replacement uses the low-level QuickSurf/grid/ray kernels exercised by the
AAV8 reanalysis. Temporary checks exercise CPU/CUDA classification agreement,
closed/open synthetic shells, user-choice stops, both preselected DDA fallbacks,
and a leak appearing between screened frames. Numerical evidence and limits are
recorded in [validation notes](docs/validation.md).

Source does not include test scripts, trajectories, results, pore-development
experiments or workstation-specific paths. CI checks package installation,
entry-point availability and wheel construction; it does not claim numerical
CUDA verification. Legacy local research trees remain separate.

## Migration from 0.1

This is a focused replacement of the earlier runner/profiler interface.
`--method` replaces the old mode/classifier combination, `--output-dir` replaces
`--output`, and scalar results use CSV tables plus a run record. The public
single-frame API is `measure_frame`; trajectory orchestration is
`volinterior_cuda.workflow.run_trajectory`. See
[the method guide](docs/volinterior_cuda.md#python-api) for a minimal Python example.
No historical scientific results are relabelled as new-method results.

Original code is BSD-3-Clause licensed; see [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md). This independent implementation does not redistribute
or relicense VMD.
