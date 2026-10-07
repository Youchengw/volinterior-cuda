# volinterior-cuda

CUDA-accelerated capsid lumen and shell volumes from molecular trajectories.
Uses QuickSurf density with connectivity or fixed/fuzzy DDA classification;
no VMD executable is required.

## Install

Python 3.10+, an NVIDIA GPU and a CUDA 12-compatible driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cuda,trajectory]'
```

## Run

```bash
volinterior-cuda \
  --topology capsid.gro --trajectory capsid.dcd \
  --selection protein --dt-ns 1.2 \
  --check-frames 9 \
  --resolution 12.5 --isovalue 0.5 --spacing 1.0 \
  --output-dir results/volume
```

Set `--dt-ns` to your saved-frame interval. Coordinates must already form an
intact, unwrapped capsid. All frames are processed by default.

The program screens 9 evenly spaced frames using the protein mass-weighted COM.
If the central free-space component is enclosed, it uses six-connected
classification and checks every subsequent frame. A leak stops the run for a
fixed/fuzzy DDA choice. An advance fallback choice recomputes the entire frame
set with DDA if needed, keeping one method throughout the final results.

Add these options to the command above:

| Option | Purpose |
|---|---|
| `--check-only` | Screen sampled frames without full processing |
| `--method fixed` | Run VMD-style fixed DDA directly |
| `--method fuzzy --cutoff 0.9` | Run fuzzy DDA; cutoff is explicitly chosen |
| `--fallback fixed` | Preselect fixed DDA if connectivity fails |
| `--fallback fuzzy --cutoff 0.9` | Preselect fuzzy DDA if connectivity fails |
| `--save-labels` | Also save per-frame voxel labels |

Use a new output directory for each run. See `volinterior-cuda --help` for all
options, including NumPy inputs, frame selection and explicit CPU execution.

## Extracted inputs and results

An existing extraction can be used instead of topology/trajectory input:

```bash
volinterior-cuda --input data/protein_extraction --output-dir results/volume
```

The extraction directory contains:

- `coordinates.npy`: floating-point `(frames, atoms, 3)` coordinates in Å.
- `atoms.csv`: ordered `atom_index` (0-based), `atom_name`, `mass_Da`; optional `element`.
- `frames.csv`: ordered `frame_index` (0-based), increasing `full_frame` and `time_ns`.

`data.csv` contains lumen, shell and outer volumes and changes from the first
input frame. `counts.csv` contains voxel counts; JSON files record run settings
and checks. Connectivity and fuzzy preserve the density-defined shell; fixed
DDA can reclassify material as exterior. These methods need not give identical
volumes; fuzzy uses a blocked-ray fraction, not exact VMD probability-map parity.

## Performance

Measured on 2026-10-07: RTX 4060 Laptop GPU, Ryzen 7 7735H (16 CPU threads),
Linux, CUDA 12.9 and CuPy 14.1.1. AAV8 protein: 492,780 atoms, approximately
43 million voxels; resolution 12.5, spacing 1 Å, isovalue 0.5, 32 rays for DDA.

| Method | Median (s/frame) | Observed range (s/frame) |
|---|---:|---:|
| CUDA connectivity | 0.116 | 0.115–0.119 |
| CUDA fixed DDA | 0.323 | 0.307–0.325 |
| CUDA fuzzy DDA (cutoff 0.9) | 2.564 | 2.517–2.585 |
| VMD 2.0.1a1 fixed (GPU density + CPU DDA) | 7.235 | 6.683–7.301 |

Each method had one warm-up, then three repetitions of frames 0, 400 and 800
(9 measured calls). Coordinates, atom selection and radii matched across tools.
Times cover density calculation and classification, excluding startup, input
loading and file export. Python returns scalar counts; the native
[VMD command](https://www.ks.uiuc.edu/Research/vmd/doxygen/TclMeasure_8C-source.html)
also creates its volume/gradient map. This is an analysis-call comparison, not
an isolated-kernel benchmark or full-trajectory throughput measurement.
Connectivity, fixed and fuzzy use different classification definitions.

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
