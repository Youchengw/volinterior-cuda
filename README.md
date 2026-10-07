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

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
