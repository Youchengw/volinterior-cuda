# Method and input guide

## Selection policy

The default `--method auto` first computes QuickSurf density and tests a
six-connected free-space partition on uniformly spaced frames. The seed is the
nearest grid point to the selected atoms' mass-weighted COM. The seed must be
free and its component must not touch any of the six grid faces. A material
seed also rejects connectivity. No virtual closure, dilation or hole filling
changes the density mask.

A successful sample screen selects connectivity. The same guard is evaluated
on every production frame. Any later failure invalidates that trajectory's
connectivity attempt. If `--fallback` was supplied, all requested frames and
the reference are recomputed with that DDA method. Otherwise the run saves the
failing frame and exits with `needs_dda_choice` (CLI exit code 2). The user then
runs `--method fixed` or `--method fuzzy --cutoff VALUE` in a new directory.

`--method connectivity` forces connectivity and stops on failure without a
fallback. `--method fixed` and `--method fuzzy` directly select DDA without
screening. `--check-only` computes only the screen, never full DDA production.
A screen-only success is not a completed trajectory run. A failed screen with
an advance fallback choice reports `screen_failed` and exits without starting
DDA; run again without `--check-only` to permit production.

All output frames share a classification definition. Connectivity includes
all enclosed free cavities; it does not silently replace them with only the
central component. `central_lumen_voxels` and `other_interior_voxels` provide
separate connectivity diagnostics. These fields are blank for DDA, where a
connectivity component is not the definition of the observable.

## DDA modes

`fixed` uses deterministic VMD-style Poisson directions and the VMD DDA crossing
and axis-tie rules. A voxel with any escaping ray becomes exterior. This also
applies to initially material voxels, reproducing the VMD 2.0.1a1 fixed-mode
material-overwrite behavior. All-blocked empty voxels become interior;
all-blocked material remains selection. CUDA scalar reduction and optional
voxel-label export use the same partition.

`fuzzy` retains the existing reanalysis definition: material is
`density >= isovalue`; free voxels are interior if `blocked_rays / nrays >= cutoff`.
The cutoff is mandatory. Decimal ceiling converts it to the exact integer
minimum: with 32 rays, 0.8/0.9/0.95 require 26/29/31 blocked rays. No default
fuzzy cutoff is selected by the command. Fractions are directional samples,
not calibrated physical probabilities. This material-preserving map is kept
explicitly distinct from VMD-version-specific probability-map zero/tolerance
and discretization conventions.

Both DDA paths use the same QuickSurf mask and traversal engine. If exact
comparison requires the directions from an installed VMD version, provide
`--ray-directions-npy rays.npy` with shape `(nrays, 3)`. Actual used directions
are saved. The generated Poisson sequence follows Linux libc behavior; no
cross-platform exact-ray guarantee is claimed without an explicit array.

References: [VMD measure volinterior](https://www.ks.uiuc.edu/Research/vmd/vmd-1.9.4/ug/node139.html),
[VMD classification source documentation](https://www.ks.uiuc.edu/Research/vmd/doxygen/MeasureVolInterior_8C-source.html).

## Inputs

An extraction input directory has:

- `coordinates.npy`: floating-point `(frames, atoms, 3)`, angstrom, in atom/frame table order.
- `atoms.csv`: `atom_index` (0..N-1), `atom_name`, `mass_Da`; optional `element`.
- `frames.csv`: `frame_index` (0..T-1), unique increasing integer `full_frame`, increasing `time_ns`.

The current minimal reanalysis format is supported. A `.incomplete` marker is
rejected. If legacy `metadata.json`/`validation.json` exist, their completion,
units and axes declarations are checked too. Array/table shapes, IDs and positive
masses/radii are checked, and coordinates must be finite in every measured frame.
The adapter records file paths/sizes/mtimes; it does not rehash multi-GB inputs.
Without metadata the units are the explicit format contract above, not inferred.

Default radii use the bundled VMD-like atom-name/element table; unknown atoms
fail. `--radii-npy` overrides them in exactly the selected atom order. Direct
MDAnalysis input first uses valid topology radii, falling back to that table.
Masses always refer to selected atoms. Protein/genome/solvent inclusion is
controlled by the supplied extraction or `--selection`; the package does not
reinterpret a selected genome as protein.

For raw NumPy coordinates, explicit radii and masses arrays are mandatory.
For direct trajectories, MDAnalysis supplies masses. `--dt-ns` is required for
both formats; times are `frame_index * dt_ns`. Extracted frame times are used
unchanged. Sampling is uniform in ordered frame positions, which is uniform in
time only for equally spaced frames.

The first available source frame is the reference; it is additionally measured
when omitted by `--frames`, but excluded from the selected plotting data. For a
full canonical extraction this is original trajectory frame 0. The reference
frame is recorded. A zero reference compartment volume produces a blank percent
change, while the raw and absolute-change volumes remain available.

## Grid and labels

Resolution 12.5 corresponds to a dimensionless radius scale 2.5. Spacing is in
angstrom; volume is voxel count times spacing cubed. The density kernels retain
their historical `radius_scale_A` property name, although the scale itself is
dimensionless. The grid encloses the selected coordinates with QuickSurf padding.

Saved arrays are indexed `[grid_x, grid_y, grid_z]` in C order. Each row of
`counts.csv` records origin, shape and spacing; coordinates are
`origin + grid_index * spacing`. Labels are int8: exterior 5, interior 0,
selection -5. A method's selection volume is its shell; `density_voxels` always
records the original density mask separately. Connectivity and fuzzy preserve
that mask; strict fixed DDA can reclassify part of it as exterior.

During processing, scalar counts are flushed to `counts.partial.csv`. On success
it becomes `counts.csv`; `data.csv` contains the final selected-frame volume
series. Failed connectivity attempts are kept separately. Labels from a
connectivity attempt are discarded if the entire run switches to DDA, so saved
final labels always agree with the final method. No final `data.csv` is emitted
when the program is waiting for a DDA choice.

## Python API

```python
import numpy as np
from volinterior_cuda import VolInteriorConfig, measure_frame

result = measure_frame(
    np.load("frame.coords.npy"),
    np.load("radii_A.npy"),
    np.load("masses_Da.npy"),
    method="connectivity",
    config=VolInteriorConfig(backend="cuda"),
    return_labels=True,
)
print(result.counts["closure_pass"])
print(result.counts["lumen_volume_nm3"])
print(result.grid.shape, result.labels.dtype)
```

The low-level function returns measurements even when connectivity fails; callers
must inspect `closure_pass`. The CLI and `run_trajectory` implement the automatic
screening/choice/restart policy. API `method="fixed"` selects strict fixed DDA;
`method="fuzzy", cutoff=0.9` selects the explicit fuzzy criterion.
