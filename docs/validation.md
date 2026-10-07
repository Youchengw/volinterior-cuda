# Validation record — 2026-10-07

These checks were executed temporarily during the 0.2 replacement. Test source,
fixtures and generated numerical outputs are not distributed. The record states
what was checked; it is not a claim that all trajectories, operating systems or
VMD versions have been validated.

## Environment

Linux x86-64, Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6,
CuPy 14.1.1, CUDA runtime 12.9, MDAnalysis 2.10.0, NVIDIA GeForce RTX 4060
Laptop GPU. CUDA compilation and execution were exercised, not mocked.

## Classification and method selection

- Closed, open and empty synthetic voxel masks: CPU and CUDA connectivity
  labels agreed with an independent boundary-seeded SciPy propagation.
- Fixed and fuzzy (`cutoff=0.9` and `1.0`) DDA: CPU/CUDA labels agreed voxel by
  voxel; optimized CUDA scalar counts agreed with exported-label counts.
- Atom-based closed/open shells: successful screening, screen-only operation,
  sampled leaks, forced-connectivity rejection and default DDA-choice stops
  were exercised. A choice stop produced no final `data.csv`.
- Three-frame closed/open/closed sequence with only endpoints screened:
  the middle leak was detected. Both preselected fixed and fuzzy fallbacks
  recomputed every frame; final tables contained only the chosen DDA method.
  Late fixed fallback matched a separate forced-fixed run exactly.
- Exported labels matched compartment counts. Late fallback discarded the
  connectivity labels and retained only final DDA labels. An omitted input
  frame 0 was still measured as the reference without entering selected data.
- Failed screen-only operation with a preselected fallback did not start DDA
  production and reported `screen_failed`. An open-input CLI run without a
  fallback exited with code 2 and displayed both user choices.

## Real AAV8 extraction

The 492,780-atom protein selection from the 801-frame, 0–960 ns extraction was
sampled at frame IDs 0,100,200,300,400,500,600,700,800. The automatic run screened
0,400,800 first and guarded all remaining selected frames. All nine were closed.

All 27 frame/compartment rows agreed with the existing reanalysis connectivity
series: maximum CSV floating-point difference was below 1e-10 nm³ for volume
and below 1e-10 percentage points for relative change. New connectivity
measurements took about 0.11–0.12 s/frame after initialization on this GPU;
the cold first frame took 6.66 s. These figures exclude label export and are
not an end-to-end throughput guarantee.

Fuzzy, 32 rays, explicit cutoff 0.9, resolution 12.5, spacing 1 Å, isovalue 0.5:

| Frame | Lumen (nm³) | Raw-density shell (nm³) |
|---|---:|---:|
| 0 | 1797.971 | 7918.570 |
| 400 | 2054.552 | 8001.183 |
| 800 | 2054.091 | 8026.881 |

These three anchors matched the prior reanalysis values. No complete new
801-frame production series was run as part of this replacement.

## VMD fixed reference

One real production frame at 400.8 ns was checked against the saved VMD
2.0.1a1 fixed-mode label map. Inputs used the reference's atom radii and
captured 32-ray array, resolution 12.5, spacing 1 Å and isovalue 0.5.

The 353 × 354 × 349 grid contained 43,611,738 voxels. **All voxel labels
matched**, and the separate optimized scalar path returned the same counts:

| Class | Voxels |
|---|---:|
| Interior | 2,056,768 |
| Selection after fixed DDA | 7,730,980 |
| Exterior | 33,823,990 |
| Raw density mask before DDA | 7,999,495 |

Reference identity:

- VMD version: 2.0.1a1, Linux AMD64.
- VMD executable SHA-256: `94411f53864502be25bfecf5eca2190ef4fa5ce98c4a66a36cc4d7ad24882e5c`.
- Reference label DX SHA-256: `af6cddb8be521b053d1e865cdee5b3ce3e960caff67d015774abfe1d2a28c2a8`.
- Captured rays, float32 little-endian array SHA-256: `d2689cd76fbbbe6cdf1b46ffd8fc7951a0b16b0c44e8aec1fa0f7051b85c6b10`.

The DX origin is rounded in its text header; its difference from the computed
origin was below 5.1e-5 Å. Shape and all labels agreed. This evidence applies to
this frame and matched rays/radii; it is not full-trajectory or arbitrary-VMD
bitwise certification. Fuzzy probability-map discretization was not claimed or
validated as bitwise equivalent to VMD. Connectivity is not assumed equivalent
to fixed DDA, even for a closed density surface.

## Packaging

The wheel built successfully with isolated build dependencies. It was installed
outside the source checkout and exercised from another working directory:
installed console command, `python -m volinterior_cuda`, CPU measurement and
CUDA measurement of a real extraction frame all passed. The wheel contains
LICENSE/NOTICE and excludes test sources, trajectories and generated results.
Python syntax and `git diff --check` passed. CPU CI covers installation and wheel
construction on Python 3.10–3.12; it does not perform CUDA numerical validation.
