# Handoff — 2026-10-07

This branch proposes replacing the GitHub main branch's old runner with a focused,
standalone screened-connectivity workflow. It was developed in an isolated
checkout; the two older local research repositories and the capsid reanalysis
outputs were not changed.

Default: uniformly screen frames; if closed, use CUDA connectivity with an
all-frame leak guard. If a sampled or later frame is not enclosed, save the
reason and ask the user to choose fixed or fuzzy DDA. An optional advance choice
allows automatic full-trajectory recomputation. Fuzzy requires an explicit
cutoff. There is no mixed-method final series or automatic pore closure.

Reusable API: measure_frame and workflow.run_trajectory. Inputs: minimal atomic
extraction, NumPy coordinate/radius/mass arrays, or MDAnalysis trajectory.
The latest extraction format without metadata sidecars is supported, as is the
older format with sidecars. Existing incomplete markers are honored.

No test source, trajectories, development outputs or legacy pore modules are
retained in this replacement. Numerical evidence and limits are documented in
docs/validation.md. No full trajectory production run is part of this change.
