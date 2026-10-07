# Development

Keep reusable classification and trajectory selection code in `volinterior_cuda/`.
The script entry point delegates to the package CLI. Preserve LICENSE and NOTICE.

For numerical changes, exercise deterministic closed/open masks on CPU and CUDA,
small atom-based shells, screening failure, both DDA choices and a late leak.
Check a small real input against an appropriate saved reference before making
compatibility claims. These development checks are temporary; do not add retained
test scripts, large fixtures, generated results or GPU caches to this source tree.
Record evidence and its limits in `docs/validation.md`.

Packaging checks: install a built wheel outside the checkout, run the CLI help,
check `git diff --check`, and inspect wheel contents. CPU-only CI verifies packaging;
CUDA numerical checks must be performed on an available NVIDIA workstation.

Do not silently mix connectivity, fixed DDA and fuzzy definitions within a final
trajectory. A late fallback recomputes the full frame set and reference. Default
failure leaves the DDA choice to the user; do not replace it with an assumed cutoff.
