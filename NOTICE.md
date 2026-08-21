# Provenance and third-party attribution

`volinterior-cuda` is an independent Python/CuPy implementation of the VMD
`measure volinterior` workflow. It does not redistribute VMD source code. The
implementation is designed to reproduce the relevant VMD semantics, including
VMD-style Poisson rays and the GPU Gaussian-density behavior documented by the
VMD CUDAMDFF implementation.

The original code in this repository is distributed under the BSD 3-Clause
License in [LICENSE](LICENSE). VMD and CUDAMDFF are separate projects and are
not relicensed by this repository.

References:

- [VMD](https://www.ks.uiuc.edu/Research/vmd/)
- [VMD CUDAMDFF source documentation](https://www.ks.uiuc.edu/Research/vmd/doxygen/CUDAMDFF_8cu-source.html)

