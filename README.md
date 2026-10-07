# Augmented Lagrangian-based preconditioners for Fictitious Domain solvers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23208420.svg)](https://zenodo.org/records/23208420)

This repository contains application codes demonstrating augmented
Lagrangian-based preconditioners for fictitious domain solvers, based on the
[deal.II library](https://www.dealii.org).

## Citation

If you use this code in your research, please cite: [*Scalable augmented Lagrangian preconditioners for fictitious domain problems*](https://www.sciencedirect.com/science/article/pii/S0045782525007947)
```bibtex
@article{BENZI2026118522,
title = {Scalable augmented Lagrangian preconditioners for fictitious domain problems},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {450},
pages = {118522},
year = {2026},
issn = {0045-7825},
doi = {10.1016/j.cma.2025.118522},
author = {Michele Benzi and Marco Feder and Luca Heltai and Federica Mugnaioni},
keywords = {Preconditioning, Iterative solvers, Fictitious domain method, Non-matching meshes, Finite element method}
}
```
or the following preprint (to appear in CAMWA)
[*Augmented Lagrangian preconditioners for fictitious domain formulations of elliptic interface problems*](https://arxiv.org/abs/2603.12993)
```bibtex
@misc{benzi2026augmentedlagrangianpreconditionersfictitious,
      title={Augmented Lagrangian preconditioners for fictitious domain formulations of elliptic interface problems}, 
      author={Michele Benzi and Marco Feder and Luca Heltai and Federica Mugnaioni},
      year={2026},
      eprint={2603.12993},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2603.12993}, 
}
```


## License

The code in this repository is licensed under the
[GNU General Public License, version 3 only](LICENSE) (`GPL-3.0-only`).

## Prerequisites

The examples require:
- **CMake** version >= 3.13.4.
- One of the following compilers (with C++ 17):
  -  **gcc** version  >= 11.4.0
  -  **clang** version >= 15
- **openMPI** version  >= 4.0.3
- **Trilinos** version >= 14.4.0
- **deal.II** version **9.8 or newer**. The tested version is **9.8.0-pre**
  (a development build of deal.II 9.8).
- **p4est**, **muparser**, and **UMFPACK**.

deal.II must be configured with `DEAL_II_WITH_TRILINOS`,
`DEAL_II_WITH_P4EST`, `DEAL_II_WITH_MUPARSER`, and `DEAL_II_WITH_UMFPACK`
enabled.

## Building

With deal.II installed and configured as above:

```bash
git clone https://github.com/fdrmrc/fictitious_domain_AL_preconditioners.git
cd fictitious_domain_AL_preconditioners && mkdir build &&
cmake  build -DDEAL_II_DIR=/path/to/deal.II && make -j 4
```

Replace `/path/to/deal.II` with your installation path and adjust `4` to the
desired number of build jobs. The executables are generated in `build/`:

| Executable | Problem | Parameter files |
| --- | --- | --- |
| `immersed_laplace` | Laplace equation with an internal constraint on a codimension-one immersed domain | [parameters/](parameters/) |
| `stokes_immersed_boundary` | Stokes problem with a codimension-one immersed body | [parameters_stokes.prm](parameters_stokes.prm), [parameters_stokes_3d.prm](parameters_stokes_3d.prm) |
| `elliptic_interface` | Scalar elliptic interface problem with a jump in coefficients | [parameters_elliptic_interface/](parameters_elliptic_interface/) |
| `elliptic_interface_elasticity` | Elasticity interface problem with different material properties | [elasticity.prm](parameters_elliptic_interface/elasticity.prm) |
| `nitsche_bcs` | Boundary constraints imposed using Lagrange multipliers | [parameters_nitsche.prm](parameters_nitsche.prm) |



Copy the appropriate parameter file and pass the copy to the executable. For example, useful settings within `subsection Elliptic Interface Problem` include:

- `Grid generation`: background and immersed geometries.
- `Refinement and remeshing`: initial refinements and `Refinemented cycles`.
  Reduce these values for a smaller test run.
- `AL preconditioner`: classical or modified AL formulation, mass-inverse
  approximation, etc.
- `Inner solver control` and `Outer solver control`: solver tolerances and
  iteration limits.
- `Output directory`: destination for outputs that use this setting.

Both applications write the resolved parameters to `used_parameters.prm` in
the working directory, overwriting it on subsequent runs. Some mesh outputs
are also written directly to the working directory.


## Authors and Contact

This repository is developed and maintained by:
- [Marco Feder](https://www.math.sissa.it/users/marco-feder) ([@fdrmrc](https://github.com/fdrmrc)), Numerical Analysis Group, Pisa - Università di Pisa, IT
- [Federica Mugnaioni](https://numpi.dm.unipi.it/people/federica-mugnaioni/) ([@federica-mugnaioni](https://github.com/federica-mugnaioni)), Numerical Analysis Group, Pisa - Scuola Normale Superiore, Pisa, IT

For inquiries or special requests, you can either contact the authors by email or open an issue.
