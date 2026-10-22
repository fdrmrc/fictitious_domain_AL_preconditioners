# Augmented Lagrangian-based preconditioners for Fictitious Domain solvers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23208420.svg)](https://zenodo.org/records/23208420)

This repository contains application codes with augmented Lagrangian preconditioners for several fictitious domain discretizations, based on the [deal.II library](https://www.dealii.org).

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

deal.II must be configured with the flags `DEAL_II_WITH_TRILINOS`, `DEAL_II_WITH_P4EST`, `DEAL_II_WITH_MUPARSER`, and `DEAL_II_WITH_UMFPACK` enabled.

## Building and running

With deal.II installed and configured as above:

```bash
git clone https://github.com/fdrmrc/fictitious_domain_AL_preconditioners.git
cd fictitious_domain_AL_preconditioners
mkdir build
cd build
cmake -DDEAL_II_DIR=/path/to/deal.II ..
make -j <N>
```

Replace `/path/to/deal.II` with your installation path and adjust `N` to the
desired number of build jobs. The executables are generated in the `build/` directory. The sources of the examples are in [examples/](examples/), and the shared headers in [include/](include/).

The following table provides a list of target executables. The *Paper* column links lead to the full references and BibTeX entries below. 

| Executable | Problem | Parameter files | Paper |
| --- | --- | --- | --- |
| `immersed_laplace` | Laplace equation with an internal constraint on a codimension-one immersed domain | [parameters/](parameters_immersed_laplace/) | [1 - CMAME](#paper-scalable) |
| `stokes_immersed_boundary` | Stokes problem with a codimension-one constraints | [parameters_stokes.prm](parameters_stokes/parameters_stokes.prm), [parameters_stokes_3d.prm](parameters_stokes/parameters_stokes_3d.prm) | [1 - CMAME](#paper-scalable) |
| `elliptic_interface` | Scalar elliptic interface problem with a jump in coefficients | [parameters_elliptic_interface/](parameters_elliptic_interface/) | [2 - CAMWA](#paper-elliptic_interface) |
| `elliptic_interface_elasticity` | Immersed elasticity interface problem with different material properties on subdomains | [parameters_elasticity.prm](parameters_elasticity/parameters_elasticity.prm) | [2 - CAMWA](#paper-elliptic_interface) |
| `babuska_problem` | Boundary constraints imposed using Lagrange multipliers a' la Babuska | [parameters_babuska_2D.prm](/parameters_babuska/parameters_babuska_2D.prm), [parameters_babuska_3D.prm](/parameters_babuska/parameters_babuska_3D.prm) | [3 - arXiV](#paper-unified) |
| `variational_elliptic_interface` | Scalar elliptic interface problem with a jump in coefficients (with variational augmentation) | [parameters_elliptic_interface_variational.prm](/parameters_elliptic_interface/variational/parameters_elliptic_interface_variational.prm) | [3 - arXiV](#paper-unified) |
| `variational_immersed_elasticity` | Immersed elasticity interface problem with different material properties on subdomains (with variational augmentation) | [parameters_elasticity_variational.prm](/parameters_elasticity/parameters_elasticity_variational.prm) | [3 - arXiV](#paper-unified) |


Copy the appropriate parameter file and pass the copy to the executable. Most of the examples include the following sections:

- `Grid generation`: background and immersed geometries.
- `Refinement and remeshing`: initial refinements and `Refinemented cycles`.
  Reduce these values for a smaller test run.
- `AL preconditioner`: classical or modified AL formulation, mass-inverse
  approximations, etc.
- `Inner solver control` and `Outer solver control`: solver tolerances.
- `Output directory`: destination for outputs that use this setting.

Applications write the resolved parameters to `used_parameters.prm` in the working directory, overwriting it on subsequent runs.

## Tutorial

The screencast below walks through the whole workflow: cloning the repository, configuring it with CMake, building the executables, and running one of the examples with its parameter file.

https://github.com/user-attachments/assets/e9dd8654-6cb4-44f0-8546-bda1b7c772b2

> [!NOTE]
> GitHub renders the link above as an inline video player. If you are reading this file outside GitHub, open the link in a browser instead.

## Testing

A small regression suite checks that the preconditioned outer solver of some of the examples converges within a prescribed range of iterations at every refinement cycle. After building the examples, run it from the `build/` directory:

```bash
ctest --output-on-failure
```

The tests, their parameter files, and the allowed iteration ranges are in [tests/](tests/).

## Citations

If you use one of the codes above in your research, please cite the relevant paper among the following ones:

<a id="paper-scalable"></a>

- **[1]** [*Scalable augmented Lagrangian preconditioners for fictitious domain problems* (CMAME)](https://www.sciencedirect.com/science/article/pii/S0045782525007947)
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

<a id="paper-elliptic_interface"></a>

- **[2]** [*Augmented Lagrangian preconditioners for fictitious domain formulations of elliptic interface problems* (to appear in CAMWA)](https://arxiv.org/abs/2603.12993)
```bibtex
@article{benzi2026augmentedlagrangianpreconditionersfictitious,
  title         = {Augmented {L}agrangian preconditioners for fictitious domain formulations of elliptic interface problems},
  author        = {Michele Benzi and Marco Feder and Luca Heltai and Federica Mugnaioni},
  year          = {2026},
  eprint        = {2603.12993},
  journal       = {arXiv:2603.12993},
  archiveprefix = {arXiv},
  primaryclass  = {math.NA},
  opturl        = {https://arxiv.org/abs/2603.12993}
}

```
<a id="paper-unified"></a>

- **[3]** **🚀 NEW ✨** [*A unified variational framework for augmented Lagrangian preconditioning of saddle point systems: applications to mixed-dimensional and fictitious domain methods* (arXiV)](https://www.notyet.com)
```bibtex
@article{BenziUnified,
      title = {A unified variational framework for augmented Lagrangian preconditioning of saddle point systems: applications to mixed-dimensional and fictitious domain methods},
      author={Michele Benzi and Marco Feder and Luca Heltai and Federica Mugnaioni},
      year={2026},
      eprint={-},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/-}, 
}
```


## Authors and Contact

This repository is developed and maintained by:
- [Marco Feder](https://www.math.sissa.it/users/marco-feder) ([@fdrmrc](https://github.com/fdrmrc)), Numerical Analysis Group, Pisa - Università di Pisa, IT
- [Federica Mugnaioni](https://numpi.dm.unipi.it/people/federica-mugnaioni/) ([@federica-mugnaioni](https://github.com/federica-mugnaioni)), Numerical Analysis Group, Pisa - Scuola Normale Superiore, Pisa, IT

For special requests, you can either contact the authors by email or open an issue.


## License

The code in this repository is licensed under the
[GNU General Public License, version 3 only](LICENSE) (`GPL-3.0-only`).
