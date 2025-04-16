# Augmented Lagrangian-based preconditioners for Fictitious Domain solvers

This repository contains application codes demonstrating *Augmented-Lagrangian*-based preconditioners for Fictitious Domain-type solvers. They
are based on the [deal.II library](https://www.dealii.org). All problems are meant to run both in 2D and 3D.

The associated preprint can be found online on [arXiv](https://arxiv.org/abs/2504.11339).



## Prerequisites

We require:
- **cmake** version greater than 2.8.
- One of the following compilers:
  -  **gcc** version  >= 11.4.0
  -  **clang** version >= 15
- **openMPI** version  >= 4.0.3
- **Trilinos** version >= 14.4.0
- **deal.II** version 9.7 (current master branch)
- **UMFPACK** (usually already bundled with deal.II)


## Compiling and running the examples 
Assuming deal.II is installed on your machine and meets the requirements above, all is required to do is:

```bash
git clone git@github.com:fdrmrc/fictitious_domain_AL_preconditioners.git
cd fictitious_domain_AL_preconditioners/
mkdir build
cd build/
cmake -DDEAL_II_DIR=/path/to/deal.II ..
make -j<N>
```
being ```N``` is the number of jobs you want to use to compile. If successfull, this generates the following example applications, which are
employing Fictitious Domain methodologies:

* **immersed_laplace**  
  Solves the Laplace equation with an internal constraint imposed on an immersed domain of co-dimension one. Parameter files can be found
  in the `prms/` folder.

* **stokes_immersed_boundary**  
Stokes problem with an immersed body of co-dimension one.

* **elliptic_interface**  
  Elliptic interface problem with jump in coefficients. The solid, immersed, domain is assumeted to be of co-dimension 0.









## Authors and Contact

This repository is developed and maintained by:
- [Marco Feder](https://www.math.sissa.it/users/marco-feder) ([@fdrmrc](https://github.com/fdrmrc)), Numerical Analysis Group, Pisa - Università di Pisa, IT
- [Federica Mugnaioni](https://numpi.dm.unipi.it/people/federica-mugnaioni/) ([@federica-mugnaioni](https://github.com/federica-mugnaioni)), Numerical Analysis Group, Pisa - Scuola Normale Superiore, Pisa, IT

For inquiries or special requests, you can either contact the authors by email or open an issue.

