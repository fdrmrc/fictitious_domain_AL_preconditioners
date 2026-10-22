# Set the name of the project
cmake_minimum_required(VERSION 3.13.4)
project(AL_preconditioning_fictitious_domain)

# Find deal.II library
find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
endif()

# Check required dependencies
if(NOT(DEAL_II_WITH_TRILINOS) OR NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_UMFPACK OR NOT DEAL_II_WITH_MUPARSER)
  message(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_MUPARSER = ON
    DEAL_II_WITH_UMFPACK = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_MUPARSER = ${DEAL_II_WITH_MUPARSER}
    DEAL_II_WITH_UMFPACK = ${DEAL_II_WITH_UMFPACK}
This conflicts with the requirements."
  )
endif()

deal_ii_initialize_cached_variables()

# Define multiple executables
add_executable(immersed_laplace immersed_laplace.cc)
add_executable(stokes_immersed_boundary stokes_immersed_boundary.cc)
add_executable(elliptic_interface elliptic_interface.cc)
add_executable(elliptic_interface_elasticity elliptic_interface_elasticity.cc)
add_executable(babuska_problem babuska_problem.cc)
add_executable(variational_elliptic_interface variational_elliptic_interface.cc)
add_executable(variational_immersed_elasticity variational_immersed_elasticity.cc)

# Link each executable with deal.II
deal_ii_setup_target(immersed_laplace)
deal_ii_setup_target(stokes_immersed_boundary)
deal_ii_setup_target(elliptic_interface)
deal_ii_setup_target(elliptic_interface_elasticity)
deal_ii_setup_target(babuska_problem)
deal_ii_setup_target(variational_elliptic_interface)
deal_ii_setup_target(variational_immersed_elasticity)

message(STATUS "
****************************************************************************************************
*                                                                                                    *
*  Welcome to the 'Augmented Lagrangian Preconditioner for fictitious domain Discretizations' suite! *
*  A collection of examples for fictitious-domain discretizations.                                   *
*                                                                                                    *
*  Run the following commands from the repository root:                                              *
*                                                                                                    *
*  1. Build the examples:                                                                            *
*       cmake --build <build-dir> --parallel <jobs>                                                  *
*                                                                                                    *
*     Replace <build-dir> with your CMake build directory (e.g. build),                              *
*     and <jobs> with the number of parallel build jobs (e.g. 4).                                    *
*                                                                                                    *
*  2. Run an example, passing its parameter file as the first argument:                              *
*     This example assumes your build directory is build:                                            *
*       build/babuska_problem parameters_babuska/parameters_babuska_2D.prm                           *
*                                                                                                    *
*     For examples that support MPI parallelism, prefix the command with                             *
*     mpirun -np <processes>, choosing the number of MPI processes.                                  *
*                                                                                                    *
*  See README.md for prerequisites, examples, and parameter guidance.                                *
*  For questions, you can email marco.feder@sissa.it.                                                *
*                                                                                                    *
****************************************************************************************************
")
