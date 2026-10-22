cmake_minimum_required(VERSION 3.13.4)

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

# Picks up the compiler and flags deal.II was built with (e.g. the MPI
# compiler wrapper), so it must precede project().
deal_ii_initialize_cached_variables()

# Set the name of the project
project(AL_preconditioning_fictitious_domain)

# Example programs: one executable per source file in examples/. The
# executables are placed at the top of the build directory.
set(AL_EXAMPLES
  babuska_problem
  elliptic_interface
  elliptic_interface_elasticity
  immersed_laplace
  stokes_immersed_boundary
  variational_elliptic_interface
  variational_immersed_elasticity
)

foreach(_example ${AL_EXAMPLES})
  add_executable(${_example} examples/${_example}.cc)
  target_include_directories(${_example} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  # Link with deal.II
  deal_ii_setup_target(${_example})
endforeach()

# Grids directory
set(AL_GRIDS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/grids")
foreach(_target immersed_laplace babuska_problem)
  target_compile_definitions(${_target} PRIVATE AL_GRIDS_DIR="${AL_GRIDS_DIR}")
endforeach()

# Regression tests, run with `ctest` from the build directory.
enable_testing()
add_subdirectory(tests)

message(STATUS "
****************************************************************************************************
*                                                                                                    *
*  Welcome to the 'Augmented Lagrangian preconditioners for fictitious domain discretizations' suite!*
*  A collection of preconditioners for fictitious-domain discretizations.                            *
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
