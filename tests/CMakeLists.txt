# Regression tests for the example programs.
#
# Each test runs one of the executables on a small problem (a reduced copy of
# one of the shipped parameter files, see the *.prm files in this directory)
# and checks that the outer FGMRES solver converges within an inclusive range
# of iterations at every refinement cycle, in the spirit of deal.II's
# `check_solver_within_range()`. The result is compared with the golden
# `<name>.output` file. See run_iteration_test.cmake for the details.
#
# Build the examples first, then run the suite from the build directory with
#   ctest --output-on-failure

set(AL_TEST_DRIVER "${CMAKE_CURRENT_SOURCE_DIR}/run_iteration_test.cmake")

# al_add_iteration_test(NAME <name> TARGET <target> PRM <file> OUTPUT <file>
#                       RANGES <min>-<max> [<min>-<max> ...] [MPI_PROCS <n>])
#
# RANGES gives, for every refinement cycle performed with PRM, the inclusive
# range of allowed outer iterations (one entry per cycle, in order; this is the
# equivalent of calling `check_solver_within_range()` once per solve). With
# MPI_PROCS > 1 the program is started through the MPI launcher deal.II was
# configured with.
function(al_add_iteration_test)
  cmake_parse_arguments(_t "" "NAME;TARGET;PRM;OUTPUT;MPI_PROCS" "RANGES"
    ${ARGN})

  foreach(_arg NAME TARGET PRM OUTPUT RANGES)
    if("${_t_${_arg}}" STREQUAL "")
      message(FATAL_ERROR "al_add_iteration_test: missing ${_arg}.")
    endif()
  endforeach()
  string(REPLACE ";" "," _ranges "${_t_RANGES}")

  set(_mpi_args "")
  if(_t_MPI_PROCS AND _t_MPI_PROCS GREATER 1)
    if("${DEAL_II_MPIEXEC}" STREQUAL "")
      message(STATUS "Skipping test ${_t_NAME}: no MPI launcher available.")
      return()
    endif()
    set(_mpi_args
      "-DMPIEXEC=${DEAL_II_MPIEXEC}"
      "-DMPIEXEC_NUMPROC_FLAG=${DEAL_II_MPIEXEC_NUMPROC_FLAG}"
      "-DMPIEXEC_PREFLAGS=${DEAL_II_MPIEXEC_PREFLAGS}"
      "-DMPIEXEC_POSTFLAGS=${DEAL_II_MPIEXEC_POSTFLAGS}"
      "-DMPI_PROCS=${_t_MPI_PROCS}")
  endif()

  add_test(NAME ${_t_NAME}
    COMMAND ${CMAKE_COMMAND}
      "-DTEST_NAME=${_t_NAME}"
      "-DEXECUTABLE=$<TARGET_FILE:${_t_TARGET}>"
      "-DPARAMETER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${_t_PRM}"
      "-DGOLDEN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${_t_OUTPUT}"
      "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/${_t_NAME}"
      "-DITERATION_RANGES=${_ranges}"
      ${_mpi_args}
      -P "${AL_TEST_DRIVER}")

  set_tests_properties(${_t_NAME} PROPERTIES TIMEOUT 1800 LABELS "examples")
  if(_t_MPI_PROCS AND _t_MPI_PROCS GREATER 1)
    set_tests_properties(${_t_NAME} PROPERTIES PROCESSORS ${_t_MPI_PROCS})
  endif()
endfunction()

# The ranges below bracket the iteration counts observed on the reference
# configuration (c +/- max(2, ceil(0.15 c)) for an observed count c).

al_add_iteration_test(NAME babuska_2d
  TARGET babuska_problem
  PRM babuska_2d.prm OUTPUT babuska_2d.output
  RANGES 4-8 5-9 4-8)

al_add_iteration_test(NAME elliptic_interface_2d
  TARGET elliptic_interface
  PRM elliptic_interface_2d.prm OUTPUT elliptic_interface_2d.output
  RANGES 3-7 8-12 13-19)

al_add_iteration_test(NAME variational_elliptic_interface_2d
  TARGET variational_elliptic_interface
  PRM variational_elliptic_interface_2d.prm
  OUTPUT variational_elliptic_interface_2d.output
  RANGES 3-7 15-21 19-27)

al_add_iteration_test(NAME variational_elliptic_interface_2d_mpi2
  TARGET variational_elliptic_interface
  PRM variational_elliptic_interface_2d.prm
  OUTPUT variational_elliptic_interface_2d.output
  RANGES 3-7 15-21 19-27
  MPI_PROCS 2)
