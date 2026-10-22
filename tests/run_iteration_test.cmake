# Run one example program and check its outer-solver iteration counts.
#
# This is the ctest driver behind `al_add_iteration_test()` (see
# tests/CMakeLists.txt). It mirrors deal.II's `check_solver_within_range()`
# macro (tests/tests.h in the deal.II sources): for every refinement cycle the
# number of outer FGMRES iterations is checked against that cycle's inclusive
# range [MIN, MAX], and the result is written as
#
#   DEAL::Solver stopped within MIN - MAX iterations
#
# or, if the count falls outside the range,
#
#   DEAL::Solver stopped after N iterations
#
# The resulting `output` file is then compared with the golden file. Checking a
# range rather than an exact count keeps the tests robust against small
# platform-, compiler- or partition-dependent fluctuations.
#
# The iteration counts are read from the line that deal.II's SolverControl
# logs when the outer solver converges, `DEAL:FGMRES::Convergence step N ...`.
# Nested inner solves log with a longer prefix (e.g. `DEAL:FGMRES:cg::...`) and
# are therefore not picked up.
#
# Required variables (passed with -D):
#   TEST_NAME, EXECUTABLE, PARAMETER_FILE, GOLDEN_FILE, WORK_DIR,
#   ITERATION_RANGES  comma-separated `MIN-MAX` ranges, one per refinement
#                     cycle (e.g. "4-8,5-9,4-8")
# Optional variables, used to run under MPI when MPI_PROCS > 1:
#   MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, MPIEXEC_POSTFLAGS, MPI_PROCS

foreach(_var TEST_NAME EXECUTABLE PARAMETER_FILE GOLDEN_FILE WORK_DIR
             ITERATION_RANGES)
  if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "run_iteration_test.cmake: ${_var} is not set.")
  endif()
endforeach()

# Commas are used on the command line because ctest would split a ';' list
# into separate arguments.
string(REPLACE "," ";" _ranges "${ITERATION_RANGES}")
foreach(_range IN LISTS _ranges)
  if(NOT _range MATCHES "^[0-9]+-[0-9]+$")
    message(FATAL_ERROR
      "${TEST_NAME}: invalid iteration range '${_range}', expected MIN-MAX.")
  endif()
endforeach()
list(LENGTH _ranges N_CYCLES)

# The program runs inside WORK_DIR, so make paths given relative to the
# caller's directory absolute before switching directory.
foreach(_var EXECUTABLE PARAMETER_FILE)
  if(EXISTS "${${_var}}")
    get_filename_component(${_var} "${${_var}}" ABSOLUTE)
  endif()
endforeach()

# Run the program in a fresh directory, so that the files it writes (solution
# output, used parameters, ...) never leak between tests or runs.
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(_command "${EXECUTABLE}" "${PARAMETER_FILE}")
if(DEFINED MPI_PROCS AND MPI_PROCS GREATER 1)
  if("${MPIEXEC}" STREQUAL "")
    message(FATAL_ERROR "${TEST_NAME}: MPI_PROCS=${MPI_PROCS} but MPIEXEC is not set.")
  endif()
  set(_command "${MPIEXEC}" ${MPIEXEC_NUMPROC_FLAG} ${MPI_PROCS}
               ${MPIEXEC_PREFLAGS} ${_command} ${MPIEXEC_POSTFLAGS})
endif()

# Naming the same variable for both pipes merges stdout and stderr.
execute_process(COMMAND ${_command}
  WORKING_DIRECTORY "${WORK_DIR}"
  OUTPUT_VARIABLE _log
  ERROR_VARIABLE _log
  RESULT_VARIABLE _exit_code)
file(WRITE "${WORK_DIR}/stdout.log" "${_log}")

if(NOT "${_exit_code}" STREQUAL "0")
  string(LENGTH "${_log}" _length)
  set(_tail_length 4000)
  if(_length GREATER _tail_length)
    math(EXPR _start "${_length} - ${_tail_length}")
    string(SUBSTRING "${_log}" ${_start} ${_tail_length} _log)
  endif()
  message(FATAL_ERROR
    "${TEST_NAME}: the program exited with status '${_exit_code}'.\n"
    "Full log: ${WORK_DIR}/stdout.log\n"
    "Last lines of the log:\n${_log}")
endif()

string(REGEX MATCHALL "DEAL:FGMRES::Convergence step [0-9]+" _matches "${_log}")
list(LENGTH _matches _n_solves)
if(NOT _n_solves EQUAL N_CYCLES)
  message(FATAL_ERROR
    "${TEST_NAME}: expected ${N_CYCLES} converged outer solves (one per "
    "refinement cycle), found ${_n_solves}. Make sure the outer solver control "
    "has 'Log result = true'. Full log: ${WORK_DIR}/stdout.log")
endif()

set(_output "")
set(_observed "")
set(_cycle 0)
foreach(_match IN LISTS _matches)
  string(REGEX REPLACE ".* step ([0-9]+)$" "\\1" _steps "${_match}")
  list(APPEND _observed ${_steps})
  list(GET _ranges ${_cycle} _range)
  string(REGEX REPLACE "^([0-9]+)-([0-9]+)$" "\\1" _min "${_range}")
  string(REGEX REPLACE "^([0-9]+)-([0-9]+)$" "\\2" _max "${_range}")
  string(APPEND _output "DEAL::Refinement cycle ${_cycle}\n")
  if(_steps GREATER_EQUAL _min AND _steps LESS_EQUAL _max)
    string(APPEND _output
      "DEAL::Solver stopped within ${_min} - ${_max} iterations\n")
  else()
    string(APPEND _output "DEAL::Solver stopped after ${_steps} iterations\n")
  endif()
  math(EXPR _cycle "${_cycle} + 1")
endforeach()
file(WRITE "${WORK_DIR}/output" "${_output}")

# Compare ignoring line-ending style and leading/trailing whitespace, so that
# the golden files are not sensitive to how an editor saves them.
file(READ "${GOLDEN_FILE}" _expected)
foreach(_text _expected _output)
  string(REPLACE "\r\n" "\n" ${_text} "${${_text}}")
  string(STRIP "${${_text}}" ${_text})
endforeach()

if(NOT _output STREQUAL _expected)
  message(FATAL_ERROR
    "${TEST_NAME}: output differs from the golden file.\n"
    "Observed outer iterations per cycle: ${_observed} "
    "(allowed ranges per cycle: ${ITERATION_RANGES})\n"
    "--- expected (${GOLDEN_FILE}):\n${_expected}\n"
    "--- actual (${WORK_DIR}/output):\n${_output}\n"
    "Full log: ${WORK_DIR}/stdout.log")
endif()

message(STATUS
  "${TEST_NAME}: outer iterations per cycle = ${_observed} "
  "(allowed ranges per cycle: ${ITERATION_RANGES})")
