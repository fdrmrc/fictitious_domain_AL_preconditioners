#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_tags.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <mpi.h>

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// We include the header file where AL preconditioners are implemented
#include "augmented_lagrangian_preconditioner.h"
#include "utilities.h"

using namespace dealii;

// Analytical solution taken from https://arxiv.org/pdf/2211.03443
template <int dim>
class Solution : public Function<dim> {
 public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(
      const Point<dim> &p, const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
  double return_value;

  auto r = p.norm();
  const double coefficient_omega = 1.;
  const double coefficient_omega2 = 10.;
  if (r <= 1)  // inside domain omega2
  {
    return_value = (3. * coefficient_omega2 / coefficient_omega + 1. - r * r) /
                   (2. * dim * coefficient_omega2);
  } else {
    return_value = (4. - r * r) / (2. * dim * coefficient_omega);
  }

  return return_value;
}

template <int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                       const unsigned int) const {
  const double coefficient_omega = 1.;
  const double coefficient_omega2 = 10.;
  auto r = p.norm();
  Tensor<1, dim> gradient;
  if (r <= 1) {
    for (int i = 0; i < dim; ++i) {
      gradient[i] = -p(i) / (dim * coefficient_omega2);
    }
  } else {
    for (int i = 0; i < dim; ++i) {
      gradient[i] = -p(i) / (dim * coefficient_omega);
    }
  }
  return gradient;
}

// define boundary values to be the exact solution itself
template <int dim>
class BoundaryValues : public Function<dim> {
 public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const {
  return (4. - p[0] * p[0] - p[1] * p[1]) / 4.;
}

template <int dim>
class ProblemParameters : public ParameterAcceptor {
 public:
  ProblemParameters();

  std::string output_directory = ".";

  // Number of refinement steps for background...
  unsigned int initial_background_refinement = 4;
  // ... and immersed grid.
  unsigned int initial_immersed_refinement = 2;

  std::string name_of_background_grid = "hyper_cube";
  std::string arguments_for_background_grid = "-1: 1: true";
  std::string name_of_immersed_grid = "hyper_cube";
  std::string arguments_for_immersed_grid = "-0.14: 0.44: true";

  // Number of refinement cycles
  unsigned int n_refinement_cycles = 5;

  // Coefficients determining the size of the jump. beta_1 is usually kept at 1.
  mutable double beta_1 = 1.;

  // beta_2, instead, is the one which is changed.
  mutable double beta_2 = 10.;

  std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

  unsigned int background_space_finite_element_degree = 1;

  unsigned int immersed_space_finite_element_degree = 1;

  unsigned int coupling_quadrature_order = 3;

  unsigned int verbosity_level = 10;

  bool use_modified_AL_preconditioner = true;

  bool do_parameter_study = false;
  // Define range of values for gamma, in case we want to determine gamma
  // experimentally.
  double start_gamma = 1e-3;

  double end_gamma = 1.;

  unsigned int n_steps_gamma = 100;

  std::string mass_solver = "direct";

  std::string direct_solver_type = "Amesos_Klu";

  bool use_sqrt_2_rule = false;

  bool do_sanity_checks = true;

  bool do_convergence_study = false;

  bool export_matrices_for_eig_analysis = false;

  // AL parameter. Its magnitude depends on which AL preconditioner (original
  // vs. modified AL) is chosen. We define it as mutable since with modified AL
  // its value may change upon mesh refinement.
  mutable double gamma_AL_background = 10.;
  mutable double gamma_AL_immersed = 10.;

  mutable ParameterAcceptorProxy<ReductionControl> outer_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> inner_solver_control;
  mutable ParameterAcceptorProxy<IterationNumberControl>
      iteration_number_control;

  // We define f_1 and f_2 - f right-hand sides
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> f_1;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> f_2_minus_f;

  // If true, we use a fixed number of iterations inside inner solver (only for
  // modified AL)
  bool use_fixed_iterations = true;
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
    : ParameterAcceptor("Elliptic Interface Problem/"),
      outer_solver_control("Outer solver control"),
      inner_solver_control("Inner solver control"),
      iteration_number_control("Iteration number control"),
      f_1("Right hand side f_1"),
      f_2_minus_f("Right hand side f_2 - f") {
  add_parameter("FE degree background", background_space_finite_element_degree,
                "", this->prm, Patterns::Integer(1));

  add_parameter("FE degree immersed", immersed_space_finite_element_degree, "",
                this->prm, Patterns::Integer(1));

  add_parameter("Coupling quadrature order", coupling_quadrature_order);

  add_parameter("Output directory", output_directory);

  add_parameter("Beta_1", beta_1);
  add_parameter("Beta_2", beta_2);

  add_parameter("Homogeneous Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Perform sanity checks", do_sanity_checks,
                "Check condition number of C*Ct after solve().");
  add_parameter("Perform convergence study", do_convergence_study,
                "Using analytical solution for immersed circle (f = f_2 = 1)");
  add_parameter("Export matrices for eigs-analysis",
                export_matrices_for_eig_analysis);
  add_parameter("Use fixed (inner) iterations", use_fixed_iterations,
                "Perform fixed number of iterations within inner solvers. ");

  enter_subsection("Grid generation");
  {
    add_parameter("Background grid generator", name_of_background_grid);
    add_parameter("Background grid generator arguments",
                  arguments_for_background_grid);
    add_parameter("Immersed grid generator", name_of_immersed_grid);
    add_parameter("Immersed grid generator arguments",
                  arguments_for_immersed_grid);
  }
  leave_subsection();

  enter_subsection("Refinement and remeshing");
  {
    add_parameter(
        "Initial background refinement", initial_background_refinement,
        "Initial number of refinements used for the background domain Omega.");
    add_parameter(
        "Initial immersed refinement", initial_immersed_refinement,
        "Initial number of refinements used for the immersed domain Gamma.");
    add_parameter(
        "Refinemented cycles", n_refinement_cycles,
        "Number of refinement cycles to perform convergence studies.");
  }
  leave_subsection();

  enter_subsection("AL preconditioner");
  {
    add_parameter("Use modified AL preconditioner",
                  use_modified_AL_preconditioner,
                  "Use the modified AL preconditioner. If false, the classical "
                  "AL preconditioner is used.");
    add_parameter("Do parameter study", do_parameter_study,
                  "Perform a parameter study on the AL parameter gamma on a "
                  "coarse mesh to select experimentally an optimal value.");
    add_parameter("Use sqrt(2)-rule for gamma", use_sqrt_2_rule,
                  "Use sqrt(2)-rule for gamma. It makes sense only for "
                  "modified AL variant.");
    add_parameter("gamma fluid", gamma_AL_background);
    add_parameter("gamma solid", gamma_AL_immersed);
    add_parameter("Verbosity level", verbosity_level);
  }
  leave_subsection();

  enter_subsection("Mass solver");
  {
    add_parameter("Solver for mass matrix", mass_solver,
                  "Type of solver used to solve for the mass matrix.",
                  ParameterAcceptor::prm,
                  Patterns::Selection("direct|iterative|diagonal"));
    add_parameter("Direct solver type", direct_solver_type,
                  "The type of **direct** solver you want to use (if you "
                  "selected direct above).",
                  ParameterAcceptor::prm,
                  Patterns::Selection(
                      "Amesos_Lapack|Amesos_Scalapack|Amesos_Klu|Amesos_"
                      "Umfpack|Amesos_Pardiso|Amesos_Taucs|Amesos_Superlu|"
                      "Amesos_Superludist|Amesos_Dscpack|Amesos_Mumps"));
  }
  leave_subsection();

  enter_subsection("Parameter study");
  {
    add_parameter(
        "Start gamma", start_gamma,
        "Starting value for the range of values of gamma we want to test.");
    add_parameter(
        "Stop gamma", end_gamma,
        "Last value for the range of values of gamma we want to test.");
    add_parameter("Number of steps", n_steps_gamma,
                  "Number of steps from start to stop. (Similar to linspace in "
                  "Python or MatLab).");
  }
  leave_subsection();

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Reduction", "1.e-20");
    ParameterAcceptor::prm.set("Tolerance", "1.e-6");
    ParameterAcceptor::prm.set("Log history", "true");
    ParameterAcceptor::prm.set("Log result", "true");
  });

  // Same, but for inner solver
  inner_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Reduction", "1.e-20");
    ParameterAcceptor::prm.set("Tolerance", "1.e-4");
    ParameterAcceptor::prm.set("Log history", "false");
    ParameterAcceptor::prm.set("Log result", "true");
  });

  // Same, but for the case when we want to use fixed number of iterations
  iteration_number_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "100");
    ParameterAcceptor::prm.set("Tolerance", "1.e-4");
    ParameterAcceptor::prm.set("Log history", "false");
    ParameterAcceptor::prm.set("Log result", "true");
  });

  // Right hand sides
  f_1.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });
  f_2_minus_f.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });
}

// The real class of the ellitpic interface problem.
template <int dim>
class EllipticInterfaceDLM {
 public:
  typedef double Number;

  EllipticInterfaceDLM(const ProblemParameters<dim> &prm, const MPI_Comm comm);

  void generate_grids();

  void system_setup();

  void setup_stiffness_matrix(
      const DoFHandler<dim> &dof_handler,
      AffineConstraints<double> &constraints,
      const IndexSet &locally_owned_dofs, const IndexSet &locally_relevant_dofs,
      TrilinosWrappers::SparseMatrix &stiffness_matrix) const;

  void setup_coupling();

  void assemble_subsystem(const FiniteElement<dim> &fe,
                          const DoFHandler<dim> &dof_handler,
                          const AffineConstraints<double> &constraints,
                          TrilinosWrappers::SparseMatrix &system_matrix,
                          TrilinosWrappers::MPI::Vector &system_rhs, double rho,
                          double mu, const Function<dim> &rhs_function) const;

  void assemble();

  // Returns the number of outer iterations.
  unsigned int solve();

  void output_results(const unsigned int ref_cycle) const;

  void run();

 private:
  const ProblemParameters<dim> &parameters;

  MappingQ1<dim> mapping;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;

  // locally owned dofs
  IndexSet locally_owned_dofs_bg;
  IndexSet locally_owned_dofs_fg;

  // locally relevant dofs
  IndexSet locally_relevant_dofs_bg;
  IndexSet locally_relevant_dofs_fg;

  TrilinosWrappers::SparseMatrix stiffness_matrix_bg;
  TrilinosWrappers::SparseMatrix stiffness_matrix_fg;
  TrilinosWrappers::SparseMatrix
      stiffness_matrix_fg_plus_id;  // A_2 + gammaId, needed for modifiedAL
  TrilinosWrappers::SparseMatrix coupling_matrix;
  TrilinosWrappers::SparseMatrix coupling_matrix_transpose;

  TrilinosWrappers::SparseMatrix mass_matrix_fg;

  TrilinosWrappers::MPI::BlockVector system_rhs_block;
  TrilinosWrappers::MPI::BlockVector system_solution_block;

  MPI_Comm comm;

  parallel::distributed::Triangulation<dim> tria_bg;  // fluid, distributed
  parallel::shared::Triangulation<dim> tria_fg;       // solid, shared

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;

  FE_Q<dim> fe_bg;
  FE_Q<dim> fe_fg;

  mutable ConvergenceTable convergence_table;

  std::unique_ptr<SolverCG<TrilinosWrappers::MPI::Vector>>
      solver_lagrangian_scalar;

  ConditionalOStream pcout;

  mutable TimerOutput computing_timer;

  TrilinosWrappers::PreconditionILU M_inv_ilu;

  std::unique_ptr<GridTools::Cache<dim>> space_grid_tools_cache;
};

template <int dim>
EllipticInterfaceDLM<dim>::EllipticInterfaceDLM(
    const ProblemParameters<dim> &prm, const MPI_Comm communicator)
    : parameters(prm),
      comm(communicator),
      tria_bg(comm),
      tria_fg(comm),
      dof_handler_bg(tria_bg),
      dof_handler_fg(tria_fg),
      fe_bg(parameters.background_space_finite_element_degree),
      fe_fg(parameters.immersed_space_finite_element_degree),
      pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
      computing_timer(comm, pcout, TimerOutput::every_call_and_summary,
                      TimerOutput::wall_times) {
  // First, do some sanity checks on the parameters.
  AssertThrow(parameters.beta_1 > 0., ExcMessage("Beta_1 must be positive."));
  AssertThrow(parameters.beta_1 > 0., ExcMessage("Beta_2 must be positive."));
  AssertThrow(parameters.beta_2 > parameters.beta_1,
              ExcMessage("Beta_2 must be greater than Beta_1."));
  AssertThrow(parameters.gamma_AL_background > 0.,
              ExcMessage("Gamma must be positive."));
  AssertThrow(parameters.gamma_AL_immersed > 0.,
              ExcMessage("Gamma2 must be positive."));

  // Check that the AL parameters for the solid equation is smaller than the
  // fluid one
  AssertThrow(parameters.gamma_AL_immersed <= parameters.gamma_AL_background,
              ExcMessage("The AL parameter gamma2 for the solid should be "
                         "smaller than gamma for the fluid region. Aborting."));

  // Check that some settings are used only when modified AL preconditioner is
  // selected.
  if (parameters.do_parameter_study)
    AssertThrow(
        parameters.use_modified_AL_preconditioner,
        ExcMessage(
            "Parameter study for gamma makes sense only if you use modified "
            "AL preconditioner."));
  if (parameters.use_sqrt_2_rule)
    AssertThrow(parameters.use_modified_AL_preconditioner,
                ExcMessage("The so-called sqrt(2)-rule makes sense only if you "
                           "use the modified AL preconditioner."));

  if (parameters.do_convergence_study)
    AssertThrow(dim == 2,
                ExcNotImplemented());  // we check convergence rates only in 2D.

  // Check if the folder where we want to save solutions actually exists
  if (std::filesystem::exists(parameters.output_directory)) {
    Assert(std::filesystem::is_directory(parameters.output_directory),
           ExcMessage("You specified <" + parameters.output_directory +
                      "> as the output directory in the input file, "
                      "but this is not in fact a directory."));
  } else
    std::filesystem::create_directory(parameters.output_directory);
}

// Generate the grids for the background and immersed domains
template <int dim>
void EllipticInterfaceDLM<dim>::generate_grids() {
  TimerOutput::Scope t(computing_timer, "Grid generation");

  // The convergence test is constructed ad-hoc...
  if (parameters.do_convergence_study) {
    parameters.beta_1 = 1.;
    parameters.beta_2 = 10.;
    GridGenerator::hyper_cube(tria_bg, -1.4, 1.4, false);
    tria_bg.refine_global(parameters.initial_background_refinement);
    GridGenerator::hyper_ball(tria_fg, {0., 0.}, 1., true);
    tria_fg.refine_global(parameters.initial_immersed_refinement);

  } else {
    // ... otherwise, we let the user decide the grids from the parameters file.
    try {
      GridGenerator::generate_from_name_and_arguments(
          tria_bg, parameters.name_of_background_grid,
          parameters.arguments_for_background_grid);
      tria_bg.refine_global(parameters.initial_background_refinement);
    } catch (const std::exception &exc) {
      std::cerr << exc.what() << std::endl;
      std::cerr << "Error in background grid generation. Aborting."
                << std::endl;
    }

    try {
      GridGenerator::generate_from_name_and_arguments(
          tria_fg, parameters.name_of_immersed_grid,
          parameters.arguments_for_immersed_grid);
      tria_fg.refine_global(parameters.initial_immersed_refinement);
    } catch (const std::exception &exc) {
      std::cerr << exc.what() << std::endl;
      std::cerr << "Error in immersed grid generation. Aborting." << std::endl;
    }
  }

  // Now check mesh sizes. We verify they are not too different by using safety
  // factors.
  const double h_background = GridTools::maximal_cell_diameter(tria_bg);
  const double h_immersed = GridTools::maximal_cell_diameter(tria_fg);
  const double ratio = h_background / h_immersed;
  pcout << "h background = " << h_background << "\n"
        << "h immersed = " << h_immersed << "\n"
        << "grids ratio (background/immersed) = " << ratio << std::endl;

  AssertThrow(ratio < 2.2 && ratio > 0.4,
              ExcMessage("Check mesh sizes of the two grids."));

  // Do not dump grids to disk if they are too large
  if (tria_bg.n_active_cells() < 1e3) {
    std::ofstream bg_out("background_grid.vtk");
    std::ofstream fg_out("immersed_grid.vtk");
    GridOut grid_out;
    grid_out.write_vtk(tria_bg, bg_out);
    grid_out.write_vtk(tria_fg, fg_out);
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::system_setup() {
  TimerOutput::Scope t(computing_timer, "System setup");

  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);

  // Retrieve locally owned and relevant DoFs
  locally_owned_dofs_bg = dof_handler_bg.locally_owned_dofs();
  locally_relevant_dofs_bg =
      DoFTools::extract_locally_relevant_dofs(dof_handler_bg);

  locally_owned_dofs_fg = dof_handler_fg.locally_owned_dofs();
  locally_relevant_dofs_fg =
      DoFTools::extract_locally_relevant_dofs(dof_handler_fg);

  constraints_bg.clear();
  constraints_fg.clear();

  constraints_bg.reinit(locally_owned_dofs_bg, locally_relevant_dofs_bg);
  constraints_fg.reinit(locally_owned_dofs_fg, locally_relevant_dofs_fg);

  if (parameters.do_convergence_study)
    VectorTools::interpolate_boundary_values(
        dof_handler_bg, 0, BoundaryValues<dim>(), constraints_bg);
  else
    for (const types::boundary_id id : parameters.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
          dof_handler_bg, id, Functions::ZeroFunction<dim>(), constraints_bg);

  constraints_bg.close();
  constraints_fg.close();

  setup_stiffness_matrix(dof_handler_bg, constraints_bg, locally_owned_dofs_bg,
                         locally_relevant_dofs_bg, stiffness_matrix_bg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, locally_owned_dofs_fg,
                         locally_relevant_dofs_fg, stiffness_matrix_fg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, locally_owned_dofs_fg,
                         locally_relevant_dofs_fg, stiffness_matrix_fg_plus_id);

  mass_matrix_fg.reinit(stiffness_matrix_fg);  // copy parallel layout

  // locally owned DoF for each block: (u,u_2,lambda)
  const std::vector<IndexSet> partitionings{
      locally_owned_dofs_bg, locally_owned_dofs_fg, locally_owned_dofs_fg};
  system_rhs_block.reinit(partitionings, comm, false);
  system_solution_block.reinit(system_rhs_block);

  pcout << "N DoF background: " << dof_handler_bg.n_dofs() << std::endl;
  pcout << "N DoF immersed: " << dof_handler_fg.n_dofs() << std::endl;

  pcout << "==============================================================="
           "========================="
        << std::endl;
}

template <int dim>
void EllipticInterfaceDLM<dim>::setup_stiffness_matrix(
    const DoFHandler<dim> &dof_handler, AffineConstraints<double> &constraints,
    const IndexSet &locally_owned_dofs, const IndexSet &locally_relevant_dofs,
    TrilinosWrappers::SparseMatrix &stiffness_matrix) const {
  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, comm,
                                             locally_relevant_dofs);

  stiffness_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, comm);
}

template <int dim>
void EllipticInterfaceDLM<dim>::setup_coupling() {
  TimerOutput::Scope t(computing_timer, "Coupling setup");

  QGauss<dim> quad(fe_bg.degree + 1);

  {
    space_grid_tools_cache = std::make_unique<GridTools::Cache<dim>>(tria_bg);

    TrilinosWrappers::SparsityPattern dsp(dof_handler_bg.locally_owned_dofs(),
                                          dof_handler_fg.locally_owned_dofs(),
                                          comm);

    TrilinosWrappers::SparsityPattern dsp_t(dof_handler_fg.locally_owned_dofs(),
                                            dof_handler_bg.locally_owned_dofs(),
                                            comm);

    // Here, we use velocity_dh: we want to couple DoF for velocity with the
    // ones of the multiplier.
    UtilitiesAL::create_coupling_sparsity_patterns(
        *space_grid_tools_cache, dof_handler_bg, dof_handler_fg, quad, dsp_t,
        dsp, constraints_bg, ComponentMask(), ComponentMask(), mapping,
        AffineConstraints<double>());
    dsp.compress();
    dsp_t.compress();
    pcout << "Sparsity coupling: done" << std::endl;
    coupling_matrix.reinit(dsp);
    coupling_matrix_transpose.reinit(dsp_t);

    // Assemble C and Ct simultaneously
    UtilitiesAL::create_coupling_mass_matrices(
        *space_grid_tools_cache, dof_handler_bg, dof_handler_fg, quad,
        coupling_matrix_transpose, coupling_matrix, constraints_bg,
        ComponentMask(), ComponentMask(), mapping, AffineConstraints<double>());
    coupling_matrix.compress(VectorOperation::add);
    coupling_matrix_transpose.compress(VectorOperation::add);
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::assemble_subsystem(
    const FiniteElement<dim> &fe, const DoFHandler<dim> &dof_handler,
    const AffineConstraints<double> &constraints,
    TrilinosWrappers::SparseMatrix &system_matrix,
    TrilinosWrappers::MPI::Vector &system_rhs, double rho, double mu,
    const Function<dim> &rhs_function) const {
  const QGauss<dim> quad(fe.degree + 1);

  FEValues<dim> fe_values(fe, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> rhs_values(quad.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);
      rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);

      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
        for (const unsigned int i : fe_values.dof_indices()) {
          for (const unsigned int j : fe_values.dof_indices())
            cell_matrix(i, j) +=
                (rho * fe_values.shape_value(i, q_index) *  // u
                     fe_values.shape_value(j, q_index)      // v
                 + mu * fe_values.shape_grad(i, q_index) *  // grad u
                       fe_values.shape_grad(j, q_index)) *  // grad v
                fe_values.JxW(q_index);                     // dx

          cell_rhs(i) += (rhs_values[q_index] *                // f(x)
                          fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                          fe_values.JxW(q_index));             // dx
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void EllipticInterfaceDLM<dim>::assemble() {
  TimerOutput::Scope t(computing_timer, "Assemble matrices");

  // background stiffness matrix A
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, stiffness_matrix_bg,
                     system_rhs_block.block(0),
                     0.,                 // 0, no mass matrix
                     parameters.beta_1,  // only beta_1 (which is usually 1)
                     parameters.f_1);    // rhs value, f

  // immersed matrix A2 = (beta_2 - beta) (grad u,grad v)
  // The rhs changes if we want to do a convergence study.
  if (parameters.do_convergence_study)
    assemble_subsystem(
        fe_fg, dof_handler_fg, constraints_fg, stiffness_matrix_fg,
        system_rhs_block.block(1),
        0.,                                     // 0, no mass matrix
        parameters.beta_2 - parameters.beta_1,  // hardcoded before in this case
        Functions::ConstantFunction<dim>{
            0.});  // rhs value for convergence study
  else             // read rhs values (constants) from parameters file
    assemble_subsystem(
        fe_fg, dof_handler_fg, constraints_fg, stiffness_matrix_fg,
        system_rhs_block.block(1),
        0.,                                     // 0, no mass matrix
        parameters.beta_2 - parameters.beta_1,  // only jump beta_2 - beta
        parameters.f_2_minus_f);                // rhs value, f2-f

  //  mass matrix immersed
  assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg, mass_matrix_fg,
                     system_rhs_block.block(2), 1., 0.,
                     Functions::ConstantFunction<dim>{0.});
}

void output_double_number(double input, const std::string &text) {
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << text << input << std::endl;
}

template <int dim>
unsigned int EllipticInterfaceDLM<dim>::solve() {
  // Start with defining the outer iterations to an invalid value.
  unsigned int n_outer_iterations = numbers::invalid_unsigned_int;

  using PayloadType = dealii::TrilinosWrappers::internal::
      LinearOperatorImplementation::TrilinosPayload;

  auto A_omega1 = linear_operator<TrilinosWrappers::MPI::Vector,
                                  TrilinosWrappers::MPI::Vector, PayloadType>(
      stiffness_matrix_bg);
  auto A_omega2 = linear_operator<TrilinosWrappers::MPI::Vector,
                                  TrilinosWrappers::MPI::Vector, PayloadType>(
      stiffness_matrix_fg);
  auto M = linear_operator<TrilinosWrappers::MPI::Vector,
                           TrilinosWrappers::MPI::Vector, PayloadType>(
      mass_matrix_fg);
  auto minusM = -1. * M;
  auto Zero = M * 0.0;
  auto Ct = linear_operator<TrilinosWrappers::MPI::Vector,
                            TrilinosWrappers::MPI::Vector, PayloadType>(
      coupling_matrix);
  auto C = linear_operator<TrilinosWrappers::MPI::Vector,
                           TrilinosWrappers::MPI::Vector, PayloadType>(
      coupling_matrix_transpose);

  // We create empty operators for the action of the inverse of W. Depending
  // on the the choice, we either approximate it using the diagonal of the
  // mass matrix (squaring the entries) or we use the direct inversion of the
  // mass matrix.
  auto invM = null_operator(M);
  auto invW = null_operator(M);
  SolverControl solver_control(100, 1e-12, false, false);
  SolverCG<TrilinosWrappers::MPI::Vector> cg_solver(solver_control);

  TrilinosWrappers::MPI::Vector inverse_diag_mass_squared(locally_owned_dofs_fg,
                                                          comm);

  for (const types::global_dof_index i : locally_owned_dofs_fg)
    inverse_diag_mass_squared(i) =
        1. / (mass_matrix_fg.diag_element(i) * mass_matrix_fg.diag_element(i));
  inverse_diag_mass_squared.compress(VectorOperation::insert);

  TrilinosWrappers::SparseMatrix diag_inverse;
  TrilinosWrappers::SolverDirect::AdditionalData data;
  data.solver_type = parameters.direct_solver_type;
  // dummy solvercontrol to suppress output of direct solver
  SolverControl direct_control{1, 1e-12, false, false};
  TrilinosWrappers::SolverDirect solver_direct_mass(direct_control, data);

  if (std::strcmp(parameters.mass_solver.c_str(), "diagonal") == 0) {
    TrilinosWrappers::SparsityPattern diag_inverse_sparsity;
    diag_inverse_sparsity.reinit(locally_owned_dofs_fg, comm);

    for (const types::global_dof_index dof : locally_owned_dofs_fg)
      diag_inverse_sparsity.add(dof,
                                dof);  // Add diagonal entry only
    diag_inverse_sparsity.compress();

    diag_inverse.reinit(diag_inverse_sparsity);
    for (const types::global_dof_index dof : locally_owned_dofs_fg)
      diag_inverse.set(dof, dof, inverse_diag_mass_squared(dof));
    diag_inverse.compress(VectorOperation::insert);

    invW = linear_operator<TrilinosWrappers::MPI::Vector,
                           TrilinosWrappers::MPI::Vector, PayloadType>(
        diag_inverse);
  } else if (std::strcmp(parameters.mass_solver.c_str(), "direct") == 0) {
    // Use direct solver from Trilinos
    {
      TimerOutput::Scope t(computing_timer, "Factorize mass matrix");
      solver_direct_mass.initialize(mass_matrix_fg, data);
    }

    invM = linear_operator<TrilinosWrappers::MPI::Vector,
                           TrilinosWrappers::MPI::Vector, PayloadType>(
        M, solver_direct_mass);
    invW = invM * invM;
  } else if (std::strcmp(parameters.mass_solver.c_str(), "iterative") == 0) {
    // Invert M through a iterative solver + ILU. TODO: CG + mass lumping
    {
      TimerOutput::Scope t(computing_timer, "Factorize mass matrix");
      M_inv_ilu.initialize(mass_matrix_fg);
    }
    invM = inverse_operator(M, cg_solver, M_inv_ilu);
    invW = invM * invM;
  } else {
    Assert(false, ExcNotImplemented("Choose one type of mass solver between "
                                    "diagonal, direct, and iterative."));
  }

  // Define augmented blocks. Notice that A22_aug is actually A_omega2 +
  // gamma_AL_immersed * Id
  stiffness_matrix_fg_plus_id.copy_from(stiffness_matrix_fg);
  for (const types::global_dof_index idx : locally_owned_dofs_fg)
    stiffness_matrix_fg_plus_id.add(idx, idx, parameters.gamma_AL_immersed);
  // auto A22_aug = A_omega2 + parameters.gamma_AL_immersed *
  // identity_operator(M);
  auto A11_aug = A_omega1 + parameters.gamma_AL_background * Ct * invW * C;
  auto A22_aug = linear_operator<TrilinosWrappers::MPI::Vector,
                                 TrilinosWrappers::MPI::Vector, PayloadType>(
      stiffness_matrix_fg_plus_id);
  auto A12_aug = -parameters.gamma_AL_background * Ct * invM;
  auto A21_aug = -parameters.gamma_AL_immersed * invM * C;

  // Augmented (equivalent) system to be solved
  auto system_operator =
      block_operator<3, 3, TrilinosWrappers::MPI::BlockVector>(
          {{{{A11_aug, A12_aug, Ct}},
            {{A21_aug, A22_aug, minusM}},
            {{C, minusM, Zero}}}});  // augmented the 2x2 top left block!

  // Initialize AMG preconditioners for inner solves
  TrilinosWrappers::SparseMatrix augmented_matrix;
  UtilitiesAL::create_augmented_block_in_parallel(
      stiffness_matrix_bg, coupling_matrix_transpose, coupling_matrix,
      inverse_diag_mass_squared, parameters.gamma_AL_background,
      augmented_matrix);

  TrilinosWrappers::PreconditionAMG amg_prec_A11;
  amg_prec_A11.initialize(augmented_matrix);
  pcout << "Initialized AMG for A_1" << std::endl;

  TrilinosWrappers::PreconditionAMG amg_prec_A22;
  amg_prec_A22.initialize(stiffness_matrix_fg_plus_id);
  pcout << "Initialized AMG for A_2" << std::endl;

  if (parameters.export_matrices_for_eig_analysis) {
    AssertThrow(
        false,
        ExcNotImplemented(
            "Not implemented in parallel. Use the serial demo version."));
  }

  typename SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData
      data_fgmres;
  data_fgmres.max_basis_size = 50;
  SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver_fgmres(
      parameters.outer_solver_control, data_fgmres);

  // Wrap the actual FGMRES solve in another scope in order to discard
  // setup-cost.
  {
    TimerOutput::Scope t(computing_timer, "Solve system");
    if (parameters.use_modified_AL_preconditioner) {
      // If we use the modified AL preconditioner, we check if we use a small
      // value of gamma.
      AssertThrow(
          parameters.gamma_AL_immersed <= 20.,
          ExcMessage(
              "gamma_AL_immersed is too large for modified ALpreconditioner."));

      // We wither use a fixed number of iterations inside the inner solver...
      if (parameters.use_fixed_iterations)
        solver_lagrangian_scalar =
            std::make_unique<SolverCG<TrilinosWrappers::MPI::Vector>>(
                parameters.iteration_number_control);
      else  //... or we iterate as usual.
        solver_lagrangian_scalar =
            std::make_unique<SolverCG<TrilinosWrappers::MPI::Vector>>(
                parameters.inner_solver_control);

      // Define block preconditioner using AL approach
      auto A11_aug_inv =
          inverse_operator(A11_aug, *solver_lagrangian_scalar, amg_prec_A11);
      auto A22_aug_inv =
          inverse_operator(A22_aug, *solver_lagrangian_scalar, amg_prec_A22);

      EllipticInterfacePreconditioners::BlockTriangularALPreconditionerModified
          preconditioner_AL(C, M, invM, invW, parameters.gamma_AL_background,
                            A11_aug_inv, A22_aug_inv);

      system_rhs_block.block(2) = 0;  // last row of the rhs is 0

      solver_fgmres.solve(system_operator, system_solution_block,
                          system_rhs_block, preconditioner_AL);

      pcout << "Norm of solution: " << system_solution_block.block(0).l2_norm()
            << std::endl;
    } else {
      AssertThrow(false,
                  ExcNotImplemented("Ideal case not implemented in parallel."));
    }
    // Finally, distribute the constraints
    constraints_bg.distribute(system_solution_block.block(0));
    constraints_fg.distribute(system_solution_block.block(1));
    constraints_fg.distribute(system_solution_block.block(2));
  }

  n_outer_iterations = parameters.outer_solver_control.last_step();

  // We update the table with iteration counts
  convergence_table.add_value("cells", tria_bg.n_active_cells());
  convergence_table.add_value("DoF background", dof_handler_bg.n_dofs());
  convergence_table.add_value("DoF immersed", dof_handler_fg.n_dofs());
  convergence_table.add_value("gamma (AL)", parameters.gamma_AL_background);
  if (parameters.use_modified_AL_preconditioner)
    convergence_table.add_value("gamma2 (AL)", parameters.gamma_AL_immersed);
  convergence_table.add_value("Outer iterations", n_outer_iterations);

  pcout << "Solved in " << n_outer_iterations << " iterations"
        << (parameters.outer_solver_control.last_step() < 10 ? "  " : " ")
        << "\n";

  // Do some sanity checks: Check the constraints residual and the condition
  // number of CCt.
  if (parameters.do_sanity_checks) {
    TrilinosWrappers::MPI::Vector difference_constraints =
        system_solution_block.block(2);

    coupling_matrix.Tvmult(difference_constraints,
                           system_solution_block.block(0));
    difference_constraints *= -1;

    mass_matrix_fg.vmult_add(difference_constraints,
                             system_solution_block.block(1));

    pcout << "L infty norm of constraints residual "
          << difference_constraints.linfty_norm() << "\n";

    // Estimate condition number
    pcout << "Estimate condition number of CCt using CG" << std::endl;
    TrilinosWrappers::MPI::Vector v_eig(system_solution_block.block(1));
    SolverControl solver_control_eig(v_eig.size(), 1e-12, false);
    SolverCG<TrilinosWrappers::MPI::Vector> solver_eigs(solver_control_eig);

    solver_eigs.connect_condition_number_slot(
        std::bind(output_double_number, std::placeholders::_1,
                  "Condition number estimate: "));
    auto BBt = C * Ct;

    TrilinosWrappers::MPI::Vector u(v_eig);
    u = 0.;
    TrilinosWrappers::MPI::Vector f(v_eig);
    f = 1.;
    PreconditionIdentity prec_no;
    try {
      solver_eigs.solve(BBt, u, f, prec_no);
    } catch (...) {
      std::cerr
          << "***BBt solve not successfull (see condition number above)***"
          << std::endl;
      AssertThrow(false, ExcMessage("BBt does not have full rank."));
    }
  }

  return n_outer_iterations;
}

template <int dim>
void EllipticInterfaceDLM<dim>::output_results(
    const unsigned int ref_cycle) const {
  TimerOutput::Scope t(computing_timer, "Output results");

  if (parameters.do_convergence_study) {
    Vector<float> norm_per_cell(tria_bg.n_active_cells());
    VectorTools::integrate_difference(
        mapping, dof_handler_bg, system_solution_block.block(0),
        Solution<dim>(), norm_per_cell, QGauss<dim>(fe_bg.degree + 2),
        VectorTools::L2_norm);

    const double L2_error = VectorTools::compute_global_error(
        tria_bg, norm_per_cell, VectorTools::L2_norm);

    Vector<float> norm_per_cell_H1(tria_bg.n_active_cells());
    VectorTools::integrate_difference(
        mapping, dof_handler_bg, system_solution_block.block(0),
        Solution<dim>(), norm_per_cell_H1, QGauss<dim>(fe_bg.degree + 2),
        VectorTools::H1_norm);
    const double H1_error = VectorTools::compute_global_error(
        tria_bg, norm_per_cell_H1, VectorTools::H1_norm);

    // We add more columns if we want to show errors
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);

    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);

    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);

    convergence_table.evaluate_convergence_rates(
        "L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
        "H1", ConvergenceTable::reduction_rate_log2);
  }
  pcout << "==============================================================="
           "========================="
        << std::endl;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    convergence_table.write_text(
        std::cout, TableHandler::TextOutputFormat::org_mode_table);
  pcout << "==============================================================="
           "========================="
        << std::endl;

  // Do not dump grids to disk if they are too large
  if (tria_bg.n_active_cells() < 1e6) {
    DataOut<dim> data_out_fg;
    data_out_fg.attach_dof_handler(dof_handler_fg);
    data_out_fg.add_data_vector(system_solution_block.block(1), "u2");
    data_out_fg.add_data_vector(system_solution_block.block(2), "lambda");
    data_out_fg.build_patches();
    data_out_fg.write_vtu_with_pvtu_record(
        "", "solution-immersed-cycle-" + std::to_string(ref_cycle), 1, comm,
        numbers::invalid_unsigned_int, 0);

    DataOut<dim> data_out_bg;
    data_out_bg.attach_dof_handler(dof_handler_bg);
    data_out_bg.add_data_vector(system_solution_block.block(0), "u");

    Vector<float> subdomain(tria_bg.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_bg.locally_owned_subdomain();
    data_out_bg.add_data_vector(subdomain, "subdomain");

    data_out_bg.build_patches();

    data_out_bg.write_vtu_with_pvtu_record(
        "", "solution-background-cycle" + std::to_string(ref_cycle), 1, comm,
        numbers::invalid_unsigned_int, 0);

    pcout << "Solutions written to disk." << std::endl;
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::run() {
  AssertThrow(
      parameters.use_modified_AL_preconditioner,
      ExcMessage("Only modified AL preconditioner is supported in parallel."));

  for (unsigned int ref_cycle = 0; ref_cycle < parameters.n_refinement_cycles;
       ++ref_cycle) {
    pcout << "============================================================="
             "==========================="
          << std::endl;
    pcout << "Refinement cycle: " << ref_cycle << std::endl;
    pcout << "gamma_AL_background= " << parameters.gamma_AL_background
          << std::endl;
    pcout << "gamma_AL_immersed= " << parameters.gamma_AL_immersed << std::endl;
    // Create the grids only during the first refinement cycle
    if (ref_cycle == 0) {
      generate_grids();
    } else {
      // Otherwise, refine them globally.
      tria_bg.refine_global(1);
      tria_fg.refine_global(1);
    }

    system_setup();
    setup_coupling();
    assemble();
    solve();
    output_results(ref_cycle);
  }
}

int main(int argc, char *argv[]) {
  try {
    const int dim = 2;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    mpi_initlog(true, 10);

    ProblemParameters<dim> parameters;
    std::string parameter_file;
    if (argc > 1)
      parameter_file = argv[1];
    else
      parameter_file = "parameters_elliptic_interface.prm";
    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

    EllipticInterfaceDLM<dim> solver(parameters, MPI_COMM_WORLD);
    solver.run();
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}