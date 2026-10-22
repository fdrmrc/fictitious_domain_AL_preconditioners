#include <deal.II/base/convergence_table.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
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
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_common.h>

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "utilities.h"

using namespace dealii;

/**
 * Export a deal.II SparseMatrix (square or rectangular) to a MatrixMarket
 * coordinate file (.mtx). This is directly loadable in Python via
 * scipy.io.mmread(...), which returns a scipy.sparse COO matrix. The
 * MatrixMarket format uses 1-based indices; scipy converts to 0-based on read.
 */
void export_sparse_matrix_to_mtx(const SparseMatrix<double> &matrix,
                                 const std::string &filename) {
  std::ofstream out(filename);
  AssertThrow(out.is_open(),
              ExcMessage("Cannot open file " + filename + " for writing."));
  out.precision(16);
  out << std::scientific;

  // Count the entries with a nonzero value (the MatrixMarket header needs nnz).
  std::size_t nnz = 0;
  for (unsigned int row = 0; row < matrix.m(); ++row)
    for (auto e = matrix.begin(row); e != matrix.end(row); ++e)
      if (e->value() != 0.)
        ++nnz;

  out << "%%MatrixMarket matrix coordinate real general\n";
  out << matrix.m() << " " << matrix.n() << " " << nnz << "\n";
  for (unsigned int row = 0; row < matrix.m(); ++row)
    for (auto e = matrix.begin(row); e != matrix.end(row); ++e)
      if (e->value() != 0.)
        out << (row + 1) << " " << (e->column() + 1) << " " << e->value()
            << "\n";
  out.close();

  std::cout << "Exported " << matrix.m() << "x" << matrix.n() << " matrix to "
            << filename << std::endl;
}

// Analytical solution taken from https://arxiv.org/pdf/2211.03443
template <int dim> class Solution : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
  double return_value;

  auto r = p.norm();
  const double coefficient_omega = 1.;
  const double coefficient_omega2 = 10.;
  if (r <= 1) // inside domain omega2
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
template <int dim> class BoundaryValues : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const {
  return (4. - p[0] * p[0] - p[1] * p[1]) / 4.;
}

template <int dim> class ProblemParameters : public ParameterAcceptor {
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

  bool use_modified_AL_preconditioner = false;

  bool do_parameter_study = false;
  // Define range of values for gamma, in case we want to determine gamma
  // experimentally.
  double start_gamma = 1e-3;

  double end_gamma = 1.;

  unsigned int n_steps_gamma = 100;

  bool use_diagonal_inverse = false;

  bool use_sqrt_2_rule = false;

  bool do_sanity_checks = true;

  bool do_convergence_study = false;

  bool export_matrices_for_eig_analysis = false;

  // If true, the primal diagonal blocks A1 and (A2+M) of the Riesz
  // preconditioner are inverted exactly with a direct solver (UMFPACK) instead
  // of a single AMG V-cycle. The blocks are then exact SPD linear operators, so
  // MINRES stays valid and the outer count is robust, at the price of
  // scalability (direct factorization of the background matrix).
  bool use_direct_solver_diagonal_blocks = false;

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
      f_1("Right hand side f_1"), f_2_minus_f("Right hand side f_2 - f") {
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
  add_parameter(
      "Use direct solver for diagonal blocks",
      use_direct_solver_diagonal_blocks,
      "Invert the primal diagonal blocks A1 and (A2+M) of the Riesz "
      "preconditioner exactly with a direct solver (UMFPACK) instead "
      "of a single AMG V-cycle. Keeps the preconditioner SPD and "
      "linear so MINRES stays valid and robust, but is not scalable.");
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
    add_parameter("Use diagonal inverse", use_diagonal_inverse,
                  "Use diagonal approximation for the inverse (squared) of the "
                  "immersed mass matrix.");
    add_parameter("Use sqrt(2)-rule for gamma", use_sqrt_2_rule,
                  "Use sqrt(2)-rule for gamma. It makes sense only for "
                  "modified AL variant.");
    add_parameter("gamma fluid", gamma_AL_background);
    add_parameter("gamma solid", gamma_AL_immersed);
    add_parameter("Verbosity level", verbosity_level);
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
template <int dim> class EllipticInterfaceDLM {
public:
  typedef double Number;

  EllipticInterfaceDLM(const ProblemParameters<dim> &prm);

  void generate_grids();

  void system_setup();

  void setup_stiffness_matrix(const DoFHandler<dim> &dof_handler,
                              AffineConstraints<double> &constraints,
                              SparsityPattern &stiffness_sparsity,
                              SparseMatrix<Number> &stiffness_matrix) const;

  void setup_coupling();

  void assemble_subsystem(const FiniteElement<dim> &fe,
                          const DoFHandler<dim> &dof_handler,
                          const AffineConstraints<double> &constraints,
                          SparseMatrix<Number> &system_matrix,
                          Vector<Number> &system_rhs, double rho, double mu,
                          const Function<dim> &rhs_function) const;

  void assemble();

  // Returns the number of outer iterations.
  unsigned int solve(const unsigned int ref_cycle = 0);

  void output_results(const unsigned int ref_cycle) const;

  void run();

private:
  const ProblemParameters<dim> &parameters;
  Triangulation<dim> tria_bg;
  Triangulation<dim> tria_fg;

  MappingQ1<dim> mapping;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;
  FE_Q<dim> fe_bg;
  FE_Q<dim> fe_fg;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;

  SparsityPattern stiffness_sparsity_fg;
  SparsityPattern stiffness_sparsity_bg;
  SparsityPattern coupling_sparsity;
  SparsityPattern mass_sparsity_fg;

  SparseMatrix<Number> stiffness_matrix_bg;
  SparseMatrix<Number> stiffness_matrix_fg;
  SparseMatrix<Number>
      stiffness_matrix_fg_plus_id; // A_2 + gammaId, needed for modifiedAL
  SparseMatrix<Number>
      stiffness_matrix_fg_plus_M; // A_2 + gamma*M, needed for modifiedAL
  SparseMatrix<Number> coupling_matrix;

  SparseMatrix<Number> mass_matrix_fg;

  BlockVector<Number> system_rhs_block;
  BlockVector<Number> system_solution_block;

  mutable TimerOutput computing_timer;

  mutable ConvergenceTable convergence_table;

  std::unique_ptr<SolverCG<Vector<double>>> solver_lagrangian_scalar;
};

template <int dim>
EllipticInterfaceDLM<dim>::EllipticInterfaceDLM(
    const ProblemParameters<dim> &prm)
    : parameters(prm), dof_handler_bg(tria_bg), dof_handler_fg(tria_fg),
      fe_bg(parameters.background_space_finite_element_degree),
      fe_fg(parameters.immersed_space_finite_element_degree),
      computing_timer(MPI_COMM_WORLD, std::cout,
                      TimerOutput::every_call_and_summary,
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
                ExcNotImplemented()); // we check convergence rates only in 2D.

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
template <int dim> void EllipticInterfaceDLM<dim>::generate_grids() {
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
      // Rotate the immersed grid so that the two grids are non-matching. The
      // scalar-angle overload of GridTools::rotate only exists for spacedim==2,
      // hence the compile-time branch.
      if constexpr (dim == 2)
        GridTools::rotate(numbers::PI / 6.0, tria_fg);
      else if constexpr (dim == 3)
        GridTools::rotate(Tensor<1, dim>({0.0, 1.0, 0.0}), numbers::PI / 3.0,
                          tria_fg);
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
  std::cout << "h background = " << h_background << "\n"
            << "h immersed = " << h_immersed << "\n"
            << "grids ratio (background/immersed) = " << ratio << std::endl;

  // AssertThrow(ratio < 2.2 && ratio > 0.4,
  //             ExcMessage("Check mesh sizes of the two grids."));

  // Do not dump grids to disk if they are too large
  if (tria_bg.n_active_cells() < 1e3) {
    std::ofstream bg_out("background_grid.vtk");
    std::ofstream fg_out("immersed_grid.vtk");
    GridOut grid_out;
    grid_out.write_vtk(tria_bg, bg_out);
    grid_out.write_vtk(tria_fg, fg_out);
  }
}

template <int dim> void EllipticInterfaceDLM<dim>::system_setup() {
  TimerOutput::Scope t(computing_timer, "System setup");

  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);

  constraints_bg.clear();
  constraints_fg.clear();

  if (parameters.do_convergence_study)
    VectorTools::interpolate_boundary_values(
        dof_handler_bg, 0, BoundaryValues<dim>(), constraints_bg);
  else
    for (const types::boundary_id id : parameters.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
          dof_handler_bg, id, Functions::ZeroFunction<dim>(), constraints_bg);

  constraints_bg.close();
  constraints_fg.close();

  setup_stiffness_matrix(dof_handler_bg, constraints_bg, stiffness_sparsity_bg,
                         stiffness_matrix_bg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
                         stiffness_matrix_fg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
                         stiffness_matrix_fg_plus_id);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
                         stiffness_matrix_fg_plus_M);

  mass_matrix_fg.reinit(stiffness_sparsity_fg);

  system_rhs_block.reinit(3);
  system_rhs_block.block(0).reinit(dof_handler_bg.n_dofs());
  system_rhs_block.block(1).reinit(dof_handler_fg.n_dofs());
  system_rhs_block.block(2).reinit(dof_handler_fg.n_dofs());
  // The blocks were reinitialized individually; update the cached global size
  // (block_indices) so that BlockVector::size() is correct.
  system_rhs_block.collect_sizes();

  system_solution_block.reinit(system_rhs_block);

  std::cout << "N DoF background: " << dof_handler_bg.n_dofs() << std::endl;
  std::cout << "N DoF immersed: " << dof_handler_fg.n_dofs() << std::endl;

  std::cout << "==============================================================="
               "========================="
            << std::endl;
}

template <int dim>
void EllipticInterfaceDLM<dim>::setup_stiffness_matrix(
    const DoFHandler<dim> &dof_handler, AffineConstraints<double> &constraints,
    SparsityPattern &stiffness_sparsity,
    SparseMatrix<Number> &stiffness_matrix) const {
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  stiffness_sparsity.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity);
}

template <int dim> void EllipticInterfaceDLM<dim>::setup_coupling() {
  TimerOutput::Scope t(computing_timer, "Coupling setup");

  QGauss<dim> quad(2 * fe_bg.degree + 1);

  {
    DynamicSparsityPattern dsp(dof_handler_bg.n_dofs(),
                               dof_handler_fg.n_dofs());
    NonMatching::create_coupling_sparsity_pattern(
        dof_handler_bg, dof_handler_fg, quad, dsp, constraints_bg,
        ComponentMask(), ComponentMask(), mapping, mapping, constraints_fg);
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);

    NonMatching::create_coupling_mass_matrix(
        dof_handler_bg, dof_handler_fg, quad, coupling_matrix, constraints_bg,
        ComponentMask(), ComponentMask(), mapping, mapping, constraints_fg);
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::assemble_subsystem(
    const FiniteElement<dim> &fe, const DoFHandler<dim> &dof_handler,
    const AffineConstraints<double> &constraints,
    SparseMatrix<Number> &system_matrix, Vector<Number> &system_rhs, double rho,
    double mu, const Function<dim> &rhs_function) const {
  const QGauss<dim> quad(2 * fe.degree + 1);

  std::cout << "Assembling subsystem with mu = " << mu << std::endl;

  FEValues<dim> fe_values(fe, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> rhs_values(quad.size());

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);
    rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
              (rho * fe_values.shape_value(i, q_index) * // u
                   fe_values.shape_value(j, q_index)     // v
               + mu * fe_values.shape_grad(i, q_index) * // grad u
                     fe_values.shape_grad(j, q_index)) * // grad v
              fe_values.JxW(q_index);                    // dx

        cell_rhs(i) += (rhs_values[q_index] *               // f(x)
                        fe_values.shape_value(i, q_index) * // phi_i(x_q)
                        fe_values.JxW(q_index));            // dx
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int dim> void EllipticInterfaceDLM<dim>::assemble() {
  TimerOutput::Scope t(computing_timer, "Assemble matrices");

  // background stiffness matrix A
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, stiffness_matrix_bg,
                     system_rhs_block.block(0),
                     0.,                // 0, no mass matrix
                     parameters.beta_1, // only beta_1 (which is usually 1)
                     parameters.f_1);   // rhs value, f

  // immersed matrix A2 = (beta_2 - beta) (grad u,grad v)
  // The rhs changes if we want to do a convergence study.
  if (parameters.do_convergence_study)
    assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg,
                       stiffness_matrix_fg, system_rhs_block.block(1),
                       0., // 0, no mass matrix
                       parameters.beta_2 -
                           parameters.beta_1, // hardcoded before in this case
                       Functions::ConstantFunction<dim>{
                           0.}); // rhs value for convergence study
  else // read rhs values (constants) from parameters file
    assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg,
                       stiffness_matrix_fg, system_rhs_block.block(1),
                       0., // 0, no mass matrix
                       parameters.beta_2 -
                           parameters.beta_1,   // only jump beta_2 - beta
                       parameters.f_2_minus_f); // rhs value, f2-f

  //  mass matrix immersed
  assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg, mass_matrix_fg,
                     system_rhs_block.block(2), 1., 0.,
                     Functions::ConstantFunction<dim>{0.});
}

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

template <int dim>
unsigned int EllipticInterfaceDLM<dim>::solve(const unsigned int ref_cycle) {
  // Start with defining the outer iterations to an invalid value.
  unsigned int n_outer_iterations = numbers::invalid_unsigned_int;
  const double h_bg = GridTools::maximal_cell_diameter(tria_bg);
  std::cout << "h background = " << h_bg << std::endl;
  const double h_fg = GridTools::maximal_cell_diameter(tria_fg);
  std::cout << "h immersed = " << h_fg << std::endl;

  // ---------------------------------------------------------------------
  //  Operator-preconditioning approach.
  //
  //  We solve the *original* symmetric saddle-point system
  //
  //     [ A1    0    C^T ] [u ]   [f1]
  //     [ 0     A2  -M   ] [u2] = [f2]
  //     [ C    -M    0   ] [l ]   [0 ]
  //
  //  with A1 = beta_1 (grad u, grad v)_Omega, A2 = (beta_2-beta_1)(grad u2,
  //  grad w)_Omega2, M = (u2, w)_Omega2 the immersed mass matrix and C the
  //  L^2 coupling. The constraint row imposes u = u2 in L^2(Omega2), so the
  //  multiplier lambda lives in L^2(Omega2).
  // ---------------------------------------------------------------------

  auto A_omega1 = linear_operator(stiffness_matrix_bg); // A1
  auto A_omega2 = linear_operator(stiffness_matrix_fg); // A2
  auto M = linear_operator(mass_matrix_fg);             // immersed mass matrix
  auto Ct = linear_operator(coupling_matrix);           // C^T (im -> bg)
  auto C = transpose_operator(Ct);                      // C   (bg -> im)

  // Zero blocks needed to assemble the 3x3 block operator.
  auto NullBgIm = null_operator(Ct); // (0,1): im -> bg  zero block
  auto NullImBg = null_operator(C);  // (1,0): bg -> im  zero block
  auto NullImIm = null_operator(M);  // (2,2): im -> im  zero block (lambda)

  // Symmetric saddle-point system operator (no augmentation).
  auto system_operator = block_operator<3, 3, BlockVector<double>>(
      {{{{A_omega1, NullBgIm, Ct}},
        {{NullImBg, A_omega2, -1. * M}},
        {{C, -1. * M, NullImIm}}}});

  // The immersed (2,2) block A2 = (beta_2-beta_1)(grad u2, grad w) is only a
  // seminorm (constant nullspace). Its H^1 Riesz map therefore uses A2 + M,
  // which we assemble once into stiffness_matrix_fg_plus_M.
  stiffness_matrix_fg_plus_M.copy_from(stiffness_matrix_fg);
  stiffness_matrix_fg_plus_M.add(1.0, mass_matrix_fg);

#ifdef FALSE
  if (stiffness_matrix_bg.m() == 4225) {
    // Let's check for correctness
    BlockVector<double> ones(system_rhs_block);
    for (unsigned int b = 0; b < ones.n_blocks(); ++b)
      ones.block(b) = 1.;

    BlockVector<double> result(system_rhs_block);
    system_operator.vmult(result, ones);

    std::cout << "Result of system_operator * ones:" << std::endl;
    for (unsigned int b = 0; b < result.n_blocks(); ++b) {
      std::cout << "Block " << b << ":" << std::endl;
      for (unsigned int i = 0; i < result.block(b).size(); ++i)
        std::cout << result.block(b)[i] << std::endl;
    }

    std::cout << "Norm of the result: " << result.l2_norm() << std::endl;
  }
#endif

  // Primal Riesz-map blocks B1 ~ A1^{-1} (H^1_0, background) and
  // B2 ~ (A2+M)^{-1} (H^1, immersed). By default they are applied with a single
  // AMG V-cycle (cheap and scalable). If "Use direct solver for diagonal
  // blocks" is set, they are instead inverted exactly with UMFPACK: the blocks
  // are then exact SPD linear operators, so the block-diagonal preconditioner
  // stays SPD and MINRES remains valid, at the price of scalability.
  TrilinosWrappers::PreconditionAMG amg_prec_A11, amg_prec_A22;
  SparseDirectUMFPACK A1_inv_direct, A2M_inv_direct;
  if (parameters.use_direct_solver_diagonal_blocks) {
    TimerOutput::Scope t(computing_timer, "Factorize diagonal blocks (direct)");
    A1_inv_direct.initialize(stiffness_matrix_bg);
    A2M_inv_direct.initialize(stiffness_matrix_fg_plus_M);
    std::cout << "Initialized direct solvers for A1 and (A2 + M)" << std::endl;
  } else {
    typename TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_A11;
    amg_data_A11.n_cycles = 3;
    amg_data_A11.smoother_sweeps = 3;
    amg_data_A11.aggregation_threshold = 1e-2;
    amg_prec_A11.initialize(stiffness_matrix_bg, amg_data_A11);

    typename TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_A22;
    amg_data_A22.n_cycles = 3;
    amg_data_A22.smoother_sweeps = 3;
    amg_data_A22.aggregation_threshold = 1e-2;
    amg_prec_A22.initialize(stiffness_matrix_fg_plus_M, amg_data_A22);
    std::cout << "Initialized AMG for A1 and (A2 + M)" << std::endl;
  }

  if (parameters.export_matrices_for_eig_analysis) {
    // Export the sparse building blocks of the saddle-point system and of the
    // block-diagonal Riesz preconditioner to MatrixMarket (.mtx) files, one set
    // per refinement cycle, so that the spectrum of the preconditioned operator
    // B*A can be assembled and analyzed in Python
    // (spectrum_preconditioned.ipynb):
    //   A1 = beta_1 (grad u, grad v)_Omega           (background stiffness)
    //   A2 = (beta_2-beta_1)(grad u2, grad w)_Omega2  (immersed stiffness)
    //   M  = (u2, w)_Omega2                           (immersed mass)
    //   Ct = coupling matrix (background rows, immersed cols); C = Ct^T.
    // A2+M and K2+M = (1/delta) A2 + M are rebuilt in Python from A2 and M.
    const std::string dir = parameters.output_directory;
    const std::string tag = "_" + std::to_string(ref_cycle) + ".mtx";
    export_sparse_matrix_to_mtx(stiffness_matrix_bg, dir + "/A1" + tag);
    export_sparse_matrix_to_mtx(stiffness_matrix_fg, dir + "/A2" + tag);
    export_sparse_matrix_to_mtx(mass_matrix_fg, dir + "/M" + tag);
    export_sparse_matrix_to_mtx(coupling_matrix, dir + "/Ct" + tag);

    // Metadata needed to rebuild the operator (coefficients + block sizes).
    std::ofstream meta(dir + "/meta_" + std::to_string(ref_cycle) + ".txt");
    meta.precision(16);
    meta << "beta_1 " << parameters.beta_1 << "\n"
         << "beta_2 " << parameters.beta_2 << "\n"
         << "n_bg " << dof_handler_bg.n_dofs() << "\n"
         << "n_fg " << dof_handler_fg.n_dofs() << "\n"
         << "cells_bg " << tria_bg.n_active_cells() << "\n";
    meta.close();
    std::cout << "Exported matrices for cycle " << ref_cycle << " to " << dir
              << std::endl;
  }

  // L^2 Riesz map for the multiplier lambda: exact inverse of the immersed
  // mass matrix. UMFPACK gives a strictly linear, SPD application of M^{-1},
  // so the whole block-diagonal preconditioner stays a fixed SPD operator and
  // MINRES remains applicable.
  SparseDirectUMFPACK M_inv_umfpack;
  {
    TimerOutput::Scope t(computing_timer, "Factorize immersed mass matrix");
    M_inv_umfpack.initialize(mass_matrix_fg);
  }

  // Block-diagonal Riesz-map (norm-based) preconditioner:
  //   B = diag( A1^{-1}, (A2 + M)^{-1}, N_{-1}^{-1} )

  const double delta = parameters.beta_2 - parameters.beta_1;
  const double c_lambda =
      (parameters.beta_1 * delta) / (parameters.beta_1 + delta);

  auto Minv = linear_operator(mass_matrix_fg, M_inv_umfpack);
  auto Mop = linear_operator(mass_matrix_fg);
  // K2 = (1/delta) * A2 (unit-coefficient immersed stiffness), so K2 + M:
  auto K2plusM = (1.0 / delta) * A_omega2 + Mop;

  LinearOperator<Vector<double>> B1, B2;
  if (parameters.use_direct_solver_diagonal_blocks) {
    B1 = linear_operator(stiffness_matrix_bg, A1_inv_direct); // = A1^{-1}
    B2 = linear_operator(stiffness_matrix_fg_plus_M,
                         A2M_inv_direct); // = (A2+M)^{-1}
  } else {
    B1 = linear_operator(stiffness_matrix_bg, amg_prec_A11); // ~ A1^{-1}
    B2 = linear_operator(stiffness_matrix_fg_plus_M,
                         amg_prec_A22); // ~ (A2+M)^{-1}
  }
  //   B3 = c * M^{-1} (K2 + M) M^{-1}.
  auto B3 = c_lambda * (Minv * K2plusM * Minv); // ~ H^{-1}(Omega2),
  // jump-robust
  // auto A2 = 1 / delta * A_omega2; // unit-coefficient immersed stiffness
  // auto B3 = Minv * (c_lambda * A2 + parameters.beta_1 * Mop) *
  //           Minv; // ~ H^{-1}(Omega2), jump-robust

  const std::array<LinearOperator<Vector<double>>, 3> diagonal_blocks{
      {B1, B2, B3}};
  auto preconditioner =
      block_diagonal_operator<3, BlockVector<double>>(diagonal_blocks);

  SolverMinRes<BlockVector<Number>> solver_minres(
      parameters.outer_solver_control);

  // Wrap the actual solve in a scope so that the setup cost above is not timed
  // together with the Krylov iterations.
  {
    TimerOutput::Scope t(computing_timer, "Solve system");

    system_rhs_block.block(2) = 0; // the constraint right-hand side is zero

    solver_minres.solve(system_operator, system_solution_block,
                        system_rhs_block, preconditioner);

    std::cout << "Norm of solution: "
              << system_solution_block.block(0).l2_norm() << std::endl;

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
  convergence_table.add_value("Outer iterations (MINRES)", n_outer_iterations);

  std::cout << "Solved in " << n_outer_iterations << " iterations"
            << (parameters.outer_solver_control.last_step() < 10 ? "  " : " ")
            << "\n";

  // Do some sanity checks: Check the constraints residual and the condition
  // number of CCt.
  if (parameters.do_sanity_checks) {
    Vector<Number> difference_constraints = system_solution_block.block(2);

    coupling_matrix.Tvmult(difference_constraints,
                           system_solution_block.block(0));
    difference_constraints *= -1;

    mass_matrix_fg.vmult_add(difference_constraints,
                             system_solution_block.block(1));

    std::cout << "L infty norm of constraints residual "
              << difference_constraints.linfty_norm() << "\n";

    // Estimate condition number
    std::cout << "Estimate condition number of CCt using CG" << std::endl;
    Vector<double> v_eig(system_solution_block.block(1));
    SolverControl solver_control_eig(v_eig.size(), 1e-12, false);
    SolverCG<Vector<double>> solver_eigs(solver_control_eig);

    solver_eigs.connect_condition_number_slot(
        std::bind(output_double_number, std::placeholders::_1,
                  "Condition number estimate: "));
    auto BBt = C * Ct;

    Vector<double> u(v_eig);
    u = 0.;
    Vector<double> f(v_eig);
    f = 1.;
    PreconditionIdentity prec_no;
    try {
      solver_eigs.solve(BBt, u, f, prec_no);
    } catch (...) {
      // C*C^T is generally rank-deficient in this fictitious-domain coupling
      // (that is precisely what motivates the norm-based / AL preconditioners),
      // so a failed condition-number estimate here is expected and must not
      // abort the refinement study. We only warn.
      std::cerr << "***BBt condition-number estimate did not converge (BBt is "
                   "likely rank deficient; see estimate above). Continuing.***"
                << std::endl;
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
  std::cout << "==============================================================="
               "========================="
            << std::endl;
  convergence_table.write_text(std::cout,
                               TableHandler::TextOutputFormat::org_mode_table);
  std::cout << "==============================================================="
               "========================="
            << std::endl;

  // Do not dump grids to disk if they are too large
  if (tria_bg.n_active_cells() < 2e6) {
    DataOut<dim> data_out_fg;
    data_out_fg.attach_dof_handler(dof_handler_fg);
    data_out_fg.add_data_vector(system_solution_block.block(1), "u2");
    data_out_fg.add_data_vector(system_solution_block.block(2), "lambda");
    data_out_fg.build_patches();
    std::ofstream output_fg(parameters.output_directory + "/" +
                            "solution-immersed-" + std::to_string(ref_cycle) +
                            ".vtu");
    data_out_fg.write_vtu(output_fg);

    DataOut<dim> data_out_bg;
    data_out_bg.attach_dof_handler(dof_handler_bg);
    data_out_bg.add_data_vector(system_solution_block.block(0), "u");
    data_out_bg.build_patches();
    std::ofstream output_bg(parameters.output_directory + "/" +
                            "solution-background-" + std::to_string(ref_cycle) +
                            ".vtu");
    data_out_bg.write_vtu(output_bg);
    std::cout << "Solutions written to disk." << std::endl;
  }
}

template <int dim> void EllipticInterfaceDLM<dim>::run() {

  for (unsigned int ref_cycle = 0; ref_cycle < parameters.n_refinement_cycles;
       ++ref_cycle) {
    std::cout << "============================================================="
                 "==========================="
              << std::endl;
    std::cout << "Refinement cycle: " << ref_cycle << std::endl;
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
    solve(ref_cycle);
    if (parameters.use_modified_AL_preconditioner &&
        parameters.use_sqrt_2_rule) {
      // parameters.gamma_AL_background /=
      //     std::sqrt(2.); // using sqrt(2)-rule from modified - AL paper
      parameters.gamma_AL_immersed /=
          std::sqrt(2.); // using sqrt(2)-rule from modified - AL paper
    }
    output_results(ref_cycle);
  }
}

namespace {
// Build parameters, read the file, and run, for a chosen space dimension.
template <int dim> void run_simulation(const std::string &parameter_file) {
  ProblemParameters<dim> parameters;
  ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
  EllipticInterfaceDLM<dim> solver(parameters);
  solver.run();
}
} // namespace

int main(int argc, char *argv[]) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    deallog.depth_console(10);

    // Usage: ./elliptic_interface_riesz <parameter_file> [dim]
    // The optional second argument selects the space dimension (2 or 3;
    // default 2). The whole problem is templated on dim, so only the chosen
    // instantiation is constructed.
    std::string parameter_file =
        (argc > 1) ? std::string(argv[1]) : "parameters_elliptic_interface.prm";
    const unsigned int dim = (argc > 2) ? std::stoi(argv[2]) : 2u;

    AssertThrow(dim == 2 || dim == 3,
                ExcMessage("Only dim = 2 or dim = 3 are supported."));
    if (dim == 3)
      run_simulation<3>(parameter_file);
    else
      run_simulation<2>(parameter_file);
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