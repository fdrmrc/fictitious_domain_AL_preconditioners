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
#include <deal.II/dofs/dof_renumbering.h>
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
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
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

  // body forces
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> body_force_function;
  // bc for velocity
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> dirichlet_bc_function;

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
      f_2_minus_f("Right hand side f_2 - f"),
      body_force_function("Body force", dim),
      dirichlet_bc_function("Dirichlet boundary condition", dim + 1) {
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

  body_force_function.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "0;0;0");
  });

  dirichlet_bc_function.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "0;0;0;0");
  });
}

// The real class of the ellitpic interface problem.
template <int dim>
class StokesDLM {
 public:
  typedef double Number;

  StokesDLM(const ProblemParameters<dim> &prm);

  void generate_grids();

  void system_setup();

  void system_setup_stokes();

  void setup_stiffness_matrix(const DoFHandler<dim> &dof_handler,
                              AffineConstraints<double> &constraints_stokes,
                              SparsityPattern &stiffness_sparsity,
                              SparseMatrix<Number> &stiffness_matrix) const;

  void setup_coupling();

  void assemble_subsystem(const FiniteElement<dim> &fe,
                          const DoFHandler<dim> &dof_handler,
                          const AffineConstraints<double> &constraints_stokes,
                          SparseMatrix<Number> &system_matrix,
                          Vector<Number> &system_rhs, double rho, double mu,
                          const Function<dim> &rhs_function) const;

  void assemble_stokes();

  void assemble();

  // Returns the number of outer iterations.
  unsigned int solve();

  void output_results(const unsigned int ref_cycle) const;

  void run();

 private:
  const ProblemParameters<dim> &parameters;
  Triangulation<dim> tria_bg;
  Triangulation<dim> tria_fg;

  MappingQ1<dim> mapping;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;

  DoFHandler<dim> stokes_dh;
  DoFHandler<dim> velocity_dh;
  FE_Q<dim> fe_bg;
  FE_Q<dim> fe_fg;

  FESystem<dim> space_fe;
  FESystem<dim> velocity_fe;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;

  AffineConstraints<double> constraints_stokes;

  SparsityPattern stiffness_sparsity_fg;
  SparsityPattern stiffness_sparsity_bg;
  SparsityPattern coupling_sparsity;
  SparsityPattern mass_sparsity_fg;

  SparseMatrix<Number> stiffness_matrix_bg;
  SparseMatrix<Number> stiffness_matrix_fg;
  SparseMatrix<Number>
      stiffness_matrix_fg_plus_id;  // A_2 + gammaId, needed for modifiedAL
  SparseMatrix<Number> coupling_matrix;

  SparseMatrix<Number> mass_matrix_fg;

  // Stokes matrices
  BlockSparsityPattern sparsity_pattern_stokes;
  BlockSparseMatrix<double> stokes_matrix;

  BlockSparsityPattern preconditioner_sparsity_pattern;
  BlockSparseMatrix<double> preconditioner_matrix;

  BlockVector<Number> system_rhs_block;
  BlockVector<Number> system_solution_block;
  BlockVector<double> stokes_rhs;

  mutable TimerOutput computing_timer;

  mutable ConvergenceTable convergence_table;

  std::unique_ptr<SolverCG<Vector<double>>> solver_lagrangian_scalar;
};

template <int dim>
StokesDLM<dim>::StokesDLM(const ProblemParameters<dim> &prm)
    : parameters(prm),
      dof_handler_bg(tria_bg),
      dof_handler_fg(tria_fg),
      stokes_dh(tria_bg),
      velocity_dh(tria_bg),
      fe_bg(parameters.background_space_finite_element_degree),
      fe_fg(parameters.immersed_space_finite_element_degree),
      space_fe(FE_Q<dim>(2) ^ dim, FE_Q<dim>(2 - 1)),
      velocity_fe(FE_Q<dim>(2) ^ dim),
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
void StokesDLM<dim>::generate_grids() {
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
  std::cout << "h background = " << h_background << "\n"
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
void StokesDLM<dim>::system_setup() {
  TimerOutput::Scope t(computing_timer, "System setup");

  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);

  constraints_bg.clear();
  constraints_fg.clear();

  constraints_bg.close();
  constraints_fg.close();

  setup_stiffness_matrix(dof_handler_bg, constraints_bg, stiffness_sparsity_bg,
                         stiffness_matrix_bg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
                         stiffness_matrix_fg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
                         stiffness_matrix_fg_plus_id);

  mass_matrix_fg.reinit(stiffness_sparsity_fg);

  // std::cout << "N DoF background: " << dof_handler_bg.n_dofs() << std::endl;
  // std::cout << "N DoF immersed: " << dof_handler_fg.n_dofs() << std::endl;
}

template <int dim>
void StokesDLM<dim>::system_setup_stokes() {
  TimerOutput::Scope t(computing_timer, "System setup for Stokes");

  stokes_dh.distribute_dofs(space_fe);
  velocity_dh.distribute_dofs(velocity_fe);
  DoFRenumbering::Cuthill_McKee(stokes_dh);
  DoFRenumbering::Cuthill_McKee(
      velocity_dh);  // we need to renumber in the same way we renumbered DoFs
  // for velocity

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(stokes_dh, block_component);

  {
    constraints_stokes.clear();

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(stokes_dh, constraints_stokes);
    for (const unsigned int id : parameters.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
          stokes_dh, id, parameters.dirichlet_bc_function, constraints_stokes,
          space_fe.component_mask(velocities));
  }
  constraints_stokes.close();

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(stokes_dh, block_component);
  const types::global_dof_index n_u = dofs_per_block[0];
  const types::global_dof_index n_p = dofs_per_block[1];
  std::cout << "Number of degrees of freedom: " << stokes_dh.n_dofs() << " ("
            << n_u << '+' << dof_handler_fg.n_dofs() << '+'
            << dof_handler_fg.n_dofs() << '+' << n_p << ')' << std::endl;

  std::cout << "==============================================================="
               "========================="
            << std::endl;
  // Define blocksparsityPattern

  {
    BlockDynamicSparsityPattern dsp_stokes(dofs_per_block, dofs_per_block);
    Table<2, DoFTools::Coupling> coupling_table(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling_table[c][d] = DoFTools::always;
        else
          coupling_table[c][d] = DoFTools::none;

    std::vector<types::global_dof_index> dofs_immersed(
        dof_handler_fg.get_fe().n_dofs_per_cell());
    for (const auto &immersed_cell : dof_handler_fg.active_cell_iterators()) {
      immersed_cell->get_dof_indices(dofs_immersed);
      for (const types::global_dof_index idx_row : dofs_immersed)
        for (const types::global_dof_index idx_col : dofs_immersed)
          dsp_stokes.block(0, 0).add(idx_row, idx_col);
    }

    DoFTools::make_sparsity_pattern(stokes_dh, coupling_table, dsp_stokes,
                                    constraints_stokes, false);

    sparsity_pattern_stokes.copy_from(dsp_stokes);
  }

  {
    BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                   dofs_per_block);

    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (((c == dim) && (d == dim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(stokes_dh, preconditioner_coupling,
                                    preconditioner_dsp, constraints_stokes,
                                    false);

    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
  }

  stokes_matrix.reinit(sparsity_pattern_stokes);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

  // Initialize matrices
  stokes_matrix.reinit(sparsity_pattern_stokes);

  // DynamicSparsityPattern mass_dsp(dof_handler_fg.n_dofs(),
  //                                 dof_handler_fg.n_dofs());
  // DoFTools::make_sparsity_pattern(*dof_handler_fg, mass_dsp);
  // mass_sparsity.copy_from(mass_dsp);
  // mass_matrix_immersed.reinit(mass_sparsity);  // M_immersed

  // Initialize vectors
  // solution.reinit(dofs_per_block);
  stokes_rhs.reinit(dofs_per_block);

  system_rhs_block.reinit(4);
  system_rhs_block.block(0).reinit(n_u);                      // u
  system_rhs_block.block(1).reinit(dof_handler_fg.n_dofs());  // X
  system_rhs_block.block(2).reinit(dof_handler_fg.n_dofs());  // lambda
  system_rhs_block.block(3).reinit(n_p);                      // p

  system_solution_block.reinit(system_rhs_block);
}

template <int dim>
void StokesDLM<dim>::setup_stiffness_matrix(
    const DoFHandler<dim> &dof_handler,
    AffineConstraints<double> &constraints_stokes,
    SparsityPattern &stiffness_sparsity,
    SparseMatrix<Number> &stiffness_matrix) const {
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_stokes);
  stiffness_sparsity.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity);
}

template <int dim>
void StokesDLM<dim>::setup_coupling() {
  TimerOutput::Scope t(computing_timer, "Coupling setup");

  QGauss<dim> quad(fe_bg.degree + 1);

  {
    DynamicSparsityPattern dsp(velocity_dh.n_dofs(), dof_handler_fg.n_dofs());
    NonMatching::create_coupling_sparsity_pattern(
        velocity_dh, dof_handler_fg, quad, dsp, constraints_stokes,
        ComponentMask(), ComponentMask(), mapping, mapping, constraints_fg);
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);

    NonMatching::create_coupling_mass_matrix(
        velocity_dh, dof_handler_fg, quad, coupling_matrix, constraints_stokes,
        ComponentMask(), ComponentMask(), mapping, mapping, constraints_fg);
  }
}

template <int dim>
void StokesDLM<dim>::assemble_subsystem(
    const FiniteElement<dim> &fe, const DoFHandler<dim> &dof_handler,
    const AffineConstraints<double> &constraints_stokes,
    SparseMatrix<Number> &system_matrix, Vector<Number> &system_rhs, double rho,
    double mu, const Function<dim> &rhs_function) const {
  const QGauss<dim> quad(fe.degree + 1);

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
    constraints_stokes.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int dim>
void StokesDLM<dim>::assemble_stokes() {
  stokes_matrix = 0;
  stokes_rhs = 0;
  preconditioner_matrix = 0;
  const QGauss<dim> quadrature_formula(space_fe.degree + 2);

  FEValues<dim> fe_values(space_fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();

  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs(dofs_per_cell);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Vector<double>> body_force_values(n_q_points,
                                                Vector<double>(dim));

  // Precompute stuff for Stokes' weak form
  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);

  for (const auto &cell : stokes_dh.active_cell_iterators()) {
    fe_values.reinit(cell);
    local_matrix = 0;
    local_rhs = 0;
    local_preconditioner_matrix = 0;

    parameters.body_force_function.vector_value_list(
        fe_values.get_quadrature_points(), body_force_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      Tensor<1, dim> body_force_values_tensor{
          ArrayView{body_force_values[q].begin(), body_force_values[q].size()}};

      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
        grad_phi_u[k] = fe_values[velocities].gradient(k, q);
        div_phi_u[k] = fe_values[velocities].divergence(k, q);
        phi_u[k] = fe_values[velocities].value(k, q);
        phi_p[k] = fe_values[pressure].value(k, q);
      }

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j <= i; ++j) {
          // if (augmented_lagrangian_control.grad_div_stabilization == true) {
          //   local_matrix(i, j) +=
          //       (1. * scalar_product(grad_phi_u[i],
          //                            grad_phi_u[j])  // symgrad-symgrad
          //        - div_phi_u[i] * phi_p[j]           // div u_i p_j
          //        - phi_p[i] * div_phi_u[j]           // p_i div u_j) *
          //       fe_values.JxW(q);
          //   // TODO: grad-div stabilization?
          // } else {
          // no grad-div stabilization, usual formulation
          local_matrix(i, j) +=
              (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])  // symgrad-symgrad
               - div_phi_u[i] * phi_p[j]                  // div u_i p_j
               - phi_p[i] * div_phi_u[j]) *               // p_i div u_j
              fe_values.JxW(q);
          // }

          local_preconditioner_matrix(i, j) +=
              (phi_p[i] * phi_p[j]) * fe_values.JxW(q);  // p_i p_j
        }

        // local_rhs(i) += phi_u[i] * rhs_values[q] * fe_values.JxW(q);
        local_rhs(i) += phi_u[i] * body_force_values_tensor * fe_values.JxW(q);
      }
    }

    // exploit symmetry
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j)

      {
        local_matrix(i, j) = local_matrix(j, i);
        local_preconditioner_matrix(i, j) = local_preconditioner_matrix(j, i);
      }

    cell->get_dof_indices(local_dof_indices);
    constraints_stokes.distribute_local_to_global(
        local_matrix, local_rhs, local_dof_indices, stokes_matrix, stokes_rhs);

    constraints_stokes.distribute_local_to_global(
        local_preconditioner_matrix, local_dof_indices, preconditioner_matrix);
  }

  deallog << "A_f dimensions (" << stokes_matrix.block(0, 0).m() << ","
          << stokes_matrix.block(0, 0).n() << ")" << std::endl;
  deallog << "B dimensions (" << stokes_matrix.block(1, 0).m() << ","
          << stokes_matrix.block(1, 0).n() << ")" << std::endl;
  deallog << "C dimensions (" << coupling_matrix.n() << ","
          << coupling_matrix.m() << ")" << std::endl;
}

template <int dim>
void StokesDLM<dim>::assemble() {
  TimerOutput::Scope t(computing_timer, "Assemble matrices solid");

  // background stiffness matrix A
  // assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg,
  // stiffness_matrix_bg,
  //                    system_rhs_block.block(0),
  //                    0.,                 // 0, no mass matrix
  //                    parameters.beta_1,  // only beta_1 (which is usually 1)
  //                    parameters.f_1);    // rhs value, f

  // immersed matrix A2 = (beta_2 - beta) (grad u,grad v)

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

  deallog << "A_s dimensions (" << stiffness_matrix_fg.m() << ","
          << stiffness_matrix_fg.n() << ")" << std::endl;
}

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

template <int dim>
unsigned int StokesDLM<dim>::solve() {
  // Start with defining the outer iterations to an invalid value.
  unsigned int n_outer_iterations = numbers::invalid_unsigned_int;

  auto A_omega1 = linear_operator(stiffness_matrix_bg);
  auto A_omega2 = linear_operator(stiffness_matrix_fg);
  auto M = linear_operator(mass_matrix_fg);
  auto NullS = null_operator(linear_operator(mass_matrix_fg));
  auto NullCouplin = null_operator(linear_operator(coupling_matrix));
  auto Ct = linear_operator(coupling_matrix);
  auto C = transpose_operator(Ct);

  auto A = linear_operator(stokes_matrix.block(0, 0));
  auto NullF = null_operator(A);
  auto Bt = linear_operator(stokes_matrix.block(0, 1));
  auto B = linear_operator(stokes_matrix.block(1, 0));

  // auto system_operator = block_operator<4, 4, BlockVector<double>>(
  //     {{{{A, NullF, Ct, Bt}},
  //       {{NullS, A_omega2, -1. * M, NullS}},
  //       {{C, -1. * M, NullCouplin, NullS}},
  //       {{B, NullS, NullS, NullS}}}});

  // SolverGMRES<BlockVector<double>> solver(parameters.outer_solver_control);
  // PreconditionIdentity prec_id;

  // system_rhs_block.block(0) = stokes_rhs.block(0);

  // solver.solve(system_operator, system_solution_block, system_rhs_block,
  //              prec_id);

  // We create empty operators for the action of the inverse of W. Depending
  // on the the choice, we either approximate it using the diagonal of the
  // mass matrix (squaring the entries) or we use the direct inversion of the
  // mass matrix provided by UMFPACK.
  auto invM = null_operator(M);
  auto invW = null_operator(M);
  SparseDirectUMFPACK M_inv_umfpack;
  SparseDirectUMFPACK Mp_inv_umfpack;
  Mp_inv_umfpack.initialize(preconditioner_matrix.block(1, 1));
  auto invMp =
      linear_operator(preconditioner_matrix.block(1, 1), Mp_inv_umfpack);

  // Using inverse of diagonal mass matrix (squared)
  Vector<double> inverse_diag_mass_squared(dof_handler_fg.n_dofs());
  for (unsigned int i = 0; i < dof_handler_fg.n_dofs(); ++i)
    inverse_diag_mass_squared[i] =
        1. / (mass_matrix_fg.diag_element(i) * mass_matrix_fg.diag_element(i));

  DiagonalMatrix<Vector<double>> diag_inverse;
  if (parameters.use_diagonal_inverse == true) {
    diag_inverse.reinit(inverse_diag_mass_squared);
    invW = linear_operator(diag_inverse);
  } else {
    // Use direct inversion
    {
      TimerOutput::Scope t(computing_timer, "Factorize mass matrix");
      M_inv_umfpack.initialize(mass_matrix_fg);  // inverse immersed mass matrix
    }
    invM = linear_operator(mass_matrix_fg, M_inv_umfpack);
    invW = invM * invM;
  }

  // Define augmented blocks. Notice that A22_aug is actually A_omega2 +
  // gamma_AL_background * Id
  auto A11_aug = A + parameters.gamma_AL_background * Ct * invW * C +
                 parameters.gamma_AL_background * Bt * invMp * B;
  auto A22_aug = A_omega2 + parameters.gamma_AL_immersed * M * invW * M;
  auto A12_aug = -parameters.gamma_AL_background * Ct * invW * M;
  // Next one is just transpose_operator(A12_aug);
  auto A21_aug = -parameters.gamma_AL_immersed * M * invW * C;

  // Augmented (equivalent) system to be solved
  auto system_operator = block_operator<4, 4, BlockVector<double>>(
      {{{{A11_aug, A12_aug, Ct, Bt}},
        {{A21_aug, A22_aug, -1. * M, NullS}},
        {{C, -1. * M, NullCouplin, NullS}},
        {{B, NullS, NullS, NullS}}}});

  // Initialize AMG preconditioners for inner solves
  // TrilinosWrappers::PreconditionAMG amg_prec_A11;
  // build_AMG_augmented_block_scalar(
  //     dof_handler_bg, coupling_matrix, stiffness_matrix_bg,
  //     inverse_diag_mass_squared, coupling_sparsity, constraints_bg,
  //     parameters.gamma_AL_background, parameters.beta_1, amg_prec_A11);

  // Initialize AMG prec for the A_2 augmented block
  TrilinosWrappers::PreconditionAMG amg_prec_A22;
  // immersed matrix (beta_2 - beta) (grad u,grad v) + gamma*Id
  stiffness_matrix_fg_plus_id.copy_from(stiffness_matrix_fg);
  // Add gamma*Id to A_2
  for (unsigned int i = 0; i < stiffness_matrix_fg_plus_id.m(); ++i)
    stiffness_matrix_fg_plus_id.add(i, i, parameters.gamma_AL_immersed);
  amg_prec_A22.initialize(stiffness_matrix_fg_plus_id);
  std::cout << "Initialized AMG for A_2" << std::endl;

  if (parameters.export_matrices_for_eig_analysis) {
    std::cout << "Exporting matrices to .csv for eigenvalues analysis...";
    export_to_matlab_csv(stiffness_matrix_bg, "A_DLFDM.csv");
    export_to_matlab_csv(stiffness_matrix_fg, "A_2_DLFDM.csv");
    export_to_matlab_csv(coupling_matrix, "Ct_DLFDM.csv");
    export_to_matlab_csv(mass_matrix_fg, "M_DLFDM.csv");
    std::cout << "Done." << std::endl;
  }

  typename SolverFGMRES<BlockVector<Number>>::AdditionalData data_fgmres;
  data_fgmres.max_basis_size = 50;
  SolverFGMRES<BlockVector<Number>> solver_fgmres(
      parameters.outer_solver_control, data_fgmres);

  // Wrap the actual FGMRES solve in another scope in order to discard
  // setup-cost.
  {
    TimerOutput::Scope t(computing_timer, "Solve system");

    // If we use the modified AL preconditioner, we check if we use a small
    // value of gamma.
    AssertThrow(
        parameters.gamma_AL_immersed <= 20.,
        ExcMessage(
            "gamma_AL_immersed is too large for modified ALpreconditioner."));

    // We wither use a fixed number of iterations inside the inner solver...
    if (parameters.use_fixed_iterations)
      solver_lagrangian_scalar = std::make_unique<SolverCG<Vector<double>>>(
          parameters.iteration_number_control);
    else  //... or we iterate as usual.
      solver_lagrangian_scalar = std::make_unique<SolverCG<Vector<double>>>(
          parameters.inner_solver_control);

    // Define block preconditioner using AL approach
    // auto A11_aug_inv =
    //     inverse_operator(A11_aug, *solver_lagrangian_scalar, amg_prec_A11);
    //     TODO: adapt to stokes case
    PreconditionIdentity prec_id;
    auto A11_aug_inv = inverse_operator(A11_aug, *solver_lagrangian_scalar);
    auto A22_aug_inv =
        inverse_operator(A22_aug, *solver_lagrangian_scalar, amg_prec_A22);

    StokesDLMALPreconditioner preconditioner_AL(
        A11_aug_inv, A22_aug_inv, Bt, Ct, invW, invMp, M,
        parameters.gamma_AL_background, parameters.gamma_AL_immersed,
        parameters.gamma_AL_background);

    // system_rhs_block.block(2) = 0;  // last row of the rhs is 0

    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, preconditioner_AL);

    // std::cout << "Norm of solution: "
    //           << system_solution_block.block(0).l2_norm() << std::endl;
  }
  //   } else {
  //     // Check that gamma is not too small. We force also gamma2 to be
  //     equal to
  //     // gamma.
  //     AssertThrow(
  //         parameters.gamma_AL_background > 1.,
  //         ExcMessage("Parameter gamma is probably too small for classical
  //         AL
  //         "
  //                    "preconditioner."));
  //     parameters.gamma_AL_immersed = parameters.gamma_AL_background;

  //     std::cout << "\t ************************************** WARNING "
  //                  "************************************** \n"
  //                  "\t USING IDEAL AL PRECONDITIONER. SHOULD BE USED ONLY
  //                  FOR " "TESTING PURPOSES.\n"
  //               << "\t ***********************************************"
  //                  "************************************** "
  //               << std::endl;

  //     auto AMG_A1 = linear_operator(stiffness_matrix_bg, amg_prec_A11);
  //     auto AMG_A2 = linear_operator(stiffness_matrix_fg, amg_prec_A22);
  //     // Define preconditioner for the augmented block
  //     auto prec_aug = block_operator<2, 2, BlockVector<double>>(
  //         {{{{AMG_A1, NullF}}, {{NullS, AMG_A2}}}});
  //     PreconditionIdentity prec_id;
  //     auto Aug = block_operator<2, 2, BlockVector<double>>(
  //         {{{{A11_aug, A12_aug}},
  //           {{A21_aug, A22_aug}}}});  // augmented block to be inverted

  //     SolverCG<BlockVector<double>> solver_lagrangian(
  //         parameters.inner_solver_control);
  //     auto Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_aug);

  //     EllipticInterfacePreconditioners::BlockTriangularALPreconditioner
  //         preconditioner_AL(Aug_inv, C, M, invW,
  //                           parameters.gamma_AL_background);
  //     system_rhs_block.block(2) = 0;  // last row of the rhs is 0
  //     solver_fgmres.solve(system_operator, system_solution_block,
  //                         system_rhs_block, preconditioner_AL);
  //   }
  //   // Finally, distribute the constraints_stokes
  //   constraints_bg.distribute(system_solution_block.block(0));
  //   constraints_fg.distribute(system_solution_block.block(1));
  //   constraints_fg.distribute(system_solution_block.block(2));
  // }

  // n_outer_iterations = parameters.outer_solver_control.last_step();

  // // We update the table with iteration counts
  // convergence_table.add_value("cells", tria_bg.n_active_cells());
  // convergence_table.add_value("DoF background", dof_handler_bg.n_dofs());
  // convergence_table.add_value("DoF immersed", dof_handler_fg.n_dofs());
  // convergence_table.add_value("gamma (AL)",
  // parameters.gamma_AL_background); if
  // (parameters.use_modified_AL_preconditioner)
  //   convergence_table.add_value("gamma2 (AL)",
  //   parameters.gamma_AL_immersed);
  // convergence_table.add_value("Outer iterations", n_outer_iterations);

  // std::cout << "Solved in " << n_outer_iterations << " iterations"
  //           << (parameters.outer_solver_control.last_step() < 10 ? "  " : "
  //           ")
  //           << "\n";

  // // Do some sanity checks: Check the constraints_stokes residual and the
  // // condition number of CCt.
  // if (parameters.do_sanity_checks) {
  //   Vector<Number> difference_constraints = system_solution_block.block(2);

  //   coupling_matrix.Tvmult(difference_constraints,
  //                          system_solution_block.block(0));
  //   difference_constraints *= -1;

  //   mass_matrix_fg.vmult_add(difference_constraints,
  //                            system_solution_block.block(1));

  //   std::cout << "L infty norm of constraints_stokes residual "
  //             << difference_constraints.linfty_norm() << "\n";

  //   // Estimate condition number
  //   std::cout << "Estimate condition number of CCt using CG" << std::endl;
  //   Vector<double> v_eig(system_solution_block.block(1));
  //   SolverControl solver_control_eig(v_eig.size(), 1e-12, false);
  //   SolverCG<Vector<double>> solver_eigs(solver_control_eig);

  //   solver_eigs.connect_condition_number_slot(
  //       std::bind(output_double_number, std::placeholders::_1,
  //                 "Condition number estimate: "));
  //   auto BBt = C * Ct;

  //   Vector<double> u(v_eig);
  //   u = 0.;
  //   Vector<double> f(v_eig);
  //   f = 1.;
  //   PreconditionIdentity prec_no;
  //   try {
  //     solver_eigs.solve(BBt, u, f, prec_no);
  //   } catch (...) {
  //     std::cerr
  //         << "***BBt solve not successfull (see condition number above)***"
  //         << std::endl;
  //     AssertThrow(false, ExcMessage("BBt does not have full rank."));
  //   }
  // }

  return n_outer_iterations;
}

template <int dim>
void StokesDLM<dim>::output_results(const unsigned int ref_cycle) const {
  TimerOutput::Scope t(computing_timer, "Output results");

  // Do not dump grids to disk if they are too large
  if (tria_bg.n_active_cells() < 1e6) {
    // DataOut<dim> data_out_fg;
    // data_out_fg.attach_dof_handler(dof_handler_fg);
    // data_out_fg.add_data_vector(system_solution_block.block(1), "u2");
    // data_out_fg.add_data_vector(system_solution_block.block(2), "lambda");
    // data_out_fg.build_patches();
    // std::ofstream output_fg(parameters.output_directory + "/" +
    //                         "solution-immersed-" +
    //                         std::to_string(ref_cycle)
    //                         +
    //                         ".vtu");
    // data_out_fg.write_vtu(output_fg);

    // DataOut<dim> data_out_bg;
    // data_out_bg.attach_dof_handler(dof_handler_bg);
    // data_out_bg.add_data_vector(system_solution_block.block(0), "u");
    // data_out_bg.build_patches();
    // std::ofstream output_bg(parameters.output_directory + "/" +
    //                         "solution-background-" +
    //                         std::to_string(ref_cycle) +
    //                         ".vtu");
    // data_out_bg.write_vtu(output_bg);
    // std::cout << "Solutions written to disk." << std::endl;

    {
      std::vector<unsigned int> block_component(dim + 1, 0);
      block_component[dim] = 1;

      const std::vector<types::global_dof_index> dofs_per_block =
          DoFTools::count_dofs_per_fe_block(stokes_dh, block_component);

      BlockVector<double> solution_stokes;
      solution_stokes.reinit(dofs_per_block);
      solution_stokes.block(0) = system_solution_block.block(0);
      solution_stokes.block(1) = system_solution_block.block(3);

      std::vector<std::string> solution_names(dim, "velocity");
      solution_names.emplace_back("pressure");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back(
          DataComponentInterpretation::component_is_scalar);

      DataOut<dim> data_out_stokes;
      data_out_stokes.attach_dof_handler(stokes_dh);
      data_out_stokes.add_data_vector(solution_stokes, solution_names,
                                      DataOut<dim>::type_dof_data,
                                      data_component_interpretation);
      data_out_stokes.build_patches();

      std::ofstream output("solution-stokes.vtk");
      data_out_stokes.write_vtk(output);
    }
  }
}

template <int dim>
void StokesDLM<dim>::run() {
  // If we want to select the gamma value experimentally, we first run a
  // coarse problem for different sampled values of gamma. This makes sense
  // only for the modified AL preconditioner, so we check beforehand if it is
  // used.

  // if (parameters.do_parameter_study &&
  //     parameters.use_modified_AL_preconditioner) {
  //   std::vector<double> gamma_values = linspace(
  //       parameters.start_gamma, parameters.end_gamma,
  //       parameters.n_steps_gamma);
  //   std::vector<unsigned int> outer_iterations;
  //   generate_grids();
  //   system_setup();
  //   setup_coupling();
  //   assemble();
  //   // Loop over possible values of gamma and store outer iterations.
  //   for (const double gamma : gamma_values) {
  //     parameters.gamma_AL_background = gamma;
  //     parameters.gamma_AL_immersed = gamma;
  //     std::cout << "gamma_AL_background= " <<
  //     parameters.gamma_AL_background
  //               << std::endl;
  //     unsigned int iters = solve();
  //     outer_iterations.push_back(iters);
  //     system_solution_block = 0;  // reset solution
  //   }
  //   // Find the minimum index
  //   const unsigned int min_index =
  //       std::min_element(outer_iterations.begin(), outer_iterations.end())
  //       - outer_iterations.begin();

  //   parameters.gamma_AL_background = gamma_values[min_index];
  //   parameters.gamma_AL_immersed = parameters.gamma_AL_background;
  //   std::cout <<
  //   "============================================================="
  //                "==========================="
  //             << std::endl;
  //   std::cout << "OPTIMAL VALUE FOR GAMMA FOUND EXPERIMENTALLY: "
  //             << parameters.gamma_AL_background << std::endl;
  //   std::cout << "START CONVERGENCE STUDY WITH GAMMA: "
  //             << parameters.gamma_AL_background << std::endl;

  //   // If we have determined the optimal gamma value, we can proceed with
  //   the
  //   // refinement cycles using such a value of gamma.
  // }

  // reset solution and grids
  // system_solution_block = 0;
  // tria_bg.clear();
  // tria_fg.clear();
  // convergence_table.clear();

  for (unsigned int ref_cycle = 0; ref_cycle < parameters.n_refinement_cycles;
       ++ref_cycle) {
    std::cout << "============================================================="
                 "==========================="
              << std::endl;
    std::cout << "Refinement cycle: " << ref_cycle << std::endl;
    std::cout << "gamma_AL_background= " << parameters.gamma_AL_background
              << std::endl;
    std::cout << "gamma_AL_immersed= " << parameters.gamma_AL_immersed
              << std::endl;
    // Create the grids only during the first refinement cycle
    if (ref_cycle == 0) {
      generate_grids();
    } else {
      // Otherwise, refine them globally.
      tria_bg.refine_global(1);
      tria_fg.refine_global(1);
    }

    system_setup();
    system_setup_stokes();
    setup_coupling();
    assemble();
    assemble_stokes();
    solve();
    if (parameters.use_modified_AL_preconditioner &&
        parameters.use_sqrt_2_rule) {
      parameters.gamma_AL_background /=
          std::sqrt(2.);  // using sqrt(2)-rule from modified - AL paper
      parameters.gamma_AL_immersed /=
          std::sqrt(2.);  // using sqrt(2)-rule from modified - AL paper
    }
    output_results(ref_cycle);
  }
}

int main(int argc, char *argv[]) {
  try {
    const int dim = 2;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    deallog.depth_console(10);

    ProblemParameters<dim> parameters;
    std::string parameter_file;
    if (argc > 1)
      parameter_file = argv[1];
    else
      parameter_file = "parameters_stokes_DLM.prm";
    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

    StokesDLM<dim> solver(parameters);
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