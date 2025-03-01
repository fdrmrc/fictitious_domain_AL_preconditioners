#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/diagonal_matrix.h>
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

#include <exception>
#include <fstream>
#include <iostream>
#include <memory>

// We include the header file where AL preconditioners are implemented
#include "augmented_lagrangian_preconditioner.h"

using namespace dealii;

// Class
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
  double beta_1 = 1.;

  // beta_2, instead, is the one which is changed.
  double beta_2 = 10.;

  std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

  unsigned int background_space_finite_element_degree = 1;

  unsigned int immersed_space_finite_element_degree = 1;

  unsigned int coupling_quadrature_order = 3;

  unsigned int verbosity_level = 10;

  bool use_modified_AL_preconditioner = false;

  bool use_diagonal = false;

  bool use_sqrt_2_rule = false;

  // AL parameter. Its magnitude depends on which AL preconditioner (original
  // vs. modified AL) is chosen. We define it as mutable since with modified AL
  // its value may change upon mesh refinement.
  mutable double gamma_AL = 10.;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      rhs;  //(f,f_2,0)

  mutable ParameterAcceptorProxy<ReductionControl> outer_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> inner_solver_control;
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
    : ParameterAcceptor("Elliptic Interface Problem/"),
      rhs("Right hand side", dim + 1),
      outer_solver_control("Outer solver control"),
      inner_solver_control("Inner solver control") {
  add_parameter("FE degree background", background_space_finite_element_degree,
                "", this->prm, Patterns::Integer(1));

  add_parameter("FE degree immersed", immersed_space_finite_element_degree, "",
                this->prm, Patterns::Integer(1));

  add_parameter("Coupling quadrature order", coupling_quadrature_order);

  add_parameter("Output directory", output_directory);

  add_parameter("Beta_1", beta_1);
  add_parameter("Beta_2", beta_2);

  add_parameter("Homogeneous Dirichlet boundary ids", dirichlet_ids);

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
        "Initial number of refinements used for the background domain Omega");
    add_parameter(
        "Initial immersed refinement", initial_immersed_refinement,
        "Initial number of refinements used for the immersed domain Gamma");
    add_parameter("Refinemented cycles", n_refinement_cycles,
                  "Number of refinement cycles to perform convergence studies");
  }
  leave_subsection();

  enter_subsection("AL preconditioner");
  {
    add_parameter("Use modified AL preconditioner",
                  use_modified_AL_preconditioner,
                  "Use the modified AL preconditioner.");
    add_parameter("Use diagonal", use_diagonal,
                  "Use diagonal approximation of mass matrices.");
    add_parameter("Use sqrt(2)-rule for gamma", use_sqrt_2_rule,
                  "Use sqrt(2)-rule for gamma. It makes sense only for "
                  "modified AL variant.");
    add_parameter("gamma", gamma_AL);
    add_parameter("Verbosity level", verbosity_level);
  }
  leave_subsection();

  rhs.declare_parameters_call_back.connect([&]() {
    Functions::ParsedFunction<dim>::declare_parameters(this->prm, dim + 1);
  });

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Reduction", "1.e-6");
    ParameterAcceptor::prm.set("Tolerance", "1.e-9");
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
}

// The real class of the ellitpic interface problem.
template <int dim>
class EllipticInterfaceDLM {
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
                          double rhs_value) const;

  void assemble();

  void solve();

  void output_results() const;

  void run();

 private:
  const ProblemParameters<dim> &parameters;
  Triangulation<dim> tria_bg;
  Triangulation<dim> tria_fg;

  std::shared_ptr<Mapping<dim>> mapping;

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
  SparseMatrix<Number> coupling_matrix;

  SparseMatrix<Number> mass_matrix_fg;
  SparseMatrix<Number> mass_matrix_bg;

  BlockVector<Number> system_rhs_block;
  BlockVector<Number> system_solution_block;
};

template <int dim>
EllipticInterfaceDLM<dim>::EllipticInterfaceDLM(
    const ProblemParameters<dim> &prm)
    : parameters(prm),
      dof_handler_bg(tria_bg),
      dof_handler_fg(tria_fg),
      fe_bg(parameters.background_space_finite_element_degree),
      fe_fg(parameters.immersed_space_finite_element_degree) {
  static_assert(dim == 2);
}

// Generate the grids for the background and immersed domains
template <int dim>
void EllipticInterfaceDLM<dim>::generate_grids() {
  try {
    GridGenerator::generate_from_name_and_arguments(
        tria_bg, parameters.name_of_background_grid,
        parameters.arguments_for_background_grid);
    tria_bg.refine_global(parameters.initial_background_refinement);
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << std::endl;
    std::cerr << "Error in background grid generation. Aborting." << std::endl;
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

  // Now check mesh sizes. We verify they are not too different by using safety
  // factors.
  const double h_background = GridTools::maximal_cell_diameter(tria_bg);
  const double h_immersed = GridTools::maximal_cell_diameter(tria_fg);
  const double ratio = h_immersed / h_background;
  std::cout << "h background = " << h_background << "\n"
            << "h immersed = " << h_immersed << "\n"
            << "grids ratio = " << ratio << std::endl;

  const double safety_factor = 1.2;
  AssertThrow(ratio < safety_factor && ratio > 0.7,
              ExcMessage("Check mesh sizes of the two grids."));

  // Do not dump grid to disk if meshes too large
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
  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);

  constraints_bg.clear();
  constraints_fg.clear();

  for (const types::boundary_id id : {0, 1, 2, 3}) {
    VectorTools::interpolate_boundary_values(
        dof_handler_bg, id, Functions::ZeroFunction<dim>(), constraints_bg);
  }

  constraints_bg.close();
  constraints_fg.close();

  setup_stiffness_matrix(dof_handler_bg, constraints_bg, stiffness_sparsity_bg,
                         stiffness_matrix_bg);

  setup_stiffness_matrix(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
                         stiffness_matrix_fg);

  mass_matrix_bg.reinit(stiffness_sparsity_bg);
  mass_matrix_fg.reinit(stiffness_sparsity_fg);

  system_rhs_block.reinit(3);
  system_rhs_block.block(0).reinit(dof_handler_bg.n_dofs());
  system_rhs_block.block(1).reinit(dof_handler_fg.n_dofs());
  system_rhs_block.block(2).reinit(dof_handler_fg.n_dofs());

  system_solution_block.reinit(system_rhs_block);

  std::cout << "N DoF background: " << dof_handler_bg.n_dofs() << std::endl;
  std::cout << "N DoF immersed: " << dof_handler_fg.n_dofs() << std::endl;

  std::cout << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
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

template <int dim>
void EllipticInterfaceDLM<dim>::setup_coupling() {
  QGauss<dim> quad(fe_bg.degree + 1);

  {
    DynamicSparsityPattern dsp(dof_handler_bg.n_dofs(),
                               dof_handler_fg.n_dofs());
    NonMatching::create_coupling_sparsity_pattern(
        dof_handler_bg, dof_handler_fg, quad, dsp, constraints_bg,
        ComponentMask(), ComponentMask());
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);

    NonMatching::create_coupling_mass_matrix(
        dof_handler_bg, dof_handler_fg, quad, coupling_matrix, constraints_bg,
        ComponentMask(), ComponentMask());
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::assemble_subsystem(
    const FiniteElement<dim> &fe, const DoFHandler<dim> &dof_handler,
    const AffineConstraints<double> &constraints,
    SparseMatrix<Number> &system_matrix, Vector<Number> &system_rhs, double rho,
    double mu, double rhs_value) const {
  const QGauss<dim> quad(fe.degree + 1);

  FEValues<dim> fe_values(fe, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
              (rho * fe_values.shape_value(i, q_index) *  // u
                   fe_values.shape_value(j, q_index)      // v
               + mu * fe_values.shape_grad(i, q_index) *  // grad u
                     fe_values.shape_grad(j, q_index)) *  // grad v
              fe_values.JxW(q_index);                     // dx

        cell_rhs(i) += (rhs_value *                          // f(x)
                        fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                        fe_values.JxW(q_index));             // dx
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::assemble() {
  // background stiffness matrix A
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, stiffness_matrix_bg,
                     system_rhs_block.block(0),
                     0.,                 // 0, no mass matrix
                     parameters.beta_1,  // only beta_1 (which is usually 1)
                     1.);                // rhs value, f1

  // immersed matrix A2 = (beta_2 - beta) (grad u,grad v)
  assemble_subsystem(
      fe_fg, dof_handler_fg, constraints_fg, stiffness_matrix_fg,
      system_rhs_block.block(1),
      0.,                                     // 0, no mass matrix
      parameters.beta_2 - parameters.beta_1,  // only jump beta_2 - beta
      3.);                                    // rhs value, f2

  // mass matrix background
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, mass_matrix_bg,
                     system_rhs_block.block(0), 1., 0., 0.);

  //  mass matrix immersed
  assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg, mass_matrix_fg,
                     system_rhs_block.block(2), 1., 0., 0.);
}

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

template <int dim>
void EllipticInterfaceDLM<dim>::solve() {
  // SparseDirectUMFPACK Af_inv_umfpack;
  // Af_inv_umfpack.initialize(stiffness_matrix_bg);  // Ainv
  // auto Af_inv = linear_operator(stiffness_matrix_bg, Af_inv_umfpack);

  SparseDirectUMFPACK M_inv_umfpack;
  M_inv_umfpack.initialize(mass_matrix_fg);  // inverse immersed mass matrix

  auto A_omega1 = linear_operator(stiffness_matrix_bg);
  auto A_omega2 = linear_operator(stiffness_matrix_fg);
  auto M = linear_operator(mass_matrix_fg);
  auto NullF = null_operator(linear_operator(stiffness_matrix_bg));
  auto NullS = null_operator(linear_operator(mass_matrix_fg));
  auto NullCouplin = null_operator(linear_operator(coupling_matrix));
  auto Ct = linear_operator(coupling_matrix);
  auto C = transpose_operator(Ct);

  // SolverControl control_lagrangian(40000, 1e-4, false, true);
  SolverCG<BlockVector<double>> solver_lagrangian(
      parameters.inner_solver_control);

  auto invW1 = linear_operator(mass_matrix_fg, M_inv_umfpack);
  auto invW = invW1 * invW1;  // W = M^{-2}

  // Using inverse of diagonal mass matrix (squared)

  // Vector<double> inverse_diag_mass_squared(dof_handler_fg.n_dofs());
  // for (unsigned int i = 0; i < dof_handler_fg.n_dofs(); ++i)
  //   inverse_diag_mass_squared[i] =
  //       1. / (mass_matrix_fg.diag_element(i) *
  //       mass_matrix_fg.diag_element(i));

  // DiagonalMatrix<Vector<double>> diag_inverse(inverse_diag_mass_squared);
  // auto invW = linear_operator(diag_inverse);

  // Define augmented blocks. Notice that A22_aug is actually A_omega2 +
  // gamma_AL * Id
  auto A11_aug = A_omega1 + parameters.gamma_AL * Ct * invW * C;
  auto A22_aug = A_omega2 + parameters.gamma_AL * M * invW * M;
  auto A12_aug = -parameters.gamma_AL * Ct * invW * M;
  auto A21_aug = transpose_operator(A12_aug);

  auto system_operator = block_operator<3, 3, BlockVector<double>>(
      {{{{A11_aug, A12_aug, Ct}},
        {{A21_aug, A22_aug, -1. * M}},
        {{C, -1. * M, NullCouplin}}}});  // augmented the 2x2 top left block!

  SolverFGMRES<BlockVector<Number>> solver_fgmres(
      parameters.outer_solver_control);

  TrilinosWrappers::PreconditionAMG amg_prec_A11;
  amg_prec_A11.initialize(stiffness_matrix_bg);
  auto AMG_A1 = linear_operator(stiffness_matrix_bg, amg_prec_A11);
  TrilinosWrappers::PreconditionAMG amg_prec_A22;
  amg_prec_A22.initialize(stiffness_matrix_fg);
  auto AMG_A2 = linear_operator(stiffness_matrix_fg, amg_prec_A22);

  if (parameters.use_modified_AL_preconditioner) {
    // If we use the modified AL preconditioner, we check if we use a small
    // value of gamma.
    AssertThrow(
        parameters.gamma_AL < 1.,
        ExcMessage("gamma_AL is too large for modified AL preconditioner."));

    SolverCG<Vector<double>> solver_lagrangian_scalar(
        parameters.inner_solver_control);
    // Define block preconditioner using AL approach
    auto A11_aug_inv =
        inverse_operator(A11_aug, solver_lagrangian_scalar, AMG_A1);
    auto A22_aug_inv =
        inverse_operator(A22_aug, solver_lagrangian_scalar, AMG_A2);

    EllipticInterfacePreconditioners::BlockTriangularALPreconditionerModified
        preconditioner_AL(C, M, invW, parameters.gamma_AL, A11_aug_inv,
                          A22_aug_inv);
    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, preconditioner_AL);
  } else {
    // Check that gamma is not too small
    AssertThrow(parameters.gamma_AL > 1.,
                ExcMessage("Parameter gamma is probably small for classical AL "
                           "preconditioner."));

    // Define preconditioner for the augmented block
    auto prec_aug = block_operator<2, 2, BlockVector<double>>(
        {{{{AMG_A1, NullF}}, {{NullS, AMG_A2}}}});
    PreconditionIdentity prec_id;
    auto Aug = block_operator<2, 2, BlockVector<double>>(
        {{{{A11_aug, A12_aug}},
          {{A21_aug, A22_aug}}}});  // augmented block to be inverted
    auto Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_aug);

    EllipticInterfacePreconditioners::BlockTriangularALPreconditioner
        preconditioner_AL(Aug_inv, C, M, invW, parameters.gamma_AL);
    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, preconditioner_AL);
  }

  system_rhs_block.block(2) = 0;  // last row of the rhs is 0

  std::cout << "Solved in " << parameters.outer_solver_control.last_step()
            << " iterations"
            << (parameters.outer_solver_control.last_step() < 10 ? "  " : " ")
            << "\n";

  // Do some sanity checks: Check the constraints residual and the condition
  // number of CCt.
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
    std::cerr << "***BBt solve not successfull (see condition number above)***"
              << std::endl;
    AssertThrow(false, ExcMessage("BBt does not have full rank."));
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::output_results() const {
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_fg);
    data_out.add_data_vector(system_solution_block.block(1), "u");
    data_out.add_data_vector(system_solution_block.block(2), "lambda");
    data_out.build_patches();
    std::ofstream output("solution-fg.vtu");
    data_out.write_vtu(output);
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_bg);
    data_out.add_data_vector(system_solution_block.block(0), "u");
    data_out.build_patches();
    std::ofstream output("solution-bg.vtu");
    data_out.write_vtu(output);
  }
}

template <int dim>
void EllipticInterfaceDLM<dim>::run() {
  for (unsigned int ref_cycle = 0; ref_cycle < parameters.n_refinement_cycles;
       ++ref_cycle) {
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
    std::cout << "Refinement cycle: " << ref_cycle << std::endl;
    std::cout << "gamma_AL= " << parameters.gamma_AL << std::endl;
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
    if (parameters.use_modified_AL_preconditioner && parameters.use_sqrt_2_rule)
      parameters.gamma_AL /= std::sqrt(2.);  // using sqrt(2)-rule from
                                             // modified-AL paper.
    output_results();
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
      parameter_file = "parameters_elliptic_interface.prm";
    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

    EllipticInterfaceDLM<dim> solver(parameters);
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