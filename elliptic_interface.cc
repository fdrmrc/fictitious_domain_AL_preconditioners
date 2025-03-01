#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
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

#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

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
  unsigned int n_cycles = 5;

  // Coefficients determining the size of the jump. beta_1 is usually kept at 1.
  double beta_1 = 1.;

  // beta_2, instead, is the one which is changed.
  double beta_2 = 10.;

  std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

  unsigned int background_space_finite_element_degree = 1;

  unsigned int immersed_space_finite_element_degree = 1;

  unsigned int coupling_quadrature_order = 3;

  unsigned int verbosity_level = 10;

  std::string outer_solver = "FGMRES";

  bool use_diagonal = false;

  bool use_sqrt_2_rule = false;

  // AL parameter. Its magnitude depends on which AL preconditioner (original
  // vs. modified AL) is chosen. We define it as mutable since with modified AL
  // its value may change upon mesh refinement.
  mutable double gamma_AL = 10.;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<dim>>
      rhs;  //(f,f_2,0)
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
    : ParameterAcceptor("Elliptic Interface Problem/"),
      rhs("Right hand side", dim + 1) {
  add_parameter("Background FE degree", background_space_finite_element_degree,
                "", this->prm, Patterns::Integer(1));

  add_parameter("Immersed FE degree", immersed_space_finite_element_degree, "",
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
    add_parameter("Refinemented cycles", n_cycles,
                  "Number of refinement cycles to perform. Useful if you want "
                  "to study convergence.");
  }
  leave_subsection();

  enter_subsection("AL preconditioner");
  {
    add_parameter("Outer solver", outer_solver);
    add_parameter("Use diagonal", use_diagonal,
                  "Use diagonal approximation of mass matrices.");
    add_parameter("Verbosity level", verbosity_level);
  }
  leave_subsection();

  // Parameters for the modified AL
  enter_subsection("Modified AL preconditioner");
  {
    add_parameter("Outer solver", outer_solver);
    add_parameter("Use diagonal", use_diagonal,
                  "Use diagonal approximation of mass matrices.");
    add_parameter("Use sqrt(2)-rule for gamma", use_sqrt_2_rule);
    add_parameter("gamma", gamma_AL);
    add_parameter("Verbosity level", verbosity_level);
  }
  leave_subsection();

  rhs.declare_parameters_call_back.connect([&]() {
    Functions::ParsedFunction<dim>::declare_parameters(this->prm, dim + 1);
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

  // Old manual way. TODO: remove once you have it working
  // GridGenerator::hyper_cube(tria_bg, -1., 1., true);
  // GridGenerator::hyper_cube(tria_fg, -0.14, .44, true);
  // // GridGenerator::hyper_ball(tria_fg, Point<dim>(), 0.5, true);

  // // tria_bg.refine_global(n_refinements + 3);
  // tria_bg.refine_global(n_refinements + 2);
  // tria_fg.refine_global(n_refinements);

  // // Check mesh sizes. We try to verify they are not too differen by using
  // // suitable safety factors.
  // const double h_background = GridTools::maximal_cell_diameter(tria_bg);
  // const double h_immersed = GridTools::maximal_cell_diameter(tria_fg);
  // const double ratio = h_immersed / h_background;
  // std::cout << "h background = " << h_background << "\n"
  //           << "h immersed = " << h_immersed << "\n"
  //           << "grids ratio = " << ratio << std::endl;

  // const double safety_factor = 1.2;
  // AssertThrow((ratio) < safety_factor || (ratio) > 0.7,
  //             ExcMessage("Check mesh sizes of the two grids."));

  // // Do not dump grid to disk if meshes too large
  // if (tria_bg.n_active_cells() < 1e3) {
  //   std::ofstream bg_out("background_grid.vtk");
  //   std::ofstream fg_out("immersed_grid.vtk");
  //   GridOut grid_out;
  //   grid_out.write_vtk(tria_bg, bg_out);
  //   grid_out.write_vtk(tria_fg, fg_out);
  // }
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

// Original AL preconditioner
class BlockTriangularPreconditionerAL {
 public:
  BlockTriangularPreconditionerAL(
      const LinearOperator<BlockVector<double>> Aug_inv_,
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> M_,
      const LinearOperator<Vector<double>> invW_, const double gamma_) {
    Aug_inv = Aug_inv_;
    C = C_;
    Ct = transpose_operator(C);
    M = M_;
    invW = invW_;
    gamma = gamma_;
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const {
    v.block(0) = 0.;
    v.block(1) = 0.;
    v.block(2) = 0.;

    v.block(2) = -gamma * invW * u.block(2);

    // Now, use the action of the Aug_inv on the first two components of the
    // solution
    BlockVector<double> uu2, result;

    uu2.reinit(2);
    uu2.block(0).reinit(u.block(0));
    uu2.block(0) = u.block(0) - Ct * v.block(2);

    uu2.block(1).reinit(u.block(1));
    uu2.block(1) = u.block(1) + 1. * M * v.block(2);

    result.reinit(2);
    result.block(0).reinit(u.block(0));
    result.block(1).reinit(u.block(1));

    result = Aug_inv * uu2;
    // Copy solutions back to the output of this routine
    v.block(0) = result.block(0);
    v.block(1) = result.block(1);
  }

  LinearOperator<BlockVector<double>> Aug_inv;
  LinearOperator<Vector<double>> C;
  LinearOperator<Vector<double>> Ct;
  LinearOperator<Vector<double>> M;
  LinearOperator<Vector<double>> invW;
  double gamma;
};

// Implementation of the modified AL preconditioner. Notice how we need the
// inverse of the diagonal blocks.
class BlockTriangularPreconditionerALModified {
 public:
  BlockTriangularPreconditionerALModified(
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> M_,
      const LinearOperator<Vector<double>> invW_, const double gamma_,
      const LinearOperator<Vector<double>> A11_inv_,
      const LinearOperator<Vector<double>> A22_inv_) {
    A11_inv = A11_inv_;
    A22_inv = A22_inv_;
    C = C_;
    Ct = transpose_operator(C);
    M = M_;
    invW = invW_;
    gamma = gamma_;
  }

  template <typename BlockVectorType>
  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    // Assert that the block vectors have the right number of blocks
    Assert(src.n_blocks() == 3, ExcDimensionMismatch(src.n_blocks(), 3));
    Assert(dst.n_blocks() == 3, ExcDimensionMismatch(dst.n_blocks(), 3));

    // Extract the blocks from the source vector
    const auto &u = src.block(0);
    const auto &u2 = src.block(1);
    const auto &lambda = src.block(2);

    // Create temporary vectors for intermediate results
    typename BlockVectorType::BlockType temp(lambda.size());

    // 1. Compute the third block first (1st inverse application: invW)
    dst.block(2) = invW * lambda;
    dst.block(2) *= -gamma;

    // Prepare for the second block calculation
    temp = M * (-1.0 / gamma * dst.block(2));  // M * invW * lambda

    // 2. Compute the second block (2nd inverse application: A22_inv)
    // A22_inv * (u2 - gamma * M * invW * lambda)
    temp *= -gamma;
    temp += u2;
    dst.block(1) = A22_inv * temp;

    // Prepare for the first block calculation
    auto B_T = -gamma * Ct * invW * M;
    temp = B_T * dst.block(1);  // B_T * A22_inv * (...)
    temp *= -1.0;
    temp += u;

    // Add the term with Ct
    auto C_T_term = Ct * (-1.0 / gamma * dst.block(2));  // Ct * invW * lambda
    temp += gamma * C_T_term;

    // 3. Compute the first block (3rd inverse application: A11_inv)
    dst.block(0) = A11_inv * temp;
  }

  LinearOperator<Vector<double>> A11_inv;
  LinearOperator<Vector<double>> A22_inv;
  LinearOperator<Vector<double>> C;
  LinearOperator<Vector<double>> Ct;
  LinearOperator<Vector<double>> M;
  LinearOperator<Vector<double>> invW;
  double gamma;
};

template <int dim>
void EllipticInterfaceDLM<dim>::solve() {
  SparseDirectUMFPACK Af_inv_umfpack;
  Af_inv_umfpack.initialize(stiffness_matrix_bg);  // Ainv

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
  auto Af_inv = linear_operator(stiffness_matrix_bg, Af_inv_umfpack);

  SolverControl control_lagrangian(40000, 1e-4, false, true);
  SolverCG<BlockVector<double>> solver_lagrangian(control_lagrangian);

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

  auto Aug = block_operator<2, 2, BlockVector<double>>(
      {{{{A11_aug, A12_aug}},
        {{A21_aug, A22_aug}}}});  // augmented block to be inverted

  // Define preconditioner for the augmented block
  TrilinosWrappers::PreconditionAMG amg_prec_A11;
  amg_prec_A11.initialize(stiffness_matrix_bg);
  auto AMG_A1 = linear_operator(stiffness_matrix_bg, amg_prec_A11);
  TrilinosWrappers::PreconditionAMG amg_prec_A22;
  amg_prec_A22.initialize(stiffness_matrix_fg);
  auto AMG_A2 = linear_operator(stiffness_matrix_fg, amg_prec_A22);

  auto prec_aug = block_operator<2, 2, BlockVector<double>>(
      {{{{AMG_A1, NullF}}, {{NullS, AMG_A2}}}});
  PreconditionIdentity prec_id;
  auto Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_id);

  SolverCG<Vector<double>> solver_lagrangian_scalar(control_lagrangian);
  // Define block preconditioner using AL approach
  auto A11_aug_inv =
      inverse_operator(A11_aug, solver_lagrangian_scalar, AMG_A1);
  auto A22_aug_inv =
      inverse_operator(A22_aug, solver_lagrangian_scalar, AMG_A2);

  BlockTriangularPreconditionerALModified preconditioner_AL(
      C, M, invW, parameters.gamma_AL, A11_aug_inv, A22_aug_inv);
  // BlockTriangularPreconditionerAL preconditioner_AL(Aug_inv, C, M, invW,
  //                                                   gamma_AL);

  SolverControl outer_control(10000, 1e-6, true, true);
  SolverFGMRES<BlockVector<Number>> solver_fgmres(outer_control);

  system_rhs_block.block(2) = 0;  // last row of the rhs is 0
  solver_fgmres.solve(system_operator, system_solution_block, system_rhs_block,
                      preconditioner_AL);

  std::cout << "Solved in " << outer_control.last_step() << " iterations"
            << (outer_control.last_step() < 10 ? "  " : " ") << "\n";

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
  for (unsigned int ref_cycle = 0; ref_cycle < parameters.n_cycles;
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
    if (parameters.use_sqrt_2_rule)
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