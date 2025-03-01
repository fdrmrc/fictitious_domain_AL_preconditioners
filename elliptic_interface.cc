#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
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

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace dealii;

static constexpr double beta_2 = 10.;

template <int dim, int fe_degree>
class EllipticInterfaceDLM {
 public:
  typedef double Number;

  const constexpr static unsigned int degree_u = fe_degree;
  const constexpr static unsigned int degree_p = fe_degree;

  EllipticInterfaceDLM(const unsigned int &n_refinements = 2,
                       const double gamma_AL = 1.);

  void system_setup();

  void setup_stiffnesss(const DoFHandler<dim> &dof_handler,
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

 private:
  Triangulation<dim> tria_bg;
  Triangulation<dim> tria_fg;
  FE_Q<dim> fe_bg;
  FE_Q<dim> fe_fg;
  std::shared_ptr<Mapping<dim>> mapping;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;

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

  double gamma_AL;
};

template <int dim, int fe_degree>
EllipticInterfaceDLM<dim, fe_degree>::EllipticInterfaceDLM(
    const unsigned int &n_refinements, const double gamma_AL_)
    : fe_bg(fe_degree),
      fe_fg(fe_degree),
      dof_handler_bg(tria_bg),
      dof_handler_fg(tria_fg),
      gamma_AL(gamma_AL_) {
  static_assert(dim == 2);

  GridGenerator::hyper_cube(tria_bg, -1., 1., true);
  GridGenerator::hyper_cube(tria_fg, -0.14, .44, true);
  // GridGenerator::hyper_ball(tria_fg, Point<dim>(), 0.5, true);

  // tria_bg.refine_global(n_refinements + 3);
  tria_bg.refine_global(n_refinements + 2);
  tria_fg.refine_global(n_refinements);

  // Check mesh sizes. We try to verify they are not too differen by using
  // suitable safety factors.
  const double h_background = GridTools::maximal_cell_diameter(tria_bg);
  const double h_immersed = GridTools::maximal_cell_diameter(tria_fg);
  const double ratio = h_immersed / h_background;
  std::cout << "h background = " << h_background << "\n"
            << "h immersed = " << h_immersed << "\n"
            << "grids ratio = " << ratio << std::endl;

  const double safety_factor = 1.2;
  AssertThrow((ratio) < safety_factor || (ratio) > 0.7,
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

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::system_setup() {
  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);

  for (const types::boundary_id id : {0, 1, 2, 3}) {
    VectorTools::interpolate_boundary_values(
        dof_handler_bg, id, Functions::ZeroFunction<dim>(), constraints_bg);
  }

  constraints_bg.close();
  constraints_fg.close();

  setup_stiffnesss(dof_handler_bg, constraints_bg, stiffness_sparsity_bg,
                   stiffness_matrix_bg);

  setup_stiffnesss(dof_handler_fg, constraints_fg, stiffness_sparsity_fg,
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

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::setup_stiffnesss(
    const DoFHandler<dim> &dof_handler, AffineConstraints<double> &constraints,
    SparsityPattern &stiffness_sparsity,
    SparseMatrix<Number> &stiffness_matrix) const {
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  stiffness_sparsity.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity);
}

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::setup_coupling() {
  QGauss<dim> quad(fe_degree + 1);

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

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::assemble_subsystem(
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

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::assemble() {
  const double beta_1 = 1;  // beta

  // const double beta_2 = beta_2;  // beta_2

  // background stiffness matrix A
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, stiffness_matrix_bg,
                     system_rhs_block.block(0),
                     0.,      // 0, no mass matrix
                     beta_1,  // only beta_1
                     1.);     // rhs value, f1

  // immersed matrix A2 = (beta_2 - beta) (grad u,grad v)
  assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg, stiffness_matrix_fg,
                     system_rhs_block.block(1),
                     0.,               // 0, no mass matrix
                     beta_2 - beta_1,  // only jump beta_2 - beta
                     3.);              // rhs value, f2

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

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::solve() {
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

  // const double gamma_AL = 1e-2;  // gamma
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
  auto A11_aug = A_omega1 + gamma_AL * Ct * invW * C;
  auto A22_aug = A_omega2 + gamma_AL * M * invW * M;
  auto A12_aug = -gamma_AL * Ct * invW * M;
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
      C, M, invW, gamma_AL, A11_aug_inv, A22_aug_inv);
  // BlockTriangularPreconditionerAL preconditioner_AL(Aug_inv, C, M, invW,
  //                                                   gamma_AL);

  SolverControl outer_control(10000, 1e-6, true, true);
  SolverFGMRES<BlockVector<Number>> solver_fgmres(outer_control);

  system_rhs_block.block(2) = 0;  // last row of the rhs is 0
  solver_fgmres.solve(system_operator, system_solution_block, system_rhs_block,
                      preconditioner_AL);

  std::cout << "Solved in " << outer_control.last_step() << " iterations"
            << (outer_control.last_step() < 10 ? "  " : " ") << "\n";

  // Do a sanity check
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

template <int dim, int fe_degree>
void EllipticInterfaceDLM<dim, fe_degree>::output_results() const {
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

int main(int argc, char *argv[]) {
  const int dim = 2;
  const int degree = 1;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  deallog.depth_console(10);

  const unsigned int n_refinements =
      argc == 1 ? 2 : std::strtol(argv[1], NULL, 10);

  double gamma_AL = 1e-2;  // gamma_0, determined with a coarse solve.

  // EllipticInterfaceDLM<dim, degree> solver(n_refinements, gamma_AL);
  // solver.system_setup();
  // solver.setup_coupling();
  // solver.assemble();
  // solver.solve();
  // solver.output_results();

  for (unsigned int ref_cycle = n_refinements; ref_cycle < n_refinements + 8;
       ++ref_cycle) {
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
    std::cout << "Refinement cycle: " << ref_cycle << std::endl;
    gamma_AL /= std::sqrt(2.);  // using sqrt(2)-rule from modified-AL paper.
    std::cout << "gamma_AL= " << gamma_AL << std::endl;
    EllipticInterfaceDLM<dim, degree> solver(ref_cycle, gamma_AL);
    solver.system_setup();
    solver.setup_coupling();
    solver.assemble();
    solver.solve();
    solver.output_results();
  }
}