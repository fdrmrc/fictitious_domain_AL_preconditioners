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

static constexpr double beta_2 = 10;

template <int dim, int fe_degree>
class ImmersedLaplaceSolver {
 public:
  typedef double LevelNumber;
  typedef double Number;

  const constexpr static unsigned int degree_u = fe_degree;
  const constexpr static unsigned int degree_p = fe_degree;

  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef LinearAlgebra::distributed::Vector<LevelNumber> LevelVectorType;

  class AdditionalData;

  ImmersedLaplaceSolver(const unsigned int &n_refinements = 2);

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
  BlockVector<Number> system_solution_block_reduced;
};

template <int dim, int fe_degree>
ImmersedLaplaceSolver<dim, fe_degree>::ImmersedLaplaceSolver(
    const unsigned int &n_refinements)
    : fe_bg(fe_degree),
      fe_fg(fe_degree),
      dof_handler_bg(tria_bg),
      dof_handler_fg(tria_fg) {
  static_assert(dim == 2);

  GridGenerator::hyper_cube(tria_bg, -1., 1., true);
  // GridGenerator::hyper_cube(tria_fg, -0.44, .44, true);
  GridGenerator::hyper_ball(tria_fg, Point<dim>(), 0.5, true);

  tria_bg.refine_global(n_refinements + 3);
  // tria_bg.refine_global(n_refinements + 2);
  tria_fg.refine_global(n_refinements);

  std::ofstream bg_out("background.vtu");
  std::ofstream fg_out("foreground.vtu");
  GridOut grid_out;
  grid_out.write_vtu(tria_bg, bg_out);
  grid_out.write_vtu(tria_fg, fg_out);
}

template <int dim, int fe_degree>
void ImmersedLaplaceSolver<dim, fe_degree>::system_setup() {
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

  std::cout << " N DoF background: " << dof_handler_bg.n_dofs() << std::endl;
  std::cout << " N DoF foreground: " << dof_handler_fg.n_dofs() << std::endl;
}

template <int dim, int fe_degree>
void ImmersedLaplaceSolver<dim, fe_degree>::setup_stiffnesss(
    const DoFHandler<dim> &dof_handler, AffineConstraints<double> &constraints,
    SparsityPattern &stiffness_sparsity,
    SparseMatrix<Number> &stiffness_matrix) const {
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  stiffness_sparsity.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity);
}

template <int dim, int fe_degree>
void ImmersedLaplaceSolver<dim, fe_degree>::setup_coupling() {
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

  if (false) {
    DynamicSparsityPattern dsp(dof_handler_fg.n_dofs(),
                               dof_handler_fg.n_dofs());

    NonMatching::create_coupling_sparsity_pattern(
        dof_handler_fg, dof_handler_fg, quad, dsp, AffineConstraints<double>(),
        ComponentMask(), ComponentMask());
    mass_sparsity_fg.copy_from(dsp);
    mass_matrix_fg.reinit(mass_sparsity_fg);

    NonMatching::create_coupling_mass_matrix(
        dof_handler_fg, dof_handler_fg, quad, mass_matrix_fg,
        AffineConstraints<double>(), ComponentMask(), ComponentMask());
  }
}

template <int dim, int fe_degree>
void ImmersedLaplaceSolver<dim, fe_degree>::assemble_subsystem(
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
void ImmersedLaplaceSolver<dim, fe_degree>::assemble() {
  const double rho_f = 0;
  const double mu_f = 1;  // beta

  const double rho_s = 0;      // 10;
  const double mu_s = beta_2;  // beta_2

  // background stiffness matrix A
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, stiffness_matrix_bg,
                     system_rhs_block.block(0),
                     rho_f,  // 0, no mass matrix
                     mu_f,   // only beta
                     1.);    // rhs value, f1

  // immersed matrix A2 = (beta_2 - beta) (grad u,grad v)
  assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg, stiffness_matrix_fg,
                     system_rhs_block.block(1),
                     0.,           // 0, no mass matrix
                     mu_s - mu_f,  // only jump beta_2 - beta
                     3.);          // rhs value, f2

  // mass matrix background
  assemble_subsystem(fe_bg, dof_handler_bg, constraints_bg, mass_matrix_bg,
                     system_rhs_block.block(0), 1., 0., 0.);

  //  mass matrix immersed
  assemble_subsystem(fe_fg, dof_handler_fg, constraints_fg, mass_matrix_fg,
                     system_rhs_block.block(2), 1., 0., 0.);
}

template <typename BLOCK_VECTOR>
class BlockOperator {
 public:
  typedef BLOCK_VECTOR BlockVectorType;
  typedef typename BlockVectorType::BlockType VectorType;
  typedef LinearOperator<VectorType> BlockType;

  BlockOperator() = default;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    Af.vmult(dst.block(0), src.block(0));
    Ct.vmult_add(dst.block(0), src.block(2));

    As.vmult(dst.block(1), src.block(1));
    Mt.vmult_add(dst.block(1), src.block(2));

    C.vmult(dst.block(2), src.block(0));
    M.vmult_add(dst.block(2), src.block(1));
    S.vmult_add(dst.block(2), src.block(2));
  }

  template <class AfType, class AsType, class CType, class MType, class SType>
  void initialize(const AfType &block_Af, const AsType &block_As,
                  const CType &block_CT, const MType &block_M,
                  const SType &block_S) {
    Af = linear_operator<VectorType>(LinearOperator<VectorType>(),
                                     Utilities::get_underlying_value(block_Af));
    As = linear_operator<VectorType>(LinearOperator<VectorType>(),
                                     Utilities::get_underlying_value(block_As));
    Ct = linear_operator<VectorType>(LinearOperator<VectorType>(),
                                     Utilities::get_underlying_value(block_CT));
    M = -1 *
        linear_operator<VectorType>(LinearOperator<VectorType>(),
                                    Utilities::get_underlying_value(block_M));

    S = linear_operator<VectorType>(LinearOperator<VectorType>(),
                                    Utilities::get_underlying_value(block_S));

    C = transpose_operator(Ct);
    Mt = transpose_operator(M);
  }

 private:
  BlockType Af;
  BlockType As;
  BlockType C;
  BlockType Ct;
  BlockType M;
  BlockType Mt;
  BlockType S;
};

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

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
    std::cout << "Done first" << std::endl;

    // Now, use the action of the Aug_inv on the first two components of the
    // solution
    BlockVector<double> uu2, result;

    uu2.reinit(2);
    uu2.block(0).reinit(u.block(0));
    uu2.block(0) = u.block(0) - Ct * v.block(2);
    std::cout << "Done second" << std::endl;

    uu2.block(1).reinit(u.block(1));
    uu2.block(1) = u.block(1) + 1. * M * v.block(2);
    std::cout << "Done third" << std::endl;

    result.reinit(2);
    result.block(0).reinit(u.block(0));
    result.block(1).reinit(u.block(1));

    result = Aug_inv * uu2;
    std::cout << "Done Aug inv" << std::endl;
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

template <int dim, int fe_degree>
void ImmersedLaplaceSolver<dim, fe_degree>::solve() {
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

  SolverControl solver_control(10000, 1e-9, true, true);
  PrimitiveVectorMemory<BlockVector<Number>> mem;

  SolverControl control_lagrangian(40000, 1e-2, false, true);
  SolverCG<BlockVector<double>> solver_lagrangian(control_lagrangian);

  const double gamma_AL = 10.;  // gamma
  auto invW1 = linear_operator(mass_matrix_fg, M_inv_umfpack);
  auto invW = invW1 * invW1;  // W = M^{-2}

  // Define augmented blocks
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
  auto Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_aug);

  // Define block preconditioner using AL approach
  BlockTriangularPreconditionerAL preconditioner_AL(Aug_inv, C, M, invW,
                                                    gamma_AL);

  SolverFGMRES<BlockVector<Number>> solver_fgmres(solver_control);

  system_rhs_block.block(2) = 0;  // last row of the rhs is 0
  solver_fgmres.solve(system_operator, system_solution_block, system_rhs_block,
                      preconditioner_AL);

  std::cout << "Solved in " << solver_control.last_step() << " iterations"
            << (solver_control.last_step() < 10 ? "  " : " ") << "\n";

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
  std::cout << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
  std::cout << "Estimate condition number of BBt using CG" << std::endl;
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
void ImmersedLaplaceSolver<dim, fe_degree>::output_results() const {
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

  ImmersedLaplaceSolver<dim, degree> solver(n_refinements);
  solver.system_setup();
  solver.setup_coupling();
  solver.assemble();
  solver.solve();
  solver.output_results();
}