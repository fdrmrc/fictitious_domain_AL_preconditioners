// Parallel MPI version of the elliptic interface fictitious-domain problem
// with augmented-Lagrangian (AL) preconditioner. Only the "operator" version
// of the AL preconditioner is used: the augmented (1,1)-block is built at the
// variational level by adding contributions through particles. Both the ideal
// and the modified variants of the AL preconditioner are supported.
//
// Linear-algebra backend: Trilinos by default. Define USE_PETSC_LA at compile
// time (-DUSE_PETSC_LA) to switch to PETSc.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_bubbles.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "augmented_lagrangian_preconditioner.h"
#include "utilities.h"

namespace LA {
#if defined(USE_PETSC_LA) && defined(DEAL_II_WITH_PETSC)
using namespace dealii::LinearAlgebraPETSc;
#else
using namespace dealii::LinearAlgebraTrilinos;
#endif
} // namespace LA

using namespace dealii;

namespace ALParallel {

// CG-based action of M^{-1} on a vector, preconditioned by `Prec`.
// Replaces the previous diagonal approximation of the immersed mass inverse.
template <typename VectorType, typename SparseMatrixType, typename Prec>
struct InvMassOp {
  const SparseMatrixType &M;
  Prec &prec;
  mutable SolverControl control;
  IndexSet owned;
  MPI_Comm comm;

  InvMassOp(const SparseMatrixType &M_, Prec &p_, unsigned int max_it,
            double tol, IndexSet owned_, MPI_Comm comm_)
      : M(M_), prec(p_), control(max_it, tol, false, false), owned(owned_),
        comm(comm_) {}

  void vmult(VectorType &dst, const VectorType &src) const {
    dst = 0.;
    SolverCG<VectorType> cg(control);
    cg.solve(M, dst, src, prec);
  }
};

// All operators here apply matrix-vector products directly to the underlying
// parallel sparse matrices, on temporary vectors with explicit IndexSets.
//
// The augmented system with directly assembled coupling blocks. Blocks 0/1
// live on the bg/immersed-primal spaces; block 2 lives on the multiplier
// space. With a non-trivial FE triple the multiplier space may differ from
// the immersed primal: in that case `M` is the rectangular cross-mass
// (n_imm x n_mult), `Ct` is n_bg x n_mult, and `W` (used by `InvWOp`) is
// the multiplier-multiplier mass (n_mult x n_mult). For Q1-Q1-Q1 the
// multiplier coincides with the primal and these objects collapse to the
// (square, symmetric) immersed mass matrix used previously.
//
//   (0,0) = A_bg                            (AL term already added in place)
//   (0,1) = -gamma1 * coupling_matrix_01   (u|_Omega_2, v_2)
//   (0,2) = Ct                              (u|_Omega_2, mu)
//   (1,0) = -gamma2 * coupling_matrix_10   transpose of (0,1)
//   (1,1) = A_fg_plus_scaled_M
//   (1,2) = -M                              (u_2, mu)
//   (2,0) = Ct^T
//   (2,1) = -M^T                            (lambda, v_2)
//   (2,2) = 0
// where invW = W^{-1}.
//
// (Q1-Bubble-Q0): to keep the AL augmentation *consistent*, i.e.
// equal to the algebraic expesssion gamma_1 * Ct W^{-1} (Ct^T u_1 - M^T u_2) in
// row 0 and gamma_2 * M W^{-1} (M^T u_2 - Ct^T u_1) in row 1, which vanish
// identically on the discrete constraint manifold (row 2). The matrices named
// above hold the PROJECTED quantities in that case:
//   * coupling_matrix_01 = Ct W^{-1} M^T    (n_bg x n_imm)
//   * coupling_matrix_10 = M  W^{-1} Ct^T   (n_imm x n_bg)
//   * A_fg_plus_scaled_M = A_fg + gamma2 * M W^{-1} M^T
// Since W is diagonal for a cellwise-constant multiplier, each of these is a
// per-cell outer product of the integral vectors c_K[i] = int_K phi_i^bg and
// m_K[j] = int_K phi_j^imm: no global triple product is ever formed.

template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp>
struct System3x3 {
  const SparseMatrixType &A_bg; // already AL-augmented in place
  const SparseMatrixType
      &A_fg_plus_scaled_M;    // (beta_2 - beta_1)*K_fg + gamma2*M_fg
  const SparseMatrixType &M;  // immersed mass matrix
  const SparseMatrixType &Ct; // C: n_bg x n_mult coupling with multiplier
  const SparseMatrixType
      &coupling_matrix_01; // (u|_Omega_2, v_2) - n_bg x n_imm
  const SparseMatrixType
      &coupling_matrix_10; // (u_2, v|_Omega_2) - n_imm x n_bg (transpose)
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg, owned_mult;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm);
    VectorType tm(owned_mult, comm);
    VectorType tb(owned_bg, comm);

    // Row 0: A_bg * x0  - gamma1 * coupling_matrix_01 * x1  + Ct * x2
    A_bg.vmult(dst.block(0), src.block(0));
    coupling_matrix_01.vmult(tb, src.block(1));
    dst.block(0).add(-gamma1, tb);
    Ct.vmult(tb, src.block(2));
    dst.block(0) += tb;

    // Row 1: -gamma2 * coupling_matrix_10 * x0
    //        + A_fg_plus_scaled_M * x1
    //        - M * x2          (M is n_imm x n_mult)
    coupling_matrix_10.vmult(ti, src.block(0));
    ti *= -gamma2;
    dst.block(1) = ti;
    A_fg_plus_scaled_M.vmult_add(dst.block(1), src.block(1));
    M.vmult(ti, src.block(2));
    dst.block(1) -= ti;

    // Row 2: Ct^T * x0 - M^T * x1   (result lives in multiplier space)
    Ct.Tvmult(dst.block(2), src.block(0));
    M.Tvmult(tm, src.block(1));
    dst.block(2) -= tm;
  }
};

// 2x2 augmented block used by the ideal AL preconditioner (rows 0,1 of the
// system above, with x2 = 0).
template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp>
struct Block2x2 {
  const SparseMatrixType &A_bg;
  const SparseMatrixType &A_fg_plus_scaled_M;
  const SparseMatrixType &M;
  const SparseMatrixType &coupling_matrix_01;
  const InvWOp &invW;
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm), ti2(owned_fg, comm);
    VectorType tb(owned_bg, comm);

    A_bg.vmult(dst.block(0), src.block(0));
    coupling_matrix_01.vmult(tb, src.block(1));
    dst.block(0).add(-gamma1, tb);

    coupling_matrix_01.Tvmult(ti, src.block(0));
    ti *= -gamma2;
    dst.block(1) = ti;
    A_fg_plus_scaled_M.vmult_add(dst.block(1), src.block(1));
  }
};

template <typename BlockVectorType, typename Prec0, typename Prec1>
struct BlockDiagPrec {
  const Prec0 &p0;
  const Prec1 &p1;
  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    p0.vmult(dst.block(0), src.block(0));
    p1.vmult(dst.block(1), src.block(1));
  }
};

// Modified AL block-triangular preconditioner.
//   lam = -gamma * invW * r2
//   u2  =  A22_inv ( r1 + M * lam )
//   u1  =  A11_inv ( r0 + gamma*Ct*invW*M*u2 - Ct*lam )
template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp, typename A11InvOp,
          typename A22InvOp>
struct ModifiedALPreconditioner {
  const SparseMatrixType &M;
  const SparseMatrixType &Ct;
  const InvWOp &invW;
  double gamma;
  const A11InvOp &A11_inv;
  const A22InvOp &A22_inv;
  IndexSet owned_bg, owned_fg, owned_mult;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType lam(owned_mult, comm);
    invW.vmult(lam, src.block(2));
    lam *= -gamma;

    VectorType rhs1(owned_fg, comm);
    M.vmult(rhs1, lam); // M (n_imm x n_mult) * lam -> fg-vector
    rhs1 += src.block(1);
    dst.block(1) = 0.;
    dst.block(1) = A22_inv * rhs1;

    VectorType rhs0(owned_bg, comm);
    VectorType tm(owned_mult, comm), tm2(owned_mult, comm);
    VectorType tb(owned_bg, comm);
    M.Tvmult(tm, dst.block(1)); // M^T * u2 -> mult-vector
    invW.vmult(tm2, tm);        // W^{-1} * M^T u2 in mult
    Ct.vmult(tb, tm2);          // Ct * (W^{-1} M^T u2) in bg
    rhs0 = src.block(0);
    rhs0.add(gamma, tb);
    Ct.vmult(tb, lam);
    rhs0 -= tb;
    dst.block(0) = 0.;
    dst.block(0) = A11_inv * rhs0;

    dst.block(2) = lam;
  }
};

// Ideal AL block-triangular preconditioner.
template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp, typename AugInvOp>
struct IdealALPreconditioner {
  const SparseMatrixType &M;
  const SparseMatrixType &Ct;
  const InvWOp &invW;
  double gamma;
  const AugInvOp &Aug_inv;
  IndexSet owned_bg, owned_fg, owned_mult;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType lam(owned_mult, comm);
    invW.vmult(lam, src.block(2));
    lam *= -gamma;

    BlockVectorType rhs2(std::vector<IndexSet>{owned_bg, owned_fg}, comm);
    rhs2.block(0) = src.block(0);
    {
      VectorType tb(owned_bg, comm);
      Ct.vmult(tb, lam);
      rhs2.block(0) -= tb;
    }
    rhs2.block(1) = src.block(1);
    {
      VectorType ti(owned_fg, comm);
      M.vmult(ti, lam); // M (n_imm x n_mult) * lam -> fg-vector
      rhs2.block(1) += ti;
    }

    BlockVectorType sol2(std::vector<IndexSet>{owned_bg, owned_fg}, comm);
    sol2 = Aug_inv * rhs2;

    dst.block(0) = sol2.block(0);
    dst.block(1) = sol2.block(1);
    dst.block(2) = lam;
  }
};

} // namespace ALParallel

// Parameters

template <int dim> class ProblemParameters : public ParameterAcceptor {
public:
  ProblemParameters();

  std::string output_directory = ".";

  unsigned int initial_background_refinement = 4;
  unsigned int initial_immersed_refinement = 2;

  std::string name_of_background_grid = "hyper_cube";
  std::string arguments_for_background_grid = "-1: 1: true";
  std::string name_of_immersed_grid = "hyper_ball";
  std::string arguments_for_immersed_grid = "0.,0.: 0.3: false";

  // Cell type used for both meshes. Either "hex" (tensor-product
  // quads/hexes, default) or "simplex" (tris/tets). When "simplex" is
  // selected, the bulk grids generated by `generate_from_name_and_arguments`
  // are converted via `convert_hypercube_to_simplex_mesh`.
  std::string mesh_type = "hex";

  unsigned int n_refinement_cycles = 5;

  mutable double beta_1 = 1.;
  mutable double beta_2 = 10.;

  std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

  unsigned int background_space_finite_element_degree = 1;
  unsigned int immersed_space_finite_element_degree = 1;
  unsigned int coupling_quadrature_order = 3;
  unsigned int verbosity_level = 4;

  // Finite-element triple selecting (bg, immersed primal, multiplier):
  //   * "Q1-Q1-Q1"      (default): all three spaces are FE_Q(1). The
  //                     multiplier basis coincides with the immersed
  //                     primal basis; the augmentation in A_bg is the
  //                     particle-quadrature (phi_i, phi_j)_{Omega_2}.
  //   * "Q1-Bubble-Q0": bg = FE_Q(1), immersed primal = FE_Q_Bubbles(1),
  //                     multiplier = FE_DGQ(0). The augmentation in A_bg
  //                     is assembled directly as the bilinear form
  //                     (P_h phi_i, phi_j)_{Omega_2} where P_h is the
  //                     L^2 orthogonal projection on the (cellwise
  //                     constant) Q0 multiplier space.
  // Only the hex mesh path supports "Q1-Bubble-Q0".
  std::string fe_triple = "Q1-Q1-Q1";

  bool use_modified_AL_preconditioner = false;

  // When true, solve() additionally assembles the whole (1,1) block
  // algebraically as the explicit sparse matrix A + gamma_1 * C W^{-1} C^T
  // (= A + C^T M^{-1} C in the notation of the formulation) so that its
  // wall time can be compared against the operator-form (1,1) block
  // assembled in place via particles. Defaults to false. Only the assembly
  // is timed; the resulting matrix is verified against the operator-form
  // (1,1) block and then discarded.
  bool compare_algebraic_augmentation = false;

  bool use_fixed_iterations = true;

  // When true, the immersed grid is built only at cycle 0 and the coupling
  // particles are transferred across background refinement (step-70 style)
  // instead of being re-inserted from scratch. Only effective when the
  // background uses parallel::distributed::Triangulation (mesh_type = hex).
  bool keep_immersed_grid_fixed = false;

  // AL parameters. With the operator form the actual parameter used in the
  // assembly is gamma / h^2; values entered here are the gamma's of the
  // formulation (without the h-scaling).
  mutable double gamma_AL_background = 10.;
  mutable double gamma_AL_immersed = 10.;

  // Optional dump of the bulk stiffness K = A_bg (pre-augmentation),
  // immersed stiffness A_fg (pre-augmentation), coupling matrix C and
  // immersed mass matrix M to disk in MATLAB triplet format. Used by
  // scripts/inf_sup.ipynb for the gamma-h spectrum study. Serial runs only.
  bool export_matrices_for_matlab = false;
  std::string matrix_export_directory = "elliptic_interface_op_export";

  mutable ParameterAcceptorProxy<ReductionControl> outer_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> inner_solver_control;
  mutable ParameterAcceptorProxy<IterationNumberControl>
      iteration_number_control;

  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> f_1;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> f_2_minus_f;
  // Manufactured exact solution from https://arxiv.org/pdf/2211.03443
  //  u (smooth, defined globally on the bulk domain). Used by the convergence
  //  test to compute L^2 and H^1-seminorm  errors for both u_bg (block 0) on
  //  the whole background and u_imm (block 1) on the immersed subdomain. When
  //  the two subdomains carry *different* analytic expressions (e.g. across an
  //  interface with a diffusion jump), set them via the two separate function
  //  subsections.
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact_solution;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> exact_solution_imm;
  bool do_convergence_test = false;
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
    : ParameterAcceptor("Elliptic Interface Problem<" +
                        Utilities::int_to_string(dim) + ">/"),
      outer_solver_control("Outer solver control"),
      inner_solver_control("Inner solver control"),
      iteration_number_control("Iteration number control"),
      f_1("Right hand side f_1"), f_2_minus_f("Right hand side f_2 - f"),
      exact_solution("/Elliptic Interface Problem<" +
                     Utilities::int_to_string(dim) +
                     ">/Convergence test/Exact solution"),
      exact_solution_imm("/Elliptic Interface Problem<" +
                         Utilities::int_to_string(dim) +
                         ">/Convergence test/Exact solution immersed") {
  add_parameter("FE degree background", background_space_finite_element_degree,
                "", this->prm, Patterns::Integer(1));
  add_parameter("FE degree immersed", immersed_space_finite_element_degree, "",
                this->prm, Patterns::Integer(1));
  add_parameter("Coupling quadrature order", coupling_quadrature_order);
  add_parameter("Output directory", output_directory);
  add_parameter("Beta_1", beta_1);
  add_parameter("Beta_2", beta_2);
  add_parameter("Homogeneous Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Use fixed (inner) iterations", use_fixed_iterations,
                "Perform a fixed number of iterations within inner solvers.");

  enter_subsection("Grid generation");
  {
    add_parameter("Background grid generator", name_of_background_grid);
    add_parameter("Background grid generator arguments",
                  arguments_for_background_grid);
    add_parameter("Immersed grid generator", name_of_immersed_grid);
    add_parameter("Immersed grid generator arguments",
                  arguments_for_immersed_grid);
    add_parameter("Mesh type", mesh_type,
                  "Cell type used for both bulk meshes. Either \"hex\" "
                  "(tensor-product quads/hexes) or \"simplex\" "
                  "(tris/tets). When \"simplex\" is selected the meshes "
                  "produced by the grid generator are converted with "
                  "GridGenerator::convert_hypercube_to_simplex_mesh.",
                  this->prm, Patterns::Selection("hex|simplex"));
    add_parameter("FE triple", fe_triple,
                  "Finite-element triple (bg, immersed primal, "
                  "multiplier). \"Q1-Q1-Q1\" (default) uses FE_Q(1) for "
                  "all three spaces; \"Q1-Bubble-Q0\" uses FE_Q(1), "
                  "FE_Q_Bubbles(1) and FE_DGQ(0) respectively. The Q0 "
                  "multiplier triggers a different augmentation in A_bg, "
                  "assembled as (P_h phi_i, phi_j)_{Omega_2}. Only the "
                  "hex mesh path supports \"Q1-Bubble-Q0\".",
                  this->prm, Patterns::Selection("Q1-Q1-Q1|Q1-Bubble-Q0"));
  }
  leave_subsection();

  enter_subsection("Refinement and remeshing");
  {
    add_parameter("Initial background refinement",
                  initial_background_refinement);
    add_parameter("Initial immersed refinement", initial_immersed_refinement);
    add_parameter("Refinemented cycles", n_refinement_cycles);
    add_parameter("Keep immersed grid fixed", keep_immersed_grid_fixed,
                  "If true, the immersed grid is built only at cycle 0 and "
                  "the coupling particles are transferred across background "
                  "refinement (step-70 style) instead of being re-inserted "
                  "from scratch. Only effective when the background uses "
                  "parallel::distributed::Triangulation (mesh_type = hex).");
  }
  leave_subsection();

  enter_subsection("AL preconditioner");
  {
    add_parameter("Use modified AL preconditioner",
                  use_modified_AL_preconditioner);
    add_parameter("gamma fluid", gamma_AL_background);
    add_parameter("gamma solid", gamma_AL_immersed);
    add_parameter("Verbosity level", verbosity_level);
    add_parameter("Compare algebraic augmentation",
                  compare_algebraic_augmentation,
                  "If true, also assemble the whole (1,1) block "
                  "algebraically as the explicit sparse matrix "
                  "A + gamma_1 * C W^{-1} C^T and report its wall time, to "
                  "compare against the operator-form (particle) "
                  "(1,1)-block assembly. Currently supported only for the "
                  "Q1-Bubble-Q0 FE triple (diagonal multiplier mass).");
  }
  leave_subsection();

  enter_subsection("Convergence test");
  {
    add_parameter("Do convergence test", do_convergence_test,
                  "If true, at every refinement cycle compute the L2 and "
                  "H1 seminorm errors against the manufactured Exact "
                  "solution(s) and push them to the convergence table. "
                  "The background Exact solution also supplies Dirichlet "
                  "data. When false, homogeneous Dirichlet data are used "
                  "and the Exact solution subsections are ignored.");
  }
  leave_subsection();

  enter_subsection("Matrix export");
  {
    add_parameter("Export matrices for matlab", export_matrices_for_matlab,
                  "If true, at every refinement cycle dump the bulk "
                  "stiffness K (pre-augmentation), the immersed stiffness "
                  "A_fg (pre-augmentation), the coupling matrix C and the "
                  "immersed mass matrix M to disk in MATLAB triplet format. "
                  "Only supported with a single MPI process.");
    add_parameter("Export directory", matrix_export_directory,
                  "Directory (created if missing) where the *_cycle*.txt "
                  "triplet files are written. Path is interpreted relative "
                  "to the working directory.");
  }
  leave_subsection();

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Tolerance", "1.e-10");
    ParameterAcceptor::prm.set("Reduction", "1.e-10");
    ParameterAcceptor::prm.set("Log history", "true");
    ParameterAcceptor::prm.set("Log result", "true");
  });
  inner_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "100000");
    ParameterAcceptor::prm.set("Tolerance", "1.e-2");
    ParameterAcceptor::prm.set("Reduction", "1.e-20");
    ParameterAcceptor::prm.set("Log history", "false");
    ParameterAcceptor::prm.set("Log result", "true");
  });
  iteration_number_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "30");
    ParameterAcceptor::prm.set("Tolerance", "1.e-4");
  });

  f_1.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "1.");
  });
  f_2_minus_f.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "1.");
  });
  exact_solution.declare_parameters_call_back.connect([]() -> void {
    // Default: zero exact solution (errors will simply measure the solution
    // norm). Override in the .prm file to enable the convergence test.
    ParameterAcceptor::prm.set("Function expression", "0.");
  });
  exact_solution_imm.declare_parameters_call_back.connect([]() -> void {
    // Default: zero. Set to the analytic expression of u_2 on Omega_2
    // when it differs from u_1 (interface problems with jump in beta).
    ParameterAcceptor::prm.set("Function expression", "0.");
  });
}

// Parallel elliptic interface solver (operator-form AL)

template <int dim> class EllipticInterfaceDLMParallel {
public:
  EllipticInterfaceDLMParallel(const ProblemParameters<dim> &prm);

  void run();

private:
  void generate_grids(unsigned int extra_refinement = 0);
  void system_setup();
  void assemble_bg();
  void assemble_fg(double gamma_2_h);
  void assemble_mass_fg();
  void assemble_mass_mult();
  void add_projected_mass_fg(double gamma_2_h);
  void assemble_coupling_and_augment_bg(double gamma_1_h);
  // Assemble the whole (1,1) block algebraically as the explicit sparse
  // matrix A + gamma_1 * C W^{-1} C^T and time it, to compare against the
  // operator-form (particle) (1,1)-block assembly. Validates the result by
  // a matrix-vector comparison against the in-place augmented A_bg.
  void compute_algebraic_augmentation(double gamma_1_h);
  unsigned int solve();
  void compute_errors();
  // Consistency of the operator-form AL augmentation: measure how close
  // R_u (= what is actually added to A_bg via particle quadrature) is to
  // the ideal C^T M^{-1} C augmentation, evaluated on the discrete
  // background solution u_bg. Returns ||(R_u - C^T M^{-1} C) u_bg||_l2.
  void compute_augmentation_consistency();
  void output_results(unsigned int cycle) const;

  const ProblemParameters<dim> &parameters;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;
  mutable TimerOutput computing_timer;

  // Polymorphic background / immersed triangulations. When `mesh_type == hex`
  // we use `parallel::distributed::Triangulation` (p4est-based), which
  // supports in-place `refine_global` between cycles. When
  // `mesh_type == simplex` we fall back to
  // `parallel::fullydistributed::Triangulation`, which is rebuilt from
  // scratch at every cycle (it does not support `refine_global`).
  std::unique_ptr<parallel::DistributedTriangulationBase<dim>> tria_bg;
  std::unique_ptr<parallel::DistributedTriangulationBase<dim>> tria_fg;

  // Set in the constructor according to parameters.mesh_type. For
  // "hex" we use FE_Q + MappingQ1; for "simplex" we use FE_SimplexP +
  // MappingFE(FE_SimplexP(1)). Held by pointer because the concrete type
  // depends on a run-time parameter. For the FE triple "Q1-Bubble-Q0"
  // (hex only) `fe_fg_ptr` is FE_Q_Bubbles(1) and `fe_mult_ptr` is
  // FE_DGQ(0); otherwise `fe_mult_ptr` is just a twin of `fe_fg_ptr`
  // and `dof_handler_mult` produces the same numbering as `dof_handler_fg`.
  std::unique_ptr<FiniteElement<dim>> fe_bg_ptr;
  std::unique_ptr<FiniteElement<dim>> fe_fg_ptr;
  std::unique_ptr<FiniteElement<dim>> fe_mult_ptr;
  std::unique_ptr<Mapping<dim>> mapping_ptr;
  // Quadrature factory: QGauss for hex, QGaussSimplex for simplex.
  std::function<Quadrature<dim>(unsigned int)> make_quadrature;

  // Derived flag from parameters.fe_triple. True when the multiplier is
  // FE_DGQ(0) and the augmentation in A_bg must be assembled as
  // (P_h phi_i, phi_j)_{Omega_2}.
  bool use_q0_multiplier = false;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;
  DoFHandler<dim> dof_handler_mult;

  IndexSet owned_bg, relevant_bg;
  IndexSet owned_fg, relevant_fg;
  IndexSet owned_mult, relevant_mult;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;
  AffineConstraints<double> constraints_mult;

  // Background stiffness (will be augmented in-place by particle quadrature).
  LA::MPI::SparseMatrix A_bg;
  // Copy of the bare bulk stiffness BEFORE the AL augmentation is added
  // in place to A_bg. Used by compute_augmentation_consistency() to extract
  // the (operator-form) augmentation R_u = (A_bg - K_bg_pre_aug) / gamma_1.
  LA::MPI::SparseMatrix K_bg_pre_aug;
  // Immersed stiffness (beta_2 - beta_1)(grad u, grad v) + gamma_2*M (when
  // assembling the augmented A22 preconditioner block).
  LA::MPI::SparseMatrix A_fg;
  LA::MPI::SparseMatrix A_fg_plus_scaled_M;
  // Immersed cross mass matrix M: (psi_imm_i, mu_j)_{Omega_2}. Size
  // n_imm x n_mult. For Q1-Q1-Q1 (multiplier == primal) this collapses to
  // the usual square symmetric immersed mass matrix.
  LA::MPI::SparseMatrix mass_matrix_fg;
  // Multiplier mass matrix W: (mu_i, mu_j)_{Omega_2}. Size
  // n_mult x n_mult. Used as the W operator in `invW = W^{-1}`. For
  // Q1-Q1-Q1 it coincides entry-wise with `mass_matrix_fg` (and with
  // the immersed primal mass); for the Q0 multiplier it is the diagonal
  // matrix diag(|K|) over immersed cells.
  LA::MPI::SparseMatrix mass_matrix_mult;
  // Coupling matrices for augmented formulation.
  // coupling_matrix: C - the original coupling between background and the
  // Lagrange multiplier space (used in the (0,2) and (2,0) blocks).
  // (u|_Omega_2, mu) - size n_bg x n_mult.
  LA::MPI::SparseMatrix coupling_matrix;
  // coupling_matrix_01: (u|_Omega_2, v_2) - integral of phi_bg*psi_imm over
  // Omega_2.  Size n_bg x n_imm (always immersed primal columns).
  LA::MPI::SparseMatrix coupling_matrix_01;
  // coupling_matrix_10: (u_2, v|_Omega_2) - transpose of coupling_matrix_01
  LA::MPI::SparseMatrix coupling_matrix_10;
  // Cross matrix (n_mult x n_bg) used only when `use_q0_multiplier` is
  // true: entry [K, i] = int_K phi_bg_i, accumulated from particles. It
  // is the data we need to assemble the augmentation
  // (P_h phi_i, phi_j)_{Omega_2} = sum_K (1/|K|) (int_K phi_i)(int_K phi_j)
  // directly on each immersed cell K.
  LA::MPI::SparseMatrix bg_mult_integral_T;
  // The immersed-side twin of `bg_mult_integral_T` (n_mult x n_imm), i.e.
  // M^T: entry [K, j] = int_K phi_imm_j. Together with c_K it gives the
  // projected off-diagonal blocks as per-cell outer products. Q0 only.
  LA::MPI::SparseMatrix imm_mult_integral_T;

  LA::MPI::BlockVector system_rhs_block;
  LA::MPI::BlockVector system_solution_block;

  // Persistent particle handler. When `keep_immersed_grid_fixed` is true and
  // the background is distributed (hex), the particle handler is initialised
  // once at cycle 0 and then transferred across bg refinement; subsequent
  // calls to `assemble_coupling_and_augment_bg` only need to assemble Ct and
  // the AL augmentation (sparsity + values) without re-inserting particles.
  Particles::ParticleHandler<dim> particles;
  bool particles_built = false;

  // Last value of the (h-scaled) AL parameter on the bg block; cached by
  // solve() so that compute_augmentation_consistency() can rescale the
  // augmentation extracted from A_bg.
  double last_gamma_1 = 0.0;

  // Cumulative number of inner (P)CG iterations spent inverting the (1,1)
  // block A_bg of the preconditioner during the last outer FGMRES solve,
  // together with the number of times that block was inverted. Used to
  // report the average inner PCG iteration count per (1,1)-block solve.
  unsigned int last_inner_iters_A11 = 0;
  unsigned int last_n_A11_applications = 0;

  ConvergenceTable convergence_table;
};

template <int dim>
EllipticInterfaceDLMParallel<dim>::EllipticInterfaceDLMParallel(
    const ProblemParameters<dim> &prm)
    : parameters(prm), mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, this_mpi_process == 0),
      computing_timer(mpi_communicator, pcout, TimerOutput::summary,
                      TimerOutput::wall_times) {
  AssertThrow(parameters.mesh_type == "hex" ||
                  parameters.mesh_type == "simplex",
              ExcMessage("Mesh type must be \"hex\" or \"simplex\"."));
  AssertThrow(parameters.fe_triple == "Q1-Q1-Q1" ||
                  parameters.fe_triple == "Q1-Bubble-Q0",
              ExcMessage("FE triple must be \"Q1-Q1-Q1\" or "
                         "\"Q1-Bubble-Q0\"."));
  AssertThrow(parameters.fe_triple == "Q1-Q1-Q1" ||
                  parameters.mesh_type == "hex",
              ExcMessage("FE triple \"Q1-Bubble-Q0\" is only supported "
                         "on hex meshes."));
  use_q0_multiplier = (parameters.fe_triple == "Q1-Bubble-Q0");
  if (parameters.mesh_type == "hex") {
    tria_bg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
    tria_fg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
    fe_bg_ptr = std::make_unique<FE_Q<dim>>(
        parameters.background_space_finite_element_degree);
    if (use_q0_multiplier) {
      // The Q1-Bubble-Q0 triple ignores the per-FE-degree parameters and
      // pins the immersed primal to FE_Q_Bubbles(1) and the multiplier
      // to FE_DGQ(0).
      fe_fg_ptr = std::make_unique<FE_Q_Bubbles<dim>>(1);
      fe_mult_ptr = std::make_unique<FE_DGQ<dim>>(0);
    } else {
      fe_fg_ptr = std::make_unique<FE_Q<dim>>(
          parameters.immersed_space_finite_element_degree);
      fe_mult_ptr = std::make_unique<FE_Q<dim>>(
          parameters.immersed_space_finite_element_degree);
    }
    mapping_ptr = std::make_unique<MappingQ1<dim>>();
    make_quadrature = [](unsigned int n) -> Quadrature<dim> {
      return QGauss<dim>(n);
    };
  } else {
    tria_bg = std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
    tria_fg = std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
    fe_bg_ptr = std::make_unique<FE_SimplexP<dim>>(
        parameters.background_space_finite_element_degree);
    fe_fg_ptr = std::make_unique<FE_SimplexP<dim>>(
        parameters.immersed_space_finite_element_degree);
    fe_mult_ptr = std::make_unique<FE_SimplexP<dim>>(
        parameters.immersed_space_finite_element_degree);
    mapping_ptr = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    // QGaussSimplex is implemented for 1..4 points per direction only.
    make_quadrature = [](unsigned int n) -> Quadrature<dim> {
      return QGaussSimplex<dim>(std::min(4u, std::max(1u, n)));
    };
  }
  dof_handler_bg.reinit(*tria_bg);
  dof_handler_fg.reinit(*tria_fg);
  dof_handler_mult.reinit(*tria_fg);
  AssertThrow(parameters.beta_1 > 0., ExcMessage("Beta_1 must be positive."));
  AssertThrow(parameters.beta_2 >= parameters.beta_1,
              ExcMessage("Beta_2 must be greater than or equal to Beta_1."));
  AssertThrow(parameters.gamma_AL_background > 0.,
              ExcMessage("gamma fluid must be positive."));
  AssertThrow(parameters.gamma_AL_immersed > 0.,
              ExcMessage("gamma solid must be positive."));
  AssertThrow(parameters.gamma_AL_immersed <= parameters.gamma_AL_background,
              ExcMessage("gamma solid should not exceed gamma fluid."));

  if (this_mpi_process == 0 &&
      !std::filesystem::exists(parameters.output_directory)) {
    pcout << "Output directory does not exist, creating: "
          << parameters.output_directory << std::endl;
    std::filesystem::create_directories(parameters.output_directory);
  }
}

template <int dim>
void EllipticInterfaceDLMParallel<dim>::generate_grids(
    unsigned int extra_refinement) {
  TimerOutput::Scope t(computing_timer, "Grid generation");

  // Hex path: build directly into a `parallel::distributed::Triangulation`.
  // At the first call (cycle 0) the coarse mesh is generated and refined to
  // the initial level; on every subsequent call we just call
  // `refine_global(1)` so that the mesh hierarchy is built incrementally,
  // without clearing and rebuilding from scratch. The resulting meshes are
  // identical (at every cycle) to those produced by the previous code path
  // for hex meshes.
  //
  // When `keep_immersed_grid_fixed` is true:
  //   - the immersed grid is NOT refined after cycle 0;
  //   - the bg `refine_global(1)` is wrapped with
  //     `particles.prepare_for_coarsening_and_refinement()` /
  //     `unpack_after_coarsening_and_refinement()` so that the particle
  //     handler is transferred (step-70 style) rather than re-inserted.
  auto build_or_refine_distributed =
      [this](parallel::distributed::Triangulation<dim> &out,
             const std::string &name, const std::string &args,
             unsigned int initial_refinements, bool is_bg) {
        if (out.n_global_active_cells() == 0) {
          GridGenerator::generate_from_name_and_arguments(out, name, args);
          out.refine_global(initial_refinements);
        } else {
          if (!is_bg && parameters.keep_immersed_grid_fixed)
            return; // keep immersed grid fixed across cycles.
          const bool transfer_particles =
              is_bg && particles_built && parameters.keep_immersed_grid_fixed;
          if (transfer_particles)
            particles.prepare_for_coarsening_and_refinement();
          out.refine_global(1);
          if (transfer_particles)
            particles.unpack_after_coarsening_and_refinement();
        }
      };

  // Simplex path: parallel::fullydistributed::Triangulation does not support
  // refine_global, so we rebuild from scratch each cycle. When the bg is
  // rebuilt, the existing particle handler is invalidated; the assembly
  // routine will re-insert particles from scratch.
  auto build_one_fullydistributed =
      [this](parallel::fullydistributed::Triangulation<dim> &out,
             const std::string &name, const std::string &args,
             unsigned int n_refinements, bool is_bg) {
        if (!is_bg && parameters.keep_immersed_grid_fixed &&
            out.n_global_active_cells() > 0)
          return; // keep immersed grid fixed across cycles.
        if (is_bg)
          particles_built = false;
        out.clear();
        Triangulation<dim> serial;
        Triangulation<dim> hex_tmp;
        GridGenerator::generate_from_name_and_arguments(hex_tmp, name, args);
        hex_tmp.refine_global(n_refinements);
        GridGenerator::convert_hypercube_to_simplex_mesh(hex_tmp, serial);

        GridTools::partition_triangulation(
            Utilities::MPI::n_mpi_processes(mpi_communicator), serial);
        for (const auto id : serial.get_manifold_ids())
          if (id != numbers::flat_manifold_id)
            out.set_manifold(id, serial.get_manifold(id));
        const auto desc = TriangulationDescription::Utilities::
            create_description_from_triangulation(serial, mpi_communicator);
        out.create_triangulation(desc);
      };

  auto generate_one =
      [&](std::unique_ptr<parallel::DistributedTriangulationBase<dim>> &tria,
          const std::string &name, const std::string &args,
          unsigned int initial_refinements, bool is_bg) {
        if (auto *tria_d =
                dynamic_cast<parallel::distributed::Triangulation<dim> *>(
                    tria.get())) {
          build_or_refine_distributed(*tria_d, name, args, initial_refinements,
                                      is_bg);
        } else {
          auto &tria_fd =
              dynamic_cast<parallel::fullydistributed::Triangulation<dim> &>(
                  *tria);
          build_one_fullydistributed(tria_fd, name, args,
                                     initial_refinements + extra_refinement,
                                     is_bg);
        }
      };

  generate_one(tria_bg, parameters.name_of_background_grid,
               parameters.arguments_for_background_grid,
               parameters.initial_background_refinement, /*is_bg=*/true);
  generate_one(tria_fg, parameters.name_of_immersed_grid,
               parameters.arguments_for_immersed_grid,
               parameters.initial_immersed_refinement, /*is_bg=*/false);

  const double h_bg = GridTools::maximal_cell_diameter(*tria_bg);
  const double h_fg = GridTools::maximal_cell_diameter(*tria_fg);
  pcout << "h background = " << h_bg << "\n"
        << "h immersed   = " << h_fg << "\n"
        << "ratio (bg/imm) = " << h_bg / h_fg << std::endl;
}

template <int dim> void EllipticInterfaceDLMParallel<dim>::system_setup() {
  TimerOutput::Scope t(computing_timer, "System setup");
  const auto &fe_bg = *fe_bg_ptr;
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;

  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);
  dof_handler_mult.distribute_dofs(fe_mult);

  owned_bg = dof_handler_bg.locally_owned_dofs();
  relevant_bg = DoFTools::extract_locally_relevant_dofs(dof_handler_bg);

  owned_fg = dof_handler_fg.locally_owned_dofs();
  relevant_fg = DoFTools::extract_locally_relevant_dofs(dof_handler_fg);

  owned_mult = dof_handler_mult.locally_owned_dofs();
  relevant_mult = DoFTools::extract_locally_relevant_dofs(dof_handler_mult);

  // Use manufactured Dirichlet data only for the convergence test.
  constraints_bg.clear();
  constraints_bg.reinit(owned_bg, relevant_bg);
  DoFTools::make_hanging_node_constraints(dof_handler_bg, constraints_bg);
  for (const auto id : parameters.dirichlet_ids) {
    if (parameters.do_convergence_test)
      VectorTools::interpolate_boundary_values(*mapping_ptr, dof_handler_bg, id,
                                               parameters.exact_solution,
                                               constraints_bg);
    else
      VectorTools::interpolate_boundary_values(*mapping_ptr, dof_handler_bg, id,
                                               Functions::ZeroFunction<dim>(),
                                               constraints_bg);
  }
  constraints_bg.close();

  // No constraints on immersed primal: Lagrange multiplier and immersed
  // unknown have their natural conditions. Bubble DoFs in FE_Q_Bubbles
  // are interior to cells, so the usual hanging-node machinery still does
  // the right thing on the Q1 part.
  constraints_fg.clear();
  constraints_fg.reinit(owned_fg, relevant_fg);
  DoFTools::make_hanging_node_constraints(dof_handler_fg, constraints_fg);
  constraints_fg.close();

  // Multiplier constraints: hanging-node only (none for FE_DGQ).
  constraints_mult.clear();
  constraints_mult.reinit(owned_mult, relevant_mult);
  DoFTools::make_hanging_node_constraints(dof_handler_mult, constraints_mult);
  constraints_mult.close();

  {
    DynamicSparsityPattern dsp(dof_handler_bg.n_dofs(), dof_handler_bg.n_dofs(),
                               relevant_bg);
    DoFTools::make_sparsity_pattern(dof_handler_bg, dsp, constraints_bg, false);
    SparsityTools::distribute_sparsity_pattern(dsp, owned_bg, mpi_communicator,
                                               relevant_bg);
    A_bg.reinit(owned_bg, owned_bg, dsp, mpi_communicator);
  }
  {
    DynamicSparsityPattern dsp(dof_handler_fg.n_dofs(), dof_handler_fg.n_dofs(),
                               relevant_fg);
    DoFTools::make_sparsity_pattern(dof_handler_fg, dsp, constraints_fg, false);
    SparsityTools::distribute_sparsity_pattern(dsp, owned_fg, mpi_communicator,
                                               relevant_fg);
    A_fg.reinit(owned_fg, owned_fg, dsp, mpi_communicator);
    A_fg_plus_scaled_M.reinit(owned_fg, owned_fg, dsp, mpi_communicator);
  }
  // Cross-mass sparsity (n_imm x n_mult): on each cell, every immersed
  // primal DoF couples to every multiplier DoF of the same cell.
  {
    DynamicSparsityPattern dsp_cross(dof_handler_fg.n_dofs(),
                                     dof_handler_mult.n_dofs(), relevant_fg);
    const unsigned int n_imm_dofs = fe_fg.n_dofs_per_cell();
    const unsigned int n_mult_dofs = fe_mult.n_dofs_per_cell();
    std::vector<types::global_dof_index> imm_ldi(n_imm_dofs);
    std::vector<types::global_dof_index> mult_ldi(n_mult_dofs);
    for (const auto &cell : dof_handler_fg.active_cell_iterators())
      if (cell->is_locally_owned()) {
        cell->get_dof_indices(imm_ldi);
        typename DoFHandler<dim>::cell_iterator mcell(
            &cell->get_triangulation(), cell->level(), cell->index(),
            &dof_handler_mult);
        mcell->get_dof_indices(mult_ldi);
        for (const auto i : imm_ldi)
          for (const auto j : mult_ldi)
            dsp_cross.add(i, j);
      }
    SparsityTools::distribute_sparsity_pattern(dsp_cross, owned_fg,
                                               mpi_communicator, relevant_fg);
    mass_matrix_fg.reinit(owned_fg, owned_mult, dsp_cross, mpi_communicator);
  }
  // Multiplier mass sparsity (n_mult x n_mult): standard cell-local
  // stencil. For FE_DGQ(0) it is diagonal.
  {
    DynamicSparsityPattern dsp_mult(dof_handler_mult.n_dofs(),
                                    dof_handler_mult.n_dofs(), relevant_mult);
    DoFTools::make_sparsity_pattern(dof_handler_mult, dsp_mult,
                                    constraints_mult, false);
    SparsityTools::distribute_sparsity_pattern(dsp_mult, owned_mult,
                                               mpi_communicator, relevant_mult);
    mass_matrix_mult.reinit(owned_mult, owned_mult, dsp_mult, mpi_communicator);
  }

  // Block vectors
  std::vector<IndexSet> owned_blocks = {owned_bg, owned_fg, owned_mult};
  std::vector<IndexSet> relevant_blocks = {relevant_bg, relevant_fg,
                                           relevant_mult};
  system_rhs_block.reinit(owned_blocks, mpi_communicator);
  system_solution_block.reinit(owned_blocks, mpi_communicator);

  pcout << "N DoF background: " << dof_handler_bg.n_dofs() << "\n"
        << "N DoF immersed:   " << dof_handler_fg.n_dofs() << "\n"
        << "N DoF multiplier: " << dof_handler_mult.n_dofs() << std::endl;
}

template <int dim> void EllipticInterfaceDLMParallel<dim>::assemble_bg() {
  TimerOutput::Scope t(computing_timer, "Assemble bg (A_omega1)");
  const auto &fe_bg = *fe_bg_ptr;
  const Quadrature<dim> quad = make_quadrature(fe_bg.degree + 1);
  FEValues<dim> fe_values(*mapping_ptr, fe_bg, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe_bg.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  std::vector<double> rhs_values(quad.size());

  LA::MPI::Vector rhs_owned(owned_bg, mpi_communicator);
  rhs_owned = 0.;

  for (const auto &cell : dof_handler_bg.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      parameters.f_1.value_list(fe_values.get_quadrature_points(), rhs_values);
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += parameters.beta_1 *
                                 fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q) * fe_values.JxW(q);
          cell_rhs(i) +=
              rhs_values[q] * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
      cell->get_dof_indices(ldi);
      constraints_bg.distribute_local_to_global(cell_matrix, cell_rhs, ldi,
                                                A_bg, rhs_owned);
    }
  A_bg.compress(VectorOperation::add);
  rhs_owned.compress(VectorOperation::add);
  system_rhs_block.block(0) = rhs_owned;
}

template <int dim>
void EllipticInterfaceDLMParallel<dim>::assemble_fg(double gamma_2_scaled) {
  TimerOutput::Scope t(computing_timer, "Assemble fg (A_omega2 and A22-prec)");
  const auto &fe_fg = *fe_fg_ptr;
  const Quadrature<dim> quad = make_quadrature(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_ptr, fe_fg, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  FullMatrix<double> cm_A(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cm_Aug(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  std::vector<double> rhs_values(quad.size());

  // Q0 multiplier: the (1,1) mass term is the *projected* one,
  // gamma_2 * M W^{-1} M^T, added cellwise by `add_projected_mass_fg`.
  // Only when the multiplier equals the immersed primal space (W = M) do
  // the plain and the projected mass agree, and the term can be added here.
  const double rho = use_q0_multiplier ? 0. : gamma_2_scaled;
  const double mu =
      parameters.beta_2 - parameters.beta_1; // coeff of (grad,grad)

  LA::MPI::Vector rhs_owned(owned_fg, mpi_communicator);
  rhs_owned = 0.;

  for (const auto &cell : dof_handler_fg.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm_A = 0;
      cm_Aug = 0;
      cell_rhs = 0;
      parameters.f_2_minus_f.value_list(fe_values.get_quadrature_points(),
                                        rhs_values);
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = fe_values.shape_value(i, q);
          const auto grad_i = fe_values.shape_grad(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const double phi_j = fe_values.shape_value(j, q);
            const auto grad_j = fe_values.shape_grad(j, q);
            const double a = mu * grad_i * grad_j * fe_values.JxW(q);
            const double m = phi_i * phi_j * fe_values.JxW(q);
            cm_A(i, j) += a;
            cm_Aug(i, j) += a + rho * m;
          }
          cell_rhs(i) += rhs_values[q] * phi_i * fe_values.JxW(q);
        }
      cell->get_dof_indices(ldi);
      constraints_fg.distribute_local_to_global(cm_A, ldi, A_fg);
      constraints_fg.distribute_local_to_global(cm_Aug, cell_rhs, ldi,
                                                A_fg_plus_scaled_M, rhs_owned);
    }
  A_fg.compress(VectorOperation::add);
  A_fg_plus_scaled_M.compress(VectorOperation::add);
  rhs_owned.compress(VectorOperation::add);
  system_rhs_block.block(1) = rhs_owned;
}

// Q0 multiplier only: add the *projected* AL mass
//
//   gamma_2 * M W^{-1} M^T = gamma_2 * sum_K (1/|K|) m_K m_K^T,
//   m_K[j] = int_K phi_j^imm dx,   |K| = int_K 1 dx,
//
// to the (1,1) block, and build M^T (= imm_mult_integral_T) which the
// projected off-diagonal blocks need in `assemble_coupling_and_augment_bg`.
//
// W is diagonal for a cellwise-constant multiplier, so every term is a rank-1
// outer product over the dofs of a SINGLE immersed cell: no global triple
// product is formed and the result stays inside the standard immersed
// sparsity pattern. Together with the matching (0,0)/(0,1)/(1,0) blocks the
// whole AL contribution becomes gamma * ||P_h(u_1 - u_2)||^2_{L^2(Omega_2)},
// which vanishes identically on the discrete constraint manifold
// Ct^T u_1 = M^T u_2, so the augmented system has exactly the same
// solution as the unaugmented saddle point, for any gamma.
template <int dim>
void EllipticInterfaceDLMParallel<dim>::add_projected_mass_fg(
    double gamma_2_scaled) {
  if (!use_q0_multiplier)
    return;
  TimerOutput::Scope t(computing_timer, "Projected AL mass (1,1) block");
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;
  const Quadrature<dim> quad = make_quadrature(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_ptr, fe_fg, quad,
                          update_values | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  const unsigned int n_mult_dofs = fe_mult.n_dofs_per_cell();

  // Sparsity of M^T (n_mult x n_imm): one block row per immersed cell.
  {
    DynamicSparsityPattern dsp(dof_handler_mult.n_dofs(),
                               dof_handler_fg.n_dofs(), relevant_mult);
    std::vector<types::global_dof_index> ldi_fg(dofs_per_cell);
    std::vector<types::global_dof_index> ldi_mult(n_mult_dofs);
    typename DoFHandler<dim>::active_cell_iterator
        cell_fg = dof_handler_fg.begin_active(),
        end_fg = dof_handler_fg.end(),
        cell_mult = dof_handler_mult.begin_active();
    for (; cell_fg != end_fg; ++cell_fg, ++cell_mult)
      if (cell_fg->is_locally_owned()) {
        cell_fg->get_dof_indices(ldi_fg);
        cell_mult->get_dof_indices(ldi_mult);
        constraints_mult.add_entries_local_to_global(ldi_mult, constraints_fg,
                                                     ldi_fg, dsp);
      }
    SparsityTools::distribute_sparsity_pattern(dsp, owned_mult,
                                               mpi_communicator, relevant_mult);
    imm_mult_integral_T.reinit(owned_mult, owned_fg, dsp, mpi_communicator);
  }

  FullMatrix<double> cm(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cm_mT(n_mult_dofs, dofs_per_cell);
  std::vector<double> mvec(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  std::vector<types::global_dof_index> ldi_mult(n_mult_dofs);

  typename DoFHandler<dim>::active_cell_iterator cell_mult_it =
      dof_handler_mult.begin_active();
  for (const auto &cell : dof_handler_fg.active_cell_iterators()) {
    const auto cell_mult = cell_mult_it;
    ++cell_mult_it;
    if (!cell->is_locally_owned())
      continue;
    fe_values.reinit(cell);
    std::fill(mvec.begin(), mvec.end(), 0.);
    double measure = 0.;
    for (unsigned int q = 0; q < quad.size(); ++q) {
      measure += fe_values.JxW(q);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        mvec[i] += fe_values.shape_value(i, q) * fe_values.JxW(q);
    }
    Assert(measure > 0., ExcMessage("Immersed cell has non-positive measure."));
    const double scale = gamma_2_scaled / measure;

    cm = 0;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        cm(i, j) = scale * mvec[i] * mvec[j];

    // Row K of M^T. The Q0 multiplier shape is identically 1 on the cell,
    // so the entry is just int_K phi_j^imm = mvec[j].
    cm_mT = 0;
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      cm_mT(0, j) = mvec[j];

    cell->get_dof_indices(ldi);
    cell_mult->get_dof_indices(ldi_mult);
    constraints_fg.distribute_local_to_global(cm, ldi, A_fg_plus_scaled_M);
    constraints_mult.distribute_local_to_global(cm_mT, ldi_mult, constraints_fg,
                                                ldi, imm_mult_integral_T);
  }
  A_fg_plus_scaled_M.compress(VectorOperation::add);
  imm_mult_integral_T.compress(VectorOperation::add);
}

template <int dim> void EllipticInterfaceDLMParallel<dim>::assemble_mass_fg() {
  TimerOutput::Scope t(computing_timer, "Assemble cross-mass (immersed)");
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;
  const unsigned int q_order = std::max(fe_fg.degree, fe_mult.degree) + 1;
  const Quadrature<dim> quad = make_quadrature(q_order);
  FEValues<dim> fe_v_imm(*mapping_ptr, fe_fg, quad,
                         update_values | update_JxW_values);
  FEValues<dim> fe_v_mult(*mapping_ptr, fe_mult, quad, update_values);
  const unsigned int n_imm_dofs = fe_fg.n_dofs_per_cell();
  const unsigned int n_mult_dofs = fe_mult.n_dofs_per_cell();
  FullMatrix<double> cm(n_imm_dofs, n_mult_dofs);
  std::vector<types::global_dof_index> imm_ldi(n_imm_dofs);
  std::vector<types::global_dof_index> mult_ldi(n_mult_dofs);
  for (const auto &cell : dof_handler_fg.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_v_imm.reinit(cell);
      typename DoFHandler<dim>::cell_iterator mcell(
          &cell->get_triangulation(), cell->level(), cell->index(),
          &dof_handler_mult);
      fe_v_mult.reinit(mcell);
      cm = 0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n_imm_dofs; ++i)
          for (unsigned int j = 0; j < n_mult_dofs; ++j)
            cm(i, j) += fe_v_imm.shape_value(i, q) *
                        fe_v_mult.shape_value(j, q) * fe_v_imm.JxW(q);
      cell->get_dof_indices(imm_ldi);
      mcell->get_dof_indices(mult_ldi);
      constraints_fg.distribute_local_to_global(cm, imm_ldi, constraints_mult,
                                                mult_ldi, mass_matrix_fg);
    }
  mass_matrix_fg.compress(VectorOperation::add);
}

// Multiplier-multiplier mass matrix: W_{ij} = (mu_i, mu_j)_{Omega_2}. Used
// as the W in `invW = W^{-1}` everywhere. For FE_DGQ(0) this matrix is
// diagonal with entries |K|.
template <int dim>
void EllipticInterfaceDLMParallel<dim>::assemble_mass_mult() {
  TimerOutput::Scope t(computing_timer, "Assemble mass matrix (multiplier)");
  const auto &fe_mult = *fe_mult_ptr;
  const Quadrature<dim> quad = make_quadrature(fe_mult.degree + 1);
  FEValues<dim> fe_values(*mapping_ptr, fe_mult, quad,
                          update_values | update_JxW_values);
  const unsigned int dofs_per_cell = fe_mult.n_dofs_per_cell();
  FullMatrix<double> cm(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  for (const auto &cell : dof_handler_mult.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm = 0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cm(i, j) += fe_values.shape_value(i, q) *
                        fe_values.shape_value(j, q) * fe_values.JxW(q);
      cell->get_dof_indices(ldi);
      constraints_mult.distribute_local_to_global(cm, ldi, mass_matrix_mult);
    }
  mass_matrix_mult.compress(VectorOperation::add);
}

// Particle-based assembly of the coupling matrices for the augmented
// formulation:
//   - coupling_matrix:    (u|_Omega_2, mu)  - n_bg x n_mult
//   - coupling_matrix_01: (u|_Omega_2, v_2) - n_bg x n_imm
//   - coupling_matrix_10: (u_2, v|_Omega_2) - n_imm x n_bg (transpose of _01)
//
// Augmentation of A_bg, added in place:
//   * Q1-Q1-Q1 (default): the particle-quadrature approximation
//        gamma_1 * sum_q phi_i(x_q) phi_j(x_q) JxW_q  ~  gamma_1 (phi_i,
//        phi_j)_{Omega_2}.
//   * Q1-Bubble-Q0: the *direct* assembly of
//        gamma_1 * (P_h phi_i, phi_j)_{Omega_2} = gamma_1 * sum_K (1/|K|)
//              (int_K phi_i)(int_K phi_j),
//     where P_h is the L^2 projection onto FE_DGQ(0). The intermediate
//     matrix bg_mult_integral_T (rows = K, cols = bg dofs) is assembled
//     alongside the coupling matrices, then a K-driven outer-product pass
//     writes the rank-1 contributions. A_bg's sparsity is enlarged to
//     accommodate cross-bg-cell (i,j) pairs that may now appear.
//
// Particle property layout (n_props doubles per particle):
//   [0]                              : JxW at the immersed quadrature point
//   [1 ... 1 + n_imm_dofs - 1]       : phi_imm shape values
//   [1 + n_imm_dofs ... 1 + 2 n_imm_dofs - 1]
//                                    : imm dof indices (cast to double)
//   (only if use_q0_multiplier)
//   [1 + 2 n_imm_dofs ... + n_mult_dofs - 1]
//                                    : multiplier dof indices (cast to double)
// Multiplier shape values are not stored: for FE_DGQ(0) they are identically 1
// and for the Q1-Q1-Q1 triple the multiplier basis coincides (DoF for DoF)
// with the immersed primal basis.
template <int dim>
void EllipticInterfaceDLMParallel<dim>::assemble_coupling_and_augment_bg(
    double gamma_1_scaled) {
  TimerOutput::Scope t(computing_timer,
                       "Coupling assembly and bg augmentation (particles)");
  const auto &fe_bg = *fe_bg_ptr;
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;

  const unsigned int n_imm_dofs = fe_fg.n_dofs_per_cell();
  const unsigned int n_mult_dofs = fe_mult.n_dofs_per_cell();
  const unsigned int mult_off = 1 + 2 * n_imm_dofs; // start of mult dofs
  const unsigned int n_props =
      mult_off + (use_q0_multiplier ? n_mult_dofs : 0u);

  // Helper: extract the multiplier DoF indices stored in property pp for
  // a given particle. For Q1-Q1-Q1 the multiplier DoFs coincide with the
  // immersed DoFs (same FE + same triangulation imply the same numbering
  // across the two DoFHandlers); for Q0 they are stored explicitly.
  auto mult_dofs_from_props = [&](const ArrayView<const double> &pp,
                                  std::vector<types::global_dof_index> &out) {
    out.resize(n_mult_dofs);
    if (use_q0_multiplier) {
      for (unsigned int j = 0; j < n_mult_dofs; ++j)
        out[j] = static_cast<types::global_dof_index>(pp[mult_off + j]);
    } else {
      for (unsigned int j = 0; j < n_mult_dofs; ++j)
        out[j] = static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + j]);
    }
  };

  const bool reuse_particles =
      particles_built && parameters.keep_immersed_grid_fixed;

  if (!reuse_particles) {
    particles.clear_particles();

    // Build global bg bounding boxes (level 1) for particle insertion.
    std::vector<BoundingBox<dim>> all_local_boxes;
    all_local_boxes.reserve(tria_bg->n_locally_owned_active_cells());
    for (const auto &cell : tria_bg->active_cell_iterators())
      if (cell->is_locally_owned())
        all_local_boxes.emplace_back(cell->bounding_box());
    const auto tree = pack_rtree(all_local_boxes);
    const auto local_boxes = extract_rtree_level(tree, 1);
    const auto global_boxes =
        Utilities::MPI::all_gather(mpi_communicator, local_boxes);

    particles.initialize(*tria_bg, *mapping_ptr, n_props);

    // Generate particles at immersed quadrature points (on locally-owned
    // cells).
    const Quadrature<dim> imm_quad =
        make_quadrature(2 * std::max(fe_bg.degree, fe_fg.degree) + 1);
    FEValues<dim> fe_v_imm(*mapping_ptr, fe_fg, imm_quad,
                           update_values | update_quadrature_points |
                               update_JxW_values);

    std::vector<Point<dim>> points;
    std::vector<std::vector<double>> props;
    std::vector<types::global_dof_index> imm_ldi(n_imm_dofs);
    std::vector<types::global_dof_index> mult_ldi(n_mult_dofs);
    for (const auto &cell : dof_handler_fg.active_cell_iterators())
      if (cell->is_locally_owned()) {
        fe_v_imm.reinit(cell);
        cell->get_dof_indices(imm_ldi);
        if (use_q0_multiplier) {
          typename DoFHandler<dim>::cell_iterator mcell(
              &cell->get_triangulation(), cell->level(), cell->index(),
              &dof_handler_mult);
          mcell->get_dof_indices(mult_ldi);
        }
        const auto &qpts = fe_v_imm.get_quadrature_points();
        for (unsigned int q = 0; q < qpts.size(); ++q) {
          points.push_back(qpts[q]);
          std::vector<double> p(n_props, 0.);
          p[0] = fe_v_imm.JxW(q);
          for (unsigned int i = 0; i < n_imm_dofs; ++i)
            p[1 + i] = fe_v_imm.shape_value(i, q);
          for (unsigned int i = 0; i < n_imm_dofs; ++i)
            p[1 + n_imm_dofs + i] = static_cast<double>(imm_ldi[i]);
          if (use_q0_multiplier)
            for (unsigned int i = 0; i < n_mult_dofs; ++i)
              p[mult_off + i] = static_cast<double>(mult_ldi[i]);
          props.push_back(std::move(p));
        }
      }
    particles.insert_global_particles(points, global_boxes, props);
    particles_built = true;
  } else {
    pcout << "Reusing " << particles.n_global_particles()
          << " coupling particles transferred across bg refinement."
          << std::endl;
  }

  pcout << "Inserted " << particles.n_global_particles()
        << " coupling particles." << std::endl;

  // First pass: compute the sparsity patterns for the coupling matrices.
  const unsigned int n_bg_dofs = fe_bg.n_dofs_per_cell();
  std::vector<types::global_dof_index> bg_ldi(n_bg_dofs);

  // In parallel the immersed triangulation is partitioned independently of
  // the background one, so a particle owned by this rank (because its
  // surrounding bg cell is locally owned) can carry immersed/multiplier
  // dof indices that are NOT in this rank's relevant_fg/relevant_mult
  // set. We must therefore use writable row IndexSets that include every
  // immersed and multiplier dof touched by a local particle. Otherwise
  // DynamicSparsityPattern::add silently drops the entry and Trilinos
  // later rejects the off-pattern write with error -1.
  IndexSet imm_writable(dof_handler_fg.n_dofs());
  imm_writable.add_indices(relevant_fg);
  IndexSet mult_writable(dof_handler_mult.n_dofs());
  mult_writable.add_indices(relevant_mult);
  {
    std::vector<types::global_dof_index> touched_imm, touched_mult;
    touched_imm.reserve(particles.n_locally_owned_particles() * n_imm_dofs);
    touched_mult.reserve(particles.n_locally_owned_particles() * n_mult_dofs);
    std::vector<types::global_dof_index> mult_tmp(n_mult_dofs);
    for (auto p = particles.begin(); p != particles.end(); ++p) {
      const auto &pp = p->get_properties();
      for (unsigned int i = 0; i < n_imm_dofs; ++i)
        touched_imm.push_back(
            static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + i]));
      mult_dofs_from_props(pp, mult_tmp);
      for (auto k : mult_tmp)
        touched_mult.push_back(k);
    }
    std::sort(touched_imm.begin(), touched_imm.end());
    touched_imm.erase(std::unique(touched_imm.begin(), touched_imm.end()),
                      touched_imm.end());
    imm_writable.add_indices(touched_imm.begin(), touched_imm.end());
    imm_writable.compress();
    std::sort(touched_mult.begin(), touched_mult.end());
    touched_mult.erase(std::unique(touched_mult.begin(), touched_mult.end()),
                       touched_mult.end());
    mult_writable.add_indices(touched_mult.begin(), touched_mult.end());
    mult_writable.compress();
  }

  // DSP for coupling_matrix (C, n_bg x n_mult): (u|_Omega_2, mu).
  DynamicSparsityPattern dsp_C(dof_handler_bg.n_dofs(),
                               dof_handler_mult.n_dofs(), relevant_bg);
  // DSP for coupling_matrix_01 (n_bg x n_imm): (u|_Omega_2, v_2).
  DynamicSparsityPattern dsp_C_01(dof_handler_bg.n_dofs(),
                                  dof_handler_fg.n_dofs(), relevant_bg);
  // DSP for coupling_matrix_10 (n_imm x n_bg): transpose of (0,1).
  DynamicSparsityPattern dsp_C_10(dof_handler_fg.n_dofs(),
                                  dof_handler_bg.n_dofs(), imm_writable);
  // DSP for bg_mult_integral_T (n_mult x n_bg). Q0 only!
  DynamicSparsityPattern dsp_bg_mult_T(dof_handler_mult.n_dofs(),
                                       dof_handler_bg.n_dofs(), mult_writable);

  {
    auto particle = particles.begin();
    while (particle != particles.end()) {
      const auto &cell = particle->get_surrounding_cell();
      typename DoFHandler<dim>::cell_iterator dh_cell(*cell, &dof_handler_bg);
      dh_cell->get_dof_indices(bg_ldi);
      const auto pic = particles.particles_in_cell(cell);

      // Collect immersed-primal and multiplier dof indices touched by
      // particles in this background cell.
      std::vector<types::global_dof_index> imm_cols;
      std::vector<types::global_dof_index> mult_cols;
      imm_cols.reserve(n_imm_dofs);
      mult_cols.reserve(n_mult_dofs);
      std::vector<types::global_dof_index> mult_tmp(n_mult_dofs);
      for (const auto &p : pic) {
        const auto &pp = p.get_properties();
        for (unsigned int i = 0; i < n_imm_dofs; ++i)
          imm_cols.push_back(
              static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + i]));
        mult_dofs_from_props(pp, mult_tmp);
        for (auto k : mult_tmp)
          mult_cols.push_back(k);
      }
      std::sort(imm_cols.begin(), imm_cols.end());
      imm_cols.erase(std::unique(imm_cols.begin(), imm_cols.end()),
                     imm_cols.end());
      std::sort(mult_cols.begin(), mult_cols.end());
      mult_cols.erase(std::unique(mult_cols.begin(), mult_cols.end()),
                      mult_cols.end());

      for (const auto bg_i : bg_ldi) {
        for (const auto mu_k : mult_cols)
          dsp_C.add(bg_i, mu_k);
        // For a Q0 multiplier the (0,1)/(1,0) blocks are NOT the plain
        // variational couplings but their projected counterparts
        // Ct W^{-1} M^T / M W^{-1} Ct^T, whose sparsity is only known once
        // bg_mult_integral_T has been assembled; they are built cellwise
        // further down, so nothing is added here.
        if (!use_q0_multiplier)
          for (const auto imm_j : imm_cols) {
            dsp_C_01.add(bg_i, imm_j);
            dsp_C_10.add(imm_j, bg_i);
          }
        else
          for (const auto mu_k : mult_cols)
            dsp_bg_mult_T.add(mu_k, bg_i);
      }
      particle = pic.end();
    }
  }

  SparsityTools::distribute_sparsity_pattern(dsp_C, owned_bg, mpi_communicator,
                                             relevant_bg);
  coupling_matrix.reinit(owned_bg, owned_mult, dsp_C, mpi_communicator);
  if (!use_q0_multiplier) {
    SparsityTools::distribute_sparsity_pattern(dsp_C_01, owned_bg,
                                               mpi_communicator, relevant_bg);
    SparsityTools::distribute_sparsity_pattern(dsp_C_10, owned_fg,
                                               mpi_communicator, imm_writable);
    coupling_matrix_01.reinit(owned_bg, owned_fg, dsp_C_01, mpi_communicator);
    coupling_matrix_10.reinit(owned_fg, owned_bg, dsp_C_10, mpi_communicator);
  }
  if (use_q0_multiplier) {
    SparsityTools::distribute_sparsity_pattern(dsp_bg_mult_T, owned_mult,
                                               mpi_communicator, mult_writable);
    bg_mult_integral_T.reinit(owned_mult, owned_bg, dsp_bg_mult_T,
                              mpi_communicator);
  }

  // Second pass: assemble coupling matrix values and (for Q1-Q1-Q1 only)
  // the in-place AL augmentation of A_bg.
  FullMatrix<double> local_C(n_bg_dofs, n_mult_dofs);
  FullMatrix<double> local_C_01(n_bg_dofs, n_imm_dofs);
  FullMatrix<double> local_C_10(n_imm_dofs, n_bg_dofs);
  FullMatrix<double> local_AL(n_bg_dofs, n_bg_dofs);
  FullMatrix<double> local_bg_mult_T(n_mult_dofs, n_bg_dofs);
  {
    auto particle = particles.begin();
    while (particle != particles.end()) {
      const auto &cell = particle->get_surrounding_cell();
      typename DoFHandler<dim>::cell_iterator dh_cell(*cell, &dof_handler_bg);
      dh_cell->get_dof_indices(bg_ldi);
      local_AL = 0;

      const auto pic = particles.particles_in_cell(cell);
      for (const auto &p : pic) {
        const auto ref = p.get_reference_location();
        const auto &pp = p.get_properties();
        const double JxW = pp[0];

        // Background shape values at the particle's reference location.
        std::vector<double> bg_vals(n_bg_dofs);
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          bg_vals[i] = fe_bg.shape_value(i, ref);

        // Q1-Q1-Q1: in-place AL augmentation via the particle quadrature.
        if (!use_q0_multiplier) {
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_bg_dofs; ++j)
              local_AL(i, j) += gamma_1_scaled * bg_vals[i] * bg_vals[j] * JxW;
        }

        // coupling_matrix (n_bg x n_mult).
        // Q0: phi_bg_i * 1 * JxW (a single mult dof per particle).
        // Q1: phi_bg_i * psi_mult_j * JxW; psi_mult_j coincides with psi_imm_j.
        std::vector<types::global_dof_index> mult_idx(n_mult_dofs);
        mult_dofs_from_props(pp, mult_idx);
        local_C = 0;
        if (use_q0_multiplier) {
          // Single mult DoF; mult shape value = 1.
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            local_C(i, 0) = bg_vals[i] * JxW;
        } else {
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_mult_dofs; ++j)
              // pp[1+j] is the shape value of the j-th imm/mult basis fn.
              local_C(i, j) = bg_vals[i] * pp[1 + j] * JxW;
        }

        // coupling_matrix_01 / _10 are on the immersed primal basis. Only
        // for the triple whose multiplier equals that space; the Q0 path
        // builds the projected versions cellwise below.
        std::vector<types::global_dof_index> imm_cols(n_imm_dofs);
        for (unsigned int j = 0; j < n_imm_dofs; ++j)
          imm_cols[j] =
              static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + j]);
        if (!use_q0_multiplier) {
          local_C_01 = 0;
          local_C_10 = 0;
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_imm_dofs; ++j) {
              const double val = bg_vals[i] * pp[1 + j] * JxW;
              local_C_01(i, j) = val;
              local_C_10(j, i) = val;
            }
        }

        // bg_mult_integral_T row K = sum_q phi_bg_i(x_q) * JxW_q (per K).
        if (use_q0_multiplier) {
          local_bg_mult_T = 0;
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            local_bg_mult_T(0, i) = bg_vals[i] * JxW;
          constraints_mult.distribute_local_to_global(local_bg_mult_T, mult_idx,
                                                      constraints_bg, bg_ldi,
                                                      bg_mult_integral_T);
        }

        constraints_bg.distribute_local_to_global(
            local_C, bg_ldi, constraints_mult, mult_idx, coupling_matrix);
        if (!use_q0_multiplier) {
          constraints_bg.distribute_local_to_global(
              local_C_01, bg_ldi, constraints_fg, imm_cols, coupling_matrix_01);
          constraints_fg.distribute_local_to_global(
              local_C_10, imm_cols, constraints_bg, bg_ldi, coupling_matrix_10);
        }
      }
      // Q1: distribute the AL augmentation (per background cell).
      if (!use_q0_multiplier)
        constraints_bg.distribute_local_to_global(local_AL, bg_ldi, A_bg);
      particle = pic.end();
    }
  }
  if (!use_q0_multiplier)
    A_bg.compress(VectorOperation::add);
  coupling_matrix.compress(VectorOperation::add);
  if (!use_q0_multiplier) {
    coupling_matrix_01.compress(VectorOperation::add);
    coupling_matrix_10.compress(VectorOperation::add);
  } else {
    bg_mult_integral_T.compress(VectorOperation::add);
  }

  // ***************************
  // Q0 path: direct assembly of the augmentation
  //   A_bg += gamma_1 * (P_h phi_i, phi_j)_{Omega_2}
  //         = gamma_1 * sum_K (1/|K|) (int_K phi_i)(int_K phi_j)
  if (use_q0_multiplier) {

    // 1. Collect the bg-DoFs that appear in S_K = {i : (int_K phi_i) != 0}
    //    for every locally-owned multiplier dof K, and build a writable
    //    bg-row IndexSet so that the augmentation entries we add at any
    //    row i are shipped to the bg rank owning that row by
    //    distribute_sparsity_pattern.
    Assert(imm_mult_integral_T.m() == dof_handler_mult.n_dofs(),
           ExcMessage("imm_mult_integral_T (M^T) has not been assembled; "
                      "add_projected_mass_fg must run before this routine."));
    std::vector<std::vector<types::global_dof_index>> S_per_K, T_per_K;
    S_per_K.reserve(owned_mult.n_elements());
    T_per_K.reserve(owned_mult.n_elements());
    std::vector<types::global_dof_index> ordered_K;
    ordered_K.reserve(owned_mult.n_elements());
    IndexSet bg_writable(dof_handler_bg.n_dofs());
    bg_writable.add_indices(relevant_bg);
    {
      std::vector<types::global_dof_index> touched_bg;
      for (const auto K : owned_mult) {
        std::vector<types::global_dof_index> S_K;
        for (auto it = bg_mult_integral_T.begin(K);
             it != bg_mult_integral_T.end(K); ++it)
          if (it->value() != 0.0)
            S_K.push_back(it->column());
        std::sort(S_K.begin(), S_K.end());
        S_K.erase(std::unique(S_K.begin(), S_K.end()), S_K.end());
        if (!S_K.empty()) {
          std::vector<types::global_dof_index> T_K;
          for (auto it = imm_mult_integral_T.begin(K);
               it != imm_mult_integral_T.end(K); ++it)
            if (it->value() != 0.0)
              T_K.push_back(it->column());
          std::sort(T_K.begin(), T_K.end());
          T_K.erase(std::unique(T_K.begin(), T_K.end()), T_K.end());
          for (auto i : S_K)
            touched_bg.push_back(i);
          ordered_K.push_back(K);
          S_per_K.push_back(std::move(S_K));
          T_per_K.push_back(std::move(T_K));
        }
      }
      std::sort(touched_bg.begin(), touched_bg.end());
      touched_bg.erase(std::unique(touched_bg.begin(), touched_bg.end()),
                       touched_bg.end());
      bg_writable.add_indices(touched_bg.begin(), touched_bg.end());
      bg_writable.compress();
    }

    // 2. Build the extended A_bg DSP: original FE stencil (preserved by
    //    re-running make_sparsity_pattern with constraints_bg) plus the
    //    S_K x S_K pairs from every locally-owned K. The K-driven
    //    contributions live on rows that may be off-rank for this MPI
    //    process; the writable IndexSet built above guarantees they are
    //    forwarded to their owners by distribute_sparsity_pattern.
    DynamicSparsityPattern dsp_A_bg_ext(dof_handler_bg.n_dofs(),
                                        dof_handler_bg.n_dofs(), bg_writable);
    DoFTools::make_sparsity_pattern(dof_handler_bg, dsp_A_bg_ext,
                                    constraints_bg, false);
    DynamicSparsityPattern dsp_C_01_proj(dof_handler_bg.n_dofs(),
                                         dof_handler_fg.n_dofs(), bg_writable);
    DynamicSparsityPattern dsp_C_10_proj(dof_handler_fg.n_dofs(),
                                         dof_handler_bg.n_dofs(), imm_writable);
    for (std::size_t idx = 0; idx < ordered_K.size(); ++idx) {
      const auto &S_K = S_per_K[idx];
      const auto &T_K = T_per_K[idx];
      for (const auto i : S_K) {
        for (const auto j : S_K)
          dsp_A_bg_ext.add(i, j);
        for (const auto j : T_K) {
          dsp_C_01_proj.add(i, j);
          dsp_C_10_proj.add(j, i);
        }
      }
    }
    SparsityTools::distribute_sparsity_pattern(dsp_A_bg_ext, owned_bg,
                                               mpi_communicator, bg_writable);
    SparsityTools::distribute_sparsity_pattern(dsp_C_01_proj, owned_bg,
                                               mpi_communicator, bg_writable);
    SparsityTools::distribute_sparsity_pattern(dsp_C_10_proj, owned_fg,
                                               mpi_communicator, imm_writable);

    // 3. Snapshot the current A_bg (== K_bg, the bulk stiffness with BCs
    //    but no augmentation yet) and reinit A_bg with the extended
    //    sparsity. We use K_bg_pre_aug (already taken in solve()) as the
    //    snapshot, since it is the same object after the assemble_bg()
    //    step and has not been touched.
    A_bg.reinit(owned_bg, owned_bg, dsp_A_bg_ext, mpi_communicator);
    for (auto it = K_bg_pre_aug.begin(); it != K_bg_pre_aug.end(); ++it)
      A_bg.set(it->row(), it->column(), it->value());
    A_bg.compress(VectorOperation::insert);
    coupling_matrix_01.reinit(owned_bg, owned_fg, dsp_C_01_proj,
                              mpi_communicator);
    coupling_matrix_10.reinit(owned_fg, owned_bg, dsp_C_10_proj,
                              mpi_communicator);

    // 4. K-driven outer-product augmentation:
    //    A_bg(i,j) += gamma_1 / |K| * c_K[i] * c_K[j]   for (i,j) in S_K x S_K
    //    where c_K[i] = int_K phi_i and |K| = (mu_K, mu_K) is the diagonal
    //    entry of mass_matrix_mult at row K.

    // Time ONLY the operator-form (1,1)-block augmentation (the rank-1
    // outer-product fill of A_bg), excluding the coupling-matrix assembly
    // above, so that it can be compared fairly against the algebraic
    // (1,1)-block assembly reported under "Algebraic bg (1,1) block ...".
    {
      TimerOutput::Scope t_aug(
          computing_timer,
          "Operator-form bg (1,1) block augmentation only (particles)");

      for (std::size_t idx = 0; idx < ordered_K.size(); ++idx) {
        const auto K = ordered_K[idx];
        const auto &S_K = S_per_K[idx];
        const auto &T_K = T_per_K[idx];

        // Gather c_K (sparse row K of bg_mult_integral_T) and m_K (row K of
        // imm_mult_integral_T) into dense maps dof -> coefficient.
        std::map<types::global_dof_index, double> c_K;
        for (auto it = bg_mult_integral_T.begin(K);
             it != bg_mult_integral_T.end(K); ++it)
          c_K[it->column()] = it->value();
        std::map<types::global_dof_index, double> m_K;
        for (auto it = imm_mult_integral_T.begin(K);
             it != imm_mult_integral_T.end(K); ++it)
          m_K[it->column()] = it->value();

        const double abs_K = mass_matrix_mult.diag_element(K);
        Assert(abs_K > 0.0, ExcMessage("|K| is non-positive."));
        const double inv_K = 1. / abs_K;
        for (const auto i : S_K) {
          const double ci = c_K[i];
          for (const auto j : S_K)
            A_bg.add(i, j, gamma_1_scaled * inv_K * ci * c_K[j]);
          // Projected off-diagonals; the operator applies the -gamma_1 /
          // -gamma_2 factors, so these are stored unscaled.
          for (const auto j : T_K) {
            const double v = inv_K * ci * m_K[j];
            coupling_matrix_01.add(i, j, v);
            coupling_matrix_10.add(j, i, v);
          }
        }
      }
      A_bg.compress(VectorOperation::add);
      coupling_matrix_01.compress(VectorOperation::add);
      coupling_matrix_10.compress(VectorOperation::add);
    }
  }
}

// Algebraic counterpart of the operator-form (1,1) block.
//
// The solver augments the (1,1) block in place at the variational level
// (particle quadrature for Q1-Q1-Q1, direct rank-1 assembly for Q1-Bubble-Q0),
// ending up with A_bg = K + (operator-form augmentation).
// Here we instead assemble the *whole* (1,1) block algebraically as the
// explicit sparse matrix
//     A11_alg = A + gamma_1 * C W^{-1} C^T,
// where A = K_bg_pre_aug is the bare bulk stiffness (with BCs) snapshot taken
// in solve(), C = coupling_matrix (n_bg x n_mult) and W = mass_matrix_mult
// (n_mult x n_mult). The entire assembly (triple product + union sparsity
// pattern + addition of A) is timed under the scope
// "Algebraic bg (1,1) block (A + gamma_1 C W^{-1} C^T)", so its wall time can
// be compared against the operator-form (1,1)-block augmentation reported
// under "Operator-form bg (1,1) block augmentation only (particles)" in the
// timer summary (which excludes the coupling-matrix assembly, for a fair
// comparison).
//
// For the Q1-Bubble-Q0 triple W is diagonal (entries |K|), so W^{-1} is exact
// and C^T is available as bg_mult_integral_T. The assembled block is verified
// against the in-place augmented A_bg via a matrix-vector comparison and then
// discarded (the algebraic matrix is not used by the solver).

template <int dim>
void EllipticInterfaceDLMParallel<dim>::compute_algebraic_augmentation(
    double gamma_1_scaled) {
  AssertThrow(
      use_q0_multiplier,
      ExcMessage("Algebraic augmentation comparison is currently only "
                 "implemented for the FE triple \"Q1-Bubble-Q0\", whose "
                 "multiplier mass matrix W is diagonal (so W^{-1} is exact "
                 "and C^T is available as bg_mult_integral_T). For "
                 "\"Q1-Q1-Q1\" the multiplier mass is non-diagonal and "
                 "C^T M^{-1} C is not a sparse matrix."));

  // diag(V) entries that turn C * diag(V) * C^T into gamma_1 * C W^{-1} C^T:
  // V[K] = gamma_1 / W[K,K]. V lives on the inner (multiplier) index and must
  // be distributed exactly like the rows of the right factor (C^T).
  LA::MPI::Vector winv(owned_mult, mpi_communicator);
  for (const auto K : owned_mult) {
    const double abs_K = mass_matrix_mult.diag_element(K);
    Assert(abs_K > 0.0, ExcMessage("|K| is non-positive."));
    winv[K] = gamma_1_scaled / abs_K;
  }
  winv.compress(VectorOperation::insert);

  // Assemble the whole (1,1) block A11_alg = A + gamma_1 * C W^{-1} C^T.
  // The full assembly is timed: the triple product, the union of the bulk
  // stiffness stencil with the triple-product fill-in, and the final add.
  LA::MPI::SparseMatrix A11_alg;
  {
    TimerOutput::Scope t(computing_timer,
                         "Algebraic bg (1,1) block (A + gamma_1 C W^{-1} C^T)");

    // 1. Triple product aug_alg = C * diag(winv) * C^T = gamma_1 C W^{-1} C^T.
    //    bg_mult_integral_T (n_mult x n_bg) is exactly C^T for the Q0
    //    multiplier. Both factors have row map owned_bg / owned_mult, so the
    //    result has range owned_bg (one entry block per owned bg row).
    LA::MPI::SparseMatrix aug_alg;
    coupling_matrix.mmult(aug_alg, bg_mult_integral_T, winv);

    // 2. Union sparsity pattern of the bulk stiffness A (= K_bg_pre_aug) and
    //    the triple product. Both matrices are row-distributed over owned_bg,
    //    so we only ever write to locally-owned rows.
    DynamicSparsityPattern dsp(dof_handler_bg.n_dofs(), dof_handler_bg.n_dofs(),
                               relevant_bg);
    for (auto it = K_bg_pre_aug.begin(); it != K_bg_pre_aug.end(); ++it)
      dsp.add(it->row(), it->column());
    for (auto it = aug_alg.begin(); it != aug_alg.end(); ++it)
      dsp.add(it->row(), it->column());
    SparsityTools::distribute_sparsity_pattern(dsp, owned_bg, mpi_communicator,
                                               relevant_bg);

    // 3. A11_alg = A, then add the triple product in place.
    A11_alg.reinit(owned_bg, owned_bg, dsp, mpi_communicator);
    for (auto it = K_bg_pre_aug.begin(); it != K_bg_pre_aug.end(); ++it)
      A11_alg.set(it->row(), it->column(), it->value());
    A11_alg.compress(VectorOperation::insert);
    for (auto it = aug_alg.begin(); it != aug_alg.end(); ++it)
      A11_alg.add(it->row(), it->column(), it->value());
    A11_alg.compress(VectorOperation::add);
  }

  // Correctness check, independent of sparsity patterns: compare the action
  // of the algebraically assembled block A11_alg against the operator-form
  // (1,1) block A_bg (which already holds K + operator-form augmentation).
  LA::MPI::Vector r(owned_bg, mpi_communicator);
  for (const auto i : owned_bg)
    r[i] = static_cast<double>((i % 7) + 1); // deterministic, nonzero
  r.compress(VectorOperation::insert);

  LA::MPI::Vector y_alg(owned_bg, mpi_communicator);
  LA::MPI::Vector y_op(owned_bg, mpi_communicator);
  A11_alg.vmult(y_alg, r); // y_alg = (A + gamma_1 C W^{-1} C^T) r
  A_bg.vmult(y_op, r);     // y_op  = (K + operator-form augmentation) r

  LA::MPI::Vector diff(owned_bg, mpi_communicator);
  diff = y_op;
  diff -= y_alg;
  const double rel_err =
      (y_op.l2_norm() > 0.0) ? diff.l2_norm() / y_op.l2_norm() : diff.l2_norm();

  pcout << "Algebraic (1,1) block: assembled A + gamma_1 * C W^{-1} C^T with "
        << A11_alg.n_nonzero_elements() << " nonzeros; relative l2 mismatch "
        << "vs operator-form (1,1) block = " << rel_err
        << " (see timer summary for the wall-time comparison)." << std::endl;
}

template <int dim> unsigned int EllipticInterfaceDLMParallel<dim>::solve() {
  // h-scaling absorbed into gamma's (operator form W = h^2 M).
  const double h_immersed = GridTools::maximal_cell_diameter(*tria_fg);
  const double inv_h2 = 1. / (h_immersed * h_immersed);
  const double gamma_1 = parameters.gamma_AL_background * inv_h2;
  const double gamma_2 = parameters.gamma_AL_immersed * inv_h2;
  last_gamma_1 = gamma_1;

  // Assemble all matrices for this cycle.
  assemble_bg();
  assemble_fg(gamma_2);
  assemble_mass_fg();
  assemble_mass_mult();
  // Q0 multiplier only: completes the (1,1) block with the projected AL mass
  // gamma_2 * M W^{-1} M^T and builds M^T for the projected off-diagonal
  // blocks assembled below. No-op for Q1-Q1-Q1.
  add_projected_mass_fg(gamma_2);

  // Snapshot the bg stiffness BEFORE the AL augmentation: we will use it
  // in compute_augmentation_consistency() to extract R_u acting on u_bg
  // as (A_bg - K_bg_pre_aug) / gamma_1.
  K_bg_pre_aug.reinit(A_bg);
  K_bg_pre_aug.copy_from(A_bg);

  // Optional dump of K = A_bg BEFORE the AL augmentation. Done here so the
  // file contains the bare bulk stiffness with Dirichlet BCs but without
  // the gamma * C M^{-1} C^T contribution that the next call adds in place.
  if (parameters.export_matrices_for_matlab && this_mpi_process == 0) {
    AssertThrow(n_mpi_processes == 1,
                ExcMessage("Matrix export to MATLAB is only supported in "
                           "serial (single MPI process)."));
    std::filesystem::create_directories(parameters.matrix_export_directory);
    static unsigned int matrix_export_cycle = 0;
    const unsigned int cycle_idx = matrix_export_cycle++;
    auto write_triplets =
        [&, cycle_idx](const LA::MPI::SparseMatrix &A, const std::string &name,
                       const unsigned int n_rows, const unsigned int n_cols) {
          const std::string fname =
              parameters.matrix_export_directory + "/" + name + "_cycle" +
              Utilities::int_to_string(cycle_idx, 2) + ".txt";
          std::ofstream out(fname);
          out.precision(16);
          out << std::scientific;
          for (auto it = A.begin(); it != A.end(); ++it)
            out << (it->row() + 1) << ' ' << (it->column() + 1) << ' '
                << it->value() << '\n';
          out << n_rows << ' ' << n_cols << " 0\n";
          pcout << "Wrote " << fname << " (" << n_rows << " x " << n_cols << ")"
                << std::endl;
        };
    write_triplets(A_bg, "K", dof_handler_bg.n_dofs(), dof_handler_bg.n_dofs());
    write_triplets(A_fg, "A_fg", dof_handler_fg.n_dofs(),
                   dof_handler_fg.n_dofs());
    write_triplets(mass_matrix_fg, "M", dof_handler_fg.n_dofs(),
                   dof_handler_mult.n_dofs());
    write_triplets(mass_matrix_mult, "W", dof_handler_mult.n_dofs(),
                   dof_handler_mult.n_dofs());
  }

  assemble_coupling_and_augment_bg(gamma_1);

  // Optional: also assemble the whole (1,1) block algebraically (explicit
  // sparse matrix A + gamma_1 * C W^{-1} C^T) and time it, to compare against
  // the operator-form (1,1) block just assembled in place on A_bg.
  if (parameters.compare_algebraic_augmentation)
    compute_algebraic_augmentation(gamma_1);

  if (parameters.export_matrices_for_matlab && this_mpi_process == 0) {
    static unsigned int matrix_export_cycle_post = 0;
    const unsigned int cycle_idx = matrix_export_cycle_post++;
    auto write_triplets =
        [&, cycle_idx](const LA::MPI::SparseMatrix &A, const std::string &name,
                       const unsigned int n_rows, const unsigned int n_cols) {
          const std::string fname =
              parameters.matrix_export_directory + "/" + name + "_cycle" +
              Utilities::int_to_string(cycle_idx, 2) + ".txt";
          std::ofstream out(fname);
          out.precision(16);
          out << std::scientific;
          for (auto it = A.begin(); it != A.end(); ++it)
            out << (it->row() + 1) << ' ' << (it->column() + 1) << ' '
                << it->value() << '\n';
          out << n_rows << ' ' << n_cols << " 0\n";
          pcout << "Wrote " << fname << " (" << n_rows << " x " << n_cols << ")"
                << std::endl;
        };
    write_triplets(coupling_matrix, "C", dof_handler_bg.n_dofs(),
                   dof_handler_mult.n_dofs());
    write_triplets(coupling_matrix_01, "C_01", dof_handler_bg.n_dofs(),
                   dof_handler_fg.n_dofs());
    write_triplets(coupling_matrix_10, "C_10", dof_handler_fg.n_dofs(),
                   dof_handler_bg.n_dofs());
    // Directly assembled augmented blocks (operator form): K_aug is A_bg
    // after augment_bg added gamma_1 * C M^{-1} C^T in place, A_fg_aug is
    // A_fg + gamma_2 * M assembled in assemble_fg. h is recorded so the
    // notebook can use the exact value baked into gamma_{1,2} = c / h^2.
    write_triplets(A_bg, "K_aug", dof_handler_bg.n_dofs(),
                   dof_handler_bg.n_dofs());
    write_triplets(A_fg_plus_scaled_M, "A_fg_aug", dof_handler_fg.n_dofs(),
                   dof_handler_fg.n_dofs());
    const std::string h_fname = parameters.matrix_export_directory +
                                "/h_cycle" +
                                Utilities::int_to_string(cycle_idx, 2) + ".txt";
    std::ofstream h_out(h_fname);
    h_out.precision(16);
    h_out << std::scientific << h_immersed << '\n';
    pcout << "Wrote " << h_fname << " (h_immersed = " << h_immersed
          << ", gamma_1 = " << gamma_1 << ", gamma_2 = " << gamma_2 << ")"
          << std::endl;
  }

  // Build invW = W^{-1} action via CG with a Jacobi preconditioner on the
  // multiplier mass matrix. For FE_DGQ(0) the matrix is diagonal and CG
  // converges in a single iteration.
  LA::MPI::PreconditionJacobi mass_prec;
  mass_prec.initialize(mass_matrix_mult);
  using InvW_t = ALParallel::InvMassOp<LA::MPI::Vector, LA::MPI::SparseMatrix,
                                       LA::MPI::PreconditionJacobi>;
  InvW_t invW_op(mass_matrix_mult, mass_prec, /*max_it*/ 1000, /*tol*/ 1e-12,
                 owned_mult, mpi_communicator);

  // LinearOperators only for single matrices (no composition).
  auto A11_aug_op = linear_operator<LA::MPI::Vector>(A_bg);
  auto A22_aug_op = linear_operator<LA::MPI::Vector>(A_fg_plus_scaled_M);

  ALParallel::System3x3<LA::MPI::Vector, LA::MPI::BlockVector,
                        LA::MPI::SparseMatrix, InvW_t>
      system_operator{A_bg,
                      A_fg_plus_scaled_M,
                      mass_matrix_fg,
                      coupling_matrix,
                      coupling_matrix_01,
                      coupling_matrix_10,
                      gamma_1,
                      gamma_2,
                      owned_bg,
                      owned_fg,
                      owned_mult,
                      mpi_communicator};

  // AMG preconditioners.
  // For A_bg: Dirichlet BCs remove the constant null space -> default init.
  // For A_fg_plus_scaled_M = (beta_2-beta_1)*K_fg + gamma_2/h^2 * M_fg:
  // the matrix is SPD but K_fg alone is singular (no BCs on the immersed
  // domain), so the constant function is a near-null vector at coarse cycles
  // or when gamma_2/h^2 is small. Providing the constant modes to ML/MueLu
  // helps AMG coarsening on those near-singular low-frequency components.
  LA::MPI::PreconditionAMG amg_A11, amg_A22;
  amg_A11.initialize(A_bg);
  {
    const std::vector<std::vector<bool>> constant_modes_fg =
        DoFTools::extract_constant_modes(dof_handler_fg, ComponentMask());
    LA::MPI::PreconditionAMG::AdditionalData amg_data_a22;
    amg_data_a22.constant_modes = constant_modes_fg;
    amg_A22.initialize(A_fg_plus_scaled_M, amg_data_a22);
  }

  typename SolverFGMRES<LA::MPI::BlockVector>::AdditionalData data_fgmres;
  data_fgmres.max_basis_size = 30;
  SolverFGMRES<LA::MPI::BlockVector> solver_fgmres(
      parameters.outer_solver_control, data_fgmres);

  system_rhs_block.block(2) = 0.;
  system_solution_block = 0.;

  TimerOutput::Scope t(computing_timer, "Solve system");
  if (parameters.use_modified_AL_preconditioner) {
    AssertThrow(parameters.gamma_AL_immersed <= 20.,
                ExcMessage("gamma solid too large for modified AL."));
    AssertThrow(std::abs(parameters.gamma_AL_immersed -
                         parameters.gamma_AL_background) > 1e-1,
                ExcMessage("Modified AL requires gamma_1 != gamma_2."));

    // Use a dedicated inner CG instance for the (1,1) block A_bg so that its
    // iteration count can be tracked independently of the (2,2) block solve.
    std::unique_ptr<SolverCG<LA::MPI::Vector>> inner_cg_A11;
    std::unique_ptr<SolverCG<LA::MPI::Vector>> inner_cg_A22;
    if (parameters.use_fixed_iterations) {
      inner_cg_A11 = std::make_unique<SolverCG<LA::MPI::Vector>>(
          parameters.iteration_number_control);
      inner_cg_A22 = std::make_unique<SolverCG<LA::MPI::Vector>>(
          parameters.iteration_number_control);
    } else {
      inner_cg_A11 = std::make_unique<SolverCG<LA::MPI::Vector>>(
          parameters.inner_solver_control);
      inner_cg_A22 = std::make_unique<SolverCG<LA::MPI::Vector>>(
          parameters.inner_solver_control);
    }

    // Count the inner PCG iterations spent inverting the (1,1) block A_bg.
    // The slot fires on every CG check(): step 0 is the initial residual
    // check (one per (1,1)-block solve), steps > 0 are actual iterations.
    unsigned int inner_iters_A11 = 0;
    unsigned int n_A11_applications = 0;
    auto inner_conn_A11 = inner_cg_A11->connect(
        [&inner_iters_A11, &n_A11_applications](
            const unsigned int step, const double /*res*/,
            const LA::MPI::Vector & /*current*/) -> SolverControl::State {
          if (step == 0)
            ++n_A11_applications;
          else
            ++inner_iters_A11;
          return SolverControl::success;
        });

    auto A11_aug_inv = inverse_operator(A11_aug_op, *inner_cg_A11, amg_A11);
    auto A22_aug_inv = inverse_operator(A22_aug_op, *inner_cg_A22, amg_A22);

    ALParallel::ModifiedALPreconditioner<
        LA::MPI::Vector, LA::MPI::BlockVector, LA::MPI::SparseMatrix, InvW_t,
        decltype(A11_aug_inv), decltype(A22_aug_inv)>
        prec_AL{mass_matrix_fg, coupling_matrix, invW_op,  gamma_1,
                A11_aug_inv,    A22_aug_inv,     owned_bg, owned_fg,
                owned_mult,     mpi_communicator};

    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, prec_AL);

    inner_conn_A11.disconnect();
    last_inner_iters_A11 = inner_iters_A11;
    last_n_A11_applications = n_A11_applications;
  } else {
    AssertThrow(parameters.gamma_AL_background > 1.,
                ExcMessage("gamma probably too small for ideal AL."));
    AssertThrow(
        std::abs(parameters.gamma_AL_background -
                 parameters.gamma_AL_immersed) < 1e-12,
        ExcMessage("In the ideal case, gamma_1 == gamma_2 is required."));

    pcout << "\t *** USING IDEAL AL PRECONDITIONER (test only) ***"
          << std::endl;

    ALParallel::Block2x2<LA::MPI::Vector, LA::MPI::BlockVector,
                         LA::MPI::SparseMatrix, InvW_t>
        Aug_mat{A_bg,           A_fg_plus_scaled_M,
                mass_matrix_fg, coupling_matrix_01,
                invW_op,        gamma_1,
                gamma_2,        owned_bg,
                owned_fg,       mpi_communicator};

    ALParallel::BlockDiagPrec<LA::MPI::BlockVector, LA::MPI::PreconditionAMG,
                              LA::MPI::PreconditionAMG>
        prec_aug{amg_A11, amg_A22};

    SolverCG<LA::MPI::BlockVector> solver_block(
        parameters.inner_solver_control);
    LinearOperator<LA::MPI::BlockVector> Aug_op;
    Aug_op.vmult = [&Aug_mat](LA::MPI::BlockVector &d,
                              const LA::MPI::BlockVector &s) {
      Aug_mat.vmult(d, s);
    };
    Aug_op.vmult_add = [&Aug_mat](LA::MPI::BlockVector &d,
                                  const LA::MPI::BlockVector &s) {
      LA::MPI::BlockVector tmp;
      tmp.reinit(d);
      Aug_mat.vmult(tmp, s);
      d += tmp;
    };
    Aug_op.Tvmult = Aug_op.vmult;
    Aug_op.Tvmult_add = Aug_op.vmult_add;
    Aug_op.reinit_range_vector = [this](LA::MPI::BlockVector &v, bool) {
      v.reinit(std::vector<IndexSet>{owned_bg, owned_fg}, mpi_communicator);
    };
    Aug_op.reinit_domain_vector = Aug_op.reinit_range_vector;

    auto Aug_inv = inverse_operator(Aug_op, solver_block, prec_aug);

    ALParallel::IdealALPreconditioner<LA::MPI::Vector, LA::MPI::BlockVector,
                                      LA::MPI::SparseMatrix, InvW_t,
                                      decltype(Aug_inv)>
        ideal_prec{mass_matrix_fg, coupling_matrix, invW_op,
                   gamma_1,        Aug_inv,         owned_bg,
                   owned_fg,       owned_mult,      mpi_communicator};

    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, ideal_prec);

    // The ideal AL preconditioner inverts the full 2x2 augmented block, not
    // the (1,1) block alone, so the per-(1,1)-block counter does not apply.
    last_inner_iters_A11 = 0;
    last_n_A11_applications = 0;
  }

  constraints_bg.distribute(system_solution_block.block(0));
  constraints_fg.distribute(system_solution_block.block(1));
  constraints_mult.distribute(system_solution_block.block(2));

  const unsigned int n_outer = parameters.outer_solver_control.last_step();
  pcout << "Solved in " << n_outer << " outer FGMRES iterations." << std::endl;

// The AL augmentation is consistent, so these norms must not depend on
// gamma_1 / gamma_2 (only the iteration counts may). Printing them makes
// that easy to verify when changing the AL parameters.
#ifdef DEBUG
  pcout << "||u_1|| = " << system_solution_block.block(0).l2_norm()
        << "   ||u_2|| = " << system_solution_block.block(1).l2_norm()
        << "   ||lambda|| = " << system_solution_block.block(2).l2_norm()
        << std::endl;
#endif

  // Average number of inner (P)CG iterations performed to invert the (1,1)
  // block A_bg of the preconditioner (mirrors the inner_cg/outer column of
  // immersed_laplace.cc). Reported as iterations per (1,1)-block solve.
  const double avg_inner_A11 =
      (last_n_A11_applications == 0)
          ? 0.0
          : double(last_inner_iters_A11) / double(last_n_A11_applications);
  pcout << "Average inner PCG iterations to invert the (1,1) block: "
        << avg_inner_A11 << " (" << last_inner_iters_A11 << " total over "
        << last_n_A11_applications << " solves)." << std::endl;

  convergence_table.add_value("cells", tria_bg->n_global_active_cells());
  convergence_table.add_value("DoF bg", dof_handler_bg.n_dofs());
  convergence_table.add_value("DoF fg", dof_handler_fg.n_dofs());
  convergence_table.add_value("DoF mult", dof_handler_mult.n_dofs());
  convergence_table.add_value("gamma1/h^2", gamma_1);
  if (parameters.use_modified_AL_preconditioner)
    convergence_table.add_value("gamma2/h^2", gamma_2);
  convergence_table.add_value("outer it", n_outer);
  convergence_table.add_value("inner A11 tot", last_inner_iters_A11);
  convergence_table.add_value("inner A11/solve", avg_inner_A11);
  convergence_table.set_precision("inner A11/solve", 1);
  return n_outer;
}

template <int dim> void EllipticInterfaceDLMParallel<dim>::compute_errors() {
  TimerOutput::Scope t(computing_timer, "Compute errors");

  // Ghosted copies of the relevant solution blocks.
  LA::MPI::Vector u_bg_ghosted(owned_bg, relevant_bg, mpi_communicator);
  u_bg_ghosted = system_solution_block.block(0);
  LA::MPI::Vector u_fg_ghosted(owned_fg, relevant_fg, mpi_communicator);
  u_fg_ghosted = system_solution_block.block(1);

  // Quadrature: use a degree comfortably higher than the FE degree so the
  // error norms are not dominated by quadrature truncation.
  const Quadrature<dim> quad_bg = make_quadrature(fe_bg_ptr->degree + 2);
  const Quadrature<dim> quad_fg = make_quadrature(fe_fg_ptr->degree + 2);

  // Errors of u_bg (block 0): u is defined globally and equals the manufactured
  // solution on the whole background domain Omega.
  Vector<double> cellwise_l2_bg(tria_bg->n_active_cells());
  Vector<double> cellwise_h1_bg(tria_bg->n_active_cells());
  VectorTools::integrate_difference(*mapping_ptr, dof_handler_bg, u_bg_ghosted,
                                    parameters.exact_solution, cellwise_l2_bg,
                                    quad_bg, VectorTools::L2_norm);
  const double e_l2_bg = VectorTools::compute_global_error(
      *tria_bg, cellwise_l2_bg, VectorTools::L2_norm);
  VectorTools::integrate_difference(*mapping_ptr, dof_handler_bg, u_bg_ghosted,
                                    parameters.exact_solution, cellwise_h1_bg,
                                    quad_bg, VectorTools::H1_seminorm);
  const double e_h1_bg = VectorTools::compute_global_error(
      *tria_bg, cellwise_h1_bg, VectorTools::H1_seminorm);

  // Errors of u_imm (block 1): the immersed primal variable is the restriction
  // of u to Omega_2.
  Vector<double> cellwise_l2_fg(tria_fg->n_active_cells());
  Vector<double> cellwise_h1_fg(tria_fg->n_active_cells());
  VectorTools::integrate_difference(
      *mapping_ptr, dof_handler_fg, u_fg_ghosted, parameters.exact_solution_imm,
      cellwise_l2_fg, quad_fg, VectorTools::L2_norm);
  const double e_l2_fg = VectorTools::compute_global_error(
      *tria_fg, cellwise_l2_fg, VectorTools::L2_norm);
  VectorTools::integrate_difference(
      *mapping_ptr, dof_handler_fg, u_fg_ghosted, parameters.exact_solution_imm,
      cellwise_h1_fg, quad_fg, VectorTools::H1_seminorm);
  const double e_h1_fg = VectorTools::compute_global_error(
      *tria_fg, cellwise_h1_fg, VectorTools::H1_seminorm);

  const double h_bg = GridTools::maximal_cell_diameter(*tria_bg);

  convergence_table.add_value("h_bg", h_bg);
  convergence_table.add_value("L2_bg", e_l2_bg);
  convergence_table.add_value("H1s_bg", e_h1_bg);

  convergence_table.set_scientific("h_bg", true);
  convergence_table.set_scientific("L2_bg", true);
  convergence_table.set_scientific("H1s_bg", true);

  convergence_table.set_precision("h_bg", 3);
  convergence_table.set_precision("L2_bg", 3);
  convergence_table.set_precision("H1s_bg", 3);

  // Rates with respect to the background h (halved at every cycle).
  convergence_table.evaluate_convergence_rates(
      "L2_bg", "h_bg", ConvergenceTable::reduction_rate_log2, 1);
  convergence_table.evaluate_convergence_rates(
      "H1s_bg", "h_bg", ConvergenceTable::reduction_rate_log2, 1);

  pcout << "  errors: L2_bg=" << e_l2_bg << "  H1s_bg=" << e_h1_bg
        << "  L2_imm=" << e_l2_fg << "  H1s_imm=" << e_h1_fg << std::endl;
}

// Consistency of the operator-form AL augmentation.
//
// The augmentation actually added in place to A_bg is one of
//   Q1-Q1-Q1:      gamma_1 * R_u   with R_u_ij = sum_q phi_i(x_q) phi_j(x_q)
//   JxW_q
//                  (particle-quadrature approximation of (phi_i,
//                  phi_j)_{Omega_2})
//   Q1-Bubble-Q0:  gamma_1 * (P_h phi_i, phi_j)_{Omega_2}, assembled directly.
//
// In both cases the "ideal" (matrix-form) augmentation that would result
// from eliminating the multiplier algebraically is
//   gamma_1 * C W^{-1} C^T,    C := coupling_matrix     (n_bg x n_mult),
//                              W := mass_matrix_mult    (n_mult x n_mult).
// For the Q0 multiplier, since (P_h u, v)_{Omega_2} = (C W^{-1} C^T u, v)
// holds exactly, the consistency residual is zero up to solver tolerance.
//
// We extract R_u via (A_bg - K_bg_pre_aug) / gamma_1 (matrix-free, via two
// vmults on u_bg) and form C W^{-1} C^T u_bg by solving W z = C^T u_bg
// with CG + Jacobi preconditioning (a single iteration for FE_DGQ(0)). The
// ell^2 norm of the difference is pushed to the convergence table together
// with its log2 rate vs h_bg.

template <int dim>
void EllipticInterfaceDLMParallel<dim>::compute_augmentation_consistency() {
  TimerOutput::Scope t(computing_timer, "Augmentation consistency");

  Assert(last_gamma_1 > 0.0,
         ExcMessage("compute_augmentation_consistency() must be called after "
                    "solve()."));

  // Ghosted copy of u_bg (block 0 of the solution).
  LA::MPI::Vector u_bg(owned_bg, relevant_bg, mpi_communicator);
  u_bg = system_solution_block.block(0);

  // R_u * u_bg = (A_bg * u_bg - K_bg_pre_aug * u_bg) / gamma_1.
  LA::MPI::Vector A_u(owned_bg, mpi_communicator);
  LA::MPI::Vector K_u(owned_bg, mpi_communicator);
  A_bg.vmult(A_u, u_bg);
  K_bg_pre_aug.vmult(K_u, u_bg);
  LA::MPI::Vector R_u_x(owned_bg, mpi_communicator);
  R_u_x = A_u;
  R_u_x -= K_u;
  R_u_x /= last_gamma_1;

  // Ideal augmentation acting on u_bg:
  //   w = C^T u_bg        (in multiplier space)
  //   W z = w             (multiplier mass solve)
  //   y = C z             (back to bg)
  LA::MPI::Vector w(owned_mult, mpi_communicator);
  coupling_matrix.Tvmult(w, u_bg);

  LA::MPI::Vector z(owned_mult, mpi_communicator);
  {
    SolverControl mass_solver_control(2000,
                                      1e-12 * std::max(w.l2_norm(), 1e-30));
    SolverCG<LA::MPI::Vector> cg(mass_solver_control);
    TrilinosWrappers::PreconditionJacobi prec_W;
    prec_W.initialize(mass_matrix_mult);
    cg.solve(mass_matrix_mult, z, w, prec_W);
  }
  LA::MPI::Vector z_ghosted(owned_mult, relevant_mult, mpi_communicator);
  z_ghosted = z;
  LA::MPI::Vector ideal_x(owned_bg, mpi_communicator);
  coupling_matrix.vmult(ideal_x, z_ghosted);

  // Difference and its ell^2 norm.
  LA::MPI::Vector diff(owned_bg, mpi_communicator);
  diff = R_u_x;
  diff -= ideal_x;
  const double diff_l2 = diff.l2_norm();
  const double R_u_l2 = R_u_x.l2_norm();
  const double rel = (R_u_l2 > 0.0) ? diff_l2 / R_u_l2 : 0.0;

  // h^{-2}-scaled version of the consistency quantity (the stabilizing term
  // ||(R_u - C W^{-1} C^T) u||_{R^n} is rescaled by 1/h^2).
  const double h_bg = GridTools::maximal_cell_diameter(*tria_bg);
  const double diff_l2_scaled = diff_l2 / (h_bg * h_bg);

  pcout << "  augmentation: ||R_u u||=" << R_u_l2
        << "   ||(R_u - C W^{-1} C^T) u||=" << diff_l2 << "   relative=" << rel
        << "   h^-2*||.||=" << diff_l2_scaled << std::endl;

#ifdef FALSE
  convergence_table.add_value("Aug_diff", diff_l2);
  convergence_table.set_scientific("Aug_diff", true);
  convergence_table.set_precision("Aug_diff", 3);
  convergence_table.evaluate_convergence_rates(
      "Aug_diff", "h_bg", ConvergenceTable::reduction_rate_log2, 1);

  convergence_table.add_value("h^-2*Aug_diff", diff_l2_scaled);
  convergence_table.set_scientific("h^-2*Aug_diff", true);
  convergence_table.set_precision("h^-2*Aug_diff", 3);
  convergence_table.evaluate_convergence_rates(
      "h^-2*Aug_diff", "h_bg", ConvergenceTable::reduction_rate_log2, 1);
#endif
}

template <int dim>
void EllipticInterfaceDLMParallel<dim>::output_results(
    unsigned int cycle) const {
  TimerOutput::Scope t(computing_timer, "Output results");
  if (tria_bg->n_global_active_cells() >= 1000000)
    return;

  LA::MPI::Vector u_bg_ghosted(owned_bg, relevant_bg, mpi_communicator);
  u_bg_ghosted = system_solution_block.block(0);
  LA::MPI::Vector u_fg_ghosted(owned_fg, relevant_fg, mpi_communicator);
  u_fg_ghosted = system_solution_block.block(1);
  LA::MPI::Vector lam_ghosted(owned_mult, relevant_mult, mpi_communicator);
  lam_ghosted = system_solution_block.block(2);

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_bg);
    data_out.add_data_vector(u_bg_ghosted, "u");
    Vector<float> subdomain(tria_bg->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_bg->locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(*mapping_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-bg", cycle, mpi_communicator,
                                        2, 0);
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_fg);
    data_out.add_data_vector(u_fg_ghosted, "u2");
    data_out.build_patches(*mapping_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-imm", cycle, mpi_communicator,
                                        2, 0);
  }
  {
    // The multiplier lives on `dof_handler_mult`, which may have a different
    // FE (e.g. FE_DGQ(0)) than `dof_handler_fg`; output it separately.
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_mult);
    data_out.add_data_vector(lam_ghosted, "lambda");
    data_out.build_patches(*mapping_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-lam", cycle, mpi_communicator,
                                        2, 0);
  }
}

template <int dim> void EllipticInterfaceDLMParallel<dim>::run() {
  for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
       ++cycle) {
    pcout << "==============================================================="
          << "\nRefinement cycle: " << cycle << std::endl;
    // For hex meshes the background and immersed triangulations are
    // parallel::distributed::Triangulation, which supports `refine_global`
    // in place: we build the coarse mesh + initial refinement at cycle 0,
    // then refine_global(1) on subsequent cycles, without clearing.
    // For simplex meshes we still rebuild from scratch (the
    // fullydistributed triangulation does not support refine_global).
    generate_grids(cycle);
    system_setup();
    solve();
    if (parameters.do_convergence_test) {
      compute_errors();
      compute_augmentation_consistency();
    }
    if (this_mpi_process == 0) {
      pcout << "\nRefinement study summary:\n";
      convergence_table.write_text(
          std::cout, TableHandler::TextOutputFormat::org_mode_table);
      std::cout << std::flush;
    }
    output_results(cycle);
  }

  if (this_mpi_process == 0) {
    pcout << "\n=== Final convergence table ===" << std::endl;
    convergence_table.write_text(
        std::cout, TableHandler::TextOutputFormat::org_mode_table);
  }
}

int main(int argc, char *argv[]) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, numbers::invalid_unsigned_int);

    const std::string parameter_file =
        (argc > 1) ? argv[1] : "parameters_elliptic_interface_op.prm";

    // Detect the dimension from the parameter file by scanning for the
    // "Elliptic Interface Problem<dim>" subsection header. Supported values
    // are dim == 2 and dim == 3.
    auto scan_prm_for_dim = [](const std::string &fname) -> int {
      std::ifstream in(fname);
      AssertThrow(in, ExcMessage("Could not open parameter file: " + fname));
      std::string content((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
      auto strip = [](std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(),
                               [](unsigned char c) { return std::isspace(c); }),
                s.end());
        return s;
      };
      const std::string c = strip(content);
      const bool has_2 =
          c.find("EllipticInterfaceProblem<2>") != std::string::npos;
      const bool has_3 =
          c.find("EllipticInterfaceProblem<3>") != std::string::npos;
      AssertThrow(has_2 ^ has_3,
                  ExcMessage("The parameter file must contain exactly one of "
                             "'subsection Elliptic Interface Problem<2>' or "
                             "'subsection Elliptic Interface Problem<3>'."));
      return has_2 ? 2 : 3;
    };

    auto run = [&](auto dim_const) {
      constexpr int dim = decltype(dim_const)::value;
      ProblemParameters<dim> parameters;
      ParameterAcceptor::initialize(parameter_file, "used_parameters_op.prm");
      deallog.depth_console(
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? 10 : 0);
      EllipticInterfaceDLMParallel<dim> solver(parameters);
      solver.run();
    };

    const int dim = scan_prm_for_dim(parameter_file);
    if (dim == 2)
      run(std::integral_constant<int, 2>{});
    else
      run(std::integral_constant<int, 3>{});
  } catch (std::exception &exc) {
    std::cerr << "\n---------- Exception ----------\n"
              << exc.what() << "\nAborting!\n"
              << "-------------------------------" << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "\nUnknown exception. Aborting!" << std::endl;
    return 1;
  }
  return 0;
}