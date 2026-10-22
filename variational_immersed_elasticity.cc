// Parallel MPI version of the elasticity fictitious-domain interface problem
// with augmented-Lagrangian (AL) preconditioner. Operator-form AL only: the
// augmented (1,1)-block is built at the variational level by adding
// contributions through particles. Both the ideal and the modified variants
// of the AL preconditioner are supported.
//
// This is an extension of elliptic_interface_op.cc to linear elasticity. The
// (vector-valued) bilinear form on a sub-domain is
//
//   a(u,v) = \int lambda * div(u) * div(v) + 2 * mu * eps(u) : eps(v) dx,
//
// and Lame parameters are given separately for the background and immersed
// domains, following the serial reference elliptic_interface_elasticity.cc.
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
#include <deal.II/base/symmetric_tensor.h>
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
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/numerics/rtree.h>

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

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/utilities.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

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

// The augmented 3x3 system in operator form, with off-diagonal blocks
// assembled at the variational level (mirroring elliptic_interface_op.cc).
// With a non-trivial FE triple the multiplier space may differ from the
// immersed primal: in that case
//   * `M`                  is the rectangular cross-mass (n_imm x n_mult),
//   * `Ct`                 is the coupling (n_bg x n_mult),
//   * `coupling_matrix_01` is the variational off-diagonal (n_bg x n_imm),
//   * `coupling_matrix_10` is its transpose (n_imm x n_bg),
//   * `W` (used by InvWOp) is the multiplier-multiplier mass (n_mult x
//      n_mult).
// For Q1-Q1-Q1 (multiplier coincides with immersed primal) all of these
// collapse to the obvious square / equal matrices.
//
//   (0,0) = A_bg                              (AL term already added in
//                                              place by particle quadrature
//                                              for Q1-Q1-Q1, or by the
//                                              direct K-driven outer
//                                              product for Q1-Bubble-Q0)
//   (0,1) = -gamma1 * coupling_matrix_01     (u|_Omega_2, v_2)
//   (0,2) =  Ct                              (u|_Omega_2, mu)
//   (1,0) = -gamma2 * coupling_matrix_10     transpose of (0,1)
//   (1,1) =  A_fg + gamma2 * M_imm           (already in A_fg_plus_scaled_M)
//   (1,2) = -M                               (u_2, mu)
//   (2,0) =  Ct^T
//   (2,1) = -M^T                             (lambda, v_2)
//   (2,2) =  0
//
// IMPORTANT (Q0 multiplier triples): to keep the AL augmentation exactly
// consistent, i.e. equal to
//   gamma_1 * Ct W^{-1} (Ct^T u_1 - M^T u_2)   in row 0 and
//   gamma_2 * M  W^{-1} (M^T u_2 - Ct^T u_1)   in row 1,
// which vanishes identically on the discrete constraint manifold (row 2).
// The matrices named above CONTAIN the projected quantities in that case:
//   * coupling_matrix_01 =  Ct W^{-1} M^T      (n_bg x n_imm)
//   * coupling_matrix_10 =  M  W^{-1} Ct^T     (n_imm x n_bg)
//   * A_fg_plus_scaled_M =  A_fg + gamma2 * M W^{-1} M^T
// Because W is diagonal for a Q0 multiplier, all of these are assembled
// cellwise as outer products of single-cell integral vectors (no global
// triple product is ever formed)

template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp>
struct System3x3 {
  const SparseMatrixType &A_bg;
  const SparseMatrixType &A_fg_plus_scaled_M; // A_fg + gamma2 * M_imm
  const SparseMatrixType &M;                  // cross-mass n_imm x n_mult
  const SparseMatrixType &Ct;                 // n_bg x n_mult
  const SparseMatrixType &coupling_matrix_01; // n_bg x n_imm
  const SparseMatrixType &coupling_matrix_10; // n_imm x n_bg
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg, owned_mult;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm);
    VectorType tm(owned_mult, comm);
    VectorType tb(owned_bg, comm);

    // Row 0: A_bg * x0  - gamma1 * C_01 * x1  + Ct * x2
    A_bg.vmult(dst.block(0), src.block(0));
    coupling_matrix_01.vmult(tb, src.block(1));
    dst.block(0).add(-gamma1, tb);
    Ct.vmult(tb, src.block(2));
    dst.block(0) += tb;

    // Row 1: -gamma2 * C_10 * x0  +  A_fg_plus_scaled_M * x1  -  M * x2
    coupling_matrix_10.vmult(ti, src.block(0));
    ti *= -gamma2;
    dst.block(1) = ti;
    A_fg_plus_scaled_M.vmult_add(dst.block(1), src.block(1));
    M.vmult(ti, src.block(2)); // (n_imm x n_mult) * mult -> imm
    dst.block(1) -= ti;

    // Row 2: Ct^T * x0  -  M^T * x1   (result lives in mult space)
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
  const SparseMatrixType &coupling_matrix_01; // n_bg x n_imm
  const SparseMatrixType &coupling_matrix_10; // n_imm x n_bg
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm);
    VectorType tb(owned_bg, comm);

    A_bg.vmult(dst.block(0), src.block(0));
    coupling_matrix_01.vmult(tb, src.block(1));
    dst.block(0).add(-gamma1, tb);

    coupling_matrix_10.vmult(ti, src.block(0));
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

template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp, typename A11InvOp,
          typename A22InvOp>
struct ModifiedALPreconditioner {
  const SparseMatrixType &M;  // n_imm x n_mult cross-mass
  const SparseMatrixType &Ct; // n_bg x n_mult
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
    M.vmult(rhs1, lam); // (n_imm x n_mult) * mult -> imm
    rhs1 += src.block(1);
    dst.block(1) = 0.;
    dst.block(1) = A22_inv * rhs1;

    VectorType rhs0(owned_bg, comm);
    VectorType tm(owned_mult, comm), tm2(owned_mult, comm);
    VectorType tb(owned_bg, comm);
    M.Tvmult(tm, dst.block(1)); // M^T * u2 -> mult
    invW.vmult(tm2, tm);
    Ct.vmult(tb, tm2);
    rhs0 = src.block(0);
    rhs0.add(gamma, tb);
    Ct.vmult(tb, lam);
    rhs0 -= tb;
    dst.block(0) = 0.;
    dst.block(0) = A11_inv * rhs0;

    dst.block(2) = lam;
  }
};

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
      M.vmult(ti, lam); // mult -> imm
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
  std::string name_of_immersed_grid = "hyper_rectangle";
  std::string arguments_for_immersed_grid = "-0.4,-0.2: 0.4,0.2: false";

  // External (on-disk) meshes. When non-empty, the corresponding triangulation
  // is read from this file (using GridIn::read_msh) instead of being built
  // through GridGenerator. The immersed grid can additionally be rescaled by
  // `immersed_scale_factor` via GridTools::scale before being refined.
  std::string name_external_background_grid = "";
  std::string name_external_immersed_grid = "";
  double background_scale_factor = 1.0;
  double immersed_scale_factor = 1.0;

  // Cell type used independently for each bulk mesh: "hex" or "simplex".
  // Setting them independently allows e.g. a hex background with a simplex
  // immersed mesh (typical when the immersed body comes from gmsh as a .msh
  // tetrahedral file).
  std::string mesh_type_background = "hex";
  std::string mesh_type_immersed = "hex";

  unsigned int n_refinement_cycles = 5;

  // Number of local refinement passes applied to the (hex) background
  // mesh on every cell whose bounding box intersects the surface of the
  // immersed body. Applied once, as a post-processing step after the initial
  // grid generation at cycle 0. A value of 0 disables the feature.
  unsigned int interface_refinement_levels = 0;

  // Lame parameters per sub-domain.
  mutable double lambda_background = 2.;
  mutable double lambda_immersed = 20.;
  mutable double mu_background = 1.;
  mutable double mu_immersed = 10.;

  std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

  // Compression-style Dirichlet BC. If `compression_delta != 0`, the
  // displacement is constrained to (0, ..., 0, -compression_delta) on the
  // boundary with id `top_compression_boundary_id`. The remaining un-listed
  // boundary ids (i.e. not in `dirichlet_ids` and not the compression top)
  // are left natural, which corresponds to sigma * n = 0.
  int top_compression_boundary_id = -1;
  double compression_delta = 0.0;

  unsigned int background_space_finite_element_degree = 1;
  unsigned int immersed_space_finite_element_degree = 1;
  unsigned int coupling_quadrature_order = 3;
  unsigned int verbosity_level = 4;

  // Finite-element triple selecting (bg, immersed primal, multiplier):
  //   * "Q1-Q1-Q1"      (default): bg, immersed primal and multiplier are
  //                     all FESystem(FE_Q(deg)) on hex, or
  //                     FESystem(FE_SimplexP(deg)) on simplex. The
  //                     multiplier basis coincides with the immersed
  //                     primal basis; the augmentation in A_bg is the
  //                     particle-quadrature
  //                     (phi_i, phi_j)_{Omega_2} (component-wise).
  //   * "Q1-Bubble-Q0": (hex immersed only) bg = FESystem(FE_Q(1)),
  //                     immersed primal = FESystem(FE_Q_Bubbles(1)),
  //                     multiplier = FESystem(FE_DGQ(0)). The augmentation
  //                     in A_bg is assembled directly as the bilinear
  //                     form (P_h phi_i, phi_j)_{Omega_2} where P_h is
  //                     the component-wise L^2 orthogonal projection on
  //                     the cellwise-constant Q0 multiplier space.
  //   * "Q2-Q2-Q0":    (hex only) bg = FESystem(FE_Q(2)),
  //                     immersed primal = FESystem(FE_Q(2)),
  //                     multiplier = FESystem(FE_DGQ(0)). Requires both
  //                     "FE degree background" and "FE degree immersed"
  //                     to be set to 2. Stable Q2-Q2-Q0 triple; the
  //                     augmentation in A_bg is assembled directly by
  //                     the Q0 outer-product path, same as for the
  //                     bubble-enriched triple.
  std::string fe_triple = "Q1-Q1-Q1";

  bool use_modified_AL_preconditioner = true;
  bool use_fixed_iterations = false;

  mutable double gamma_AL_background = 10.;
  mutable double gamma_AL_immersed = 1e-2;

  mutable ParameterAcceptorProxy<ReductionControl> outer_solver_control;
  mutable ParameterAcceptorProxy<ReductionControl> inner_solver_control;
  mutable ParameterAcceptorProxy<IterationNumberControl>
      iteration_number_control;

  // Vector-valued right-hand sides (dim components each).
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> f_1;
  ParameterAcceptorProxy<Functions::ParsedFunction<dim>> f_2_minus_f;
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
    : ParameterAcceptor("Elasticity Interface Problem/"),
      outer_solver_control("Outer solver control"),
      inner_solver_control("Inner solver control"),
      iteration_number_control("Iteration number control"),
      f_1("Right hand side f_1", dim),
      f_2_minus_f("Right hand side f_2 - f", dim) {
  add_parameter("FE degree background", background_space_finite_element_degree,
                "", this->prm, Patterns::Integer(1));
  add_parameter("FE degree immersed", immersed_space_finite_element_degree, "",
                this->prm, Patterns::Integer(1));
  add_parameter("Coupling quadrature order", coupling_quadrature_order);
  add_parameter("Output directory", output_directory);
  add_parameter("lambda background", lambda_background);
  add_parameter("lambda immersed", lambda_immersed);
  add_parameter("mu background", mu_background);
  add_parameter("mu immersed", mu_immersed);
  add_parameter("Homogeneous Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Top compression boundary id", top_compression_boundary_id,
                "Boundary id where a nonzero Dirichlet condition u = "
                "(0,...,0,-delta) is enforced (compression). Use a negative "
                "value to disable.");
  add_parameter("Compression delta", compression_delta,
                "Magnitude of the imposed compression displacement on the "
                "top boundary. Applied only to the last (z) component.");
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
    add_parameter("Mesh type background", mesh_type_background,
                  "Cell type used for the background bulk mesh. Either "
                  "\"hex\" (tensor-product quads/hexes) or \"simplex\" "
                  "(tris/tets).",
                  this->prm, Patterns::Selection("hex|simplex"));
    add_parameter("Mesh type immersed", mesh_type_immersed,
                  "Cell type used for the immersed bulk mesh. Either "
                  "\"hex\" (tensor-product quads/hexes) or \"simplex\" "
                  "(tris/tets). Can differ from \"Mesh type background\".",
                  this->prm, Patterns::Selection("hex|simplex"));
    add_parameter("Name of the external background grid file",
                  name_external_background_grid,
                  "If non-empty, the background mesh is read from this file "
                  "(GridIn::read_msh) instead of being built by "
                  "GridGenerator.");
    add_parameter("Name of the external immersed grid file",
                  name_external_immersed_grid,
                  "If non-empty, the immersed mesh is read from this file "
                  "(GridIn::read_msh) instead of being built by "
                  "GridGenerator.");
    add_parameter("Background grid scale factor", background_scale_factor,
                  "Uniform scaling factor applied to the background mesh "
                  "(both generated and external) via GridTools::scale.");
    add_parameter("Immersed grid scale factor", immersed_scale_factor,
                  "Uniform scaling factor applied to the immersed mesh "
                  "(both generated and external) via GridTools::scale.");
    add_parameter("FE triple", fe_triple,
                  "Selects the (background, immersed primal, multiplier) "
                  "finite-element triple. \"Q1-Q1-Q1\" (default) uses "
                  "FESystem(FE_Q(deg)) on hex or FESystem(FE_SimplexP(deg)) "
                  "on simplex for all three spaces. \"Q1-Bubble-Q0\" "
                  "(hex immersed only) pins immersed = "
                  "FESystem(FE_Q_Bubbles(1)) and multiplier = "
                  "FESystem(FE_DGQ(0)). \"Q2-Q2-Q0\" (hex only) uses "
                  "FESystem(FE_Q(2)) for both bg and immersed primal and "
                  "FESystem(FE_DGQ(0)) for the multiplier; both FE "
                  "degrees must be set to 2.",
                  this->prm,
                  Patterns::Selection("Q1-Q1-Q1|Q1-Bubble-Q0|Q2-Q2-Q0"));
  }
  leave_subsection();

  enter_subsection("Refinement and remeshing");
  {
    add_parameter("Initial background refinement",
                  initial_background_refinement);
    add_parameter("Initial immersed refinement", initial_immersed_refinement);
    add_parameter("Refinemented cycles", n_refinement_cycles);
    add_parameter("Interface refinement levels", interface_refinement_levels,
                  "Number of local refinement passes applied (as a "
                  "post-processing step at cycle 0) to background cells "
                  "whose bounding box intersects the surface of the "
                  "immersed body. Only effective for hex "
                  "(parallel::distributed) backgrounds. 0 disables it.");
  }
  leave_subsection();

  enter_subsection("AL preconditioner");
  {
    add_parameter("Use modified AL preconditioner",
                  use_modified_AL_preconditioner);
    add_parameter("gamma fluid", gamma_AL_background);
    add_parameter("gamma solid", gamma_AL_immersed);
    add_parameter("Verbosity level", verbosity_level);
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

  // Default constant body forces (per component, separated by ";").
  f_1.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression",
                               dim == 2 ? "1.; 1." : "1.; 1.; 1.");
  });
  f_2_minus_f.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression",
                               dim == 2 ? "1.; 0." : "1.; 0.; 0.");
  });
}

// Parallel elasticity interface solver
template <int dim> class ElasticityInterfaceDLMParallel {
public:
  ElasticityInterfaceDLMParallel(const ProblemParameters<dim> &prm);

  void run();

private:
  void generate_grids(unsigned int extra_refinement = 0);
  void refine_background_at_interface(unsigned int n_levels);
  void system_setup();
  void assemble_bg();
  void assemble_fg(double gamma_2_h);
  void assemble_mass_fg();
  void assemble_mass_mult();
  void add_projected_mass_fg(double gamma_2_h);
  void assemble_coupling_and_augment_bg(double gamma_1_h);
  unsigned int solve();
  void output_results(unsigned int cycle) const;

  const ProblemParameters<dim> &parameters;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;
  mutable TimerOutput computing_timer;

  // Background triangulation is polymorphic so that hex meshes can use
  // `parallel::distributed::Triangulation` (p4est), while simplex backgrounds
  // keep using `parallel::fullydistributed::Triangulation`
  std::unique_ptr<parallel::DistributedTriangulationBase<dim>> tria_bg;

  // The immersed triangulation is polymorphic for the same reason as the
  // background one
  std::unique_ptr<parallel::DistributedTriangulationBase<dim>> tria_fg;

  // Persistent serial triangulations used to build the fullydistributed
  // simplex meshes incrementally across refinement cycles. Building a
  // (simplex) `parallel::fullydistributed::Triangulation` requires a serial
  // mesh as input
  std::unique_ptr<Triangulation<dim>> serial_tria_bg_cache;
  std::unique_ptr<Triangulation<dim>> serial_tria_fg_cache;

  // Vector-valued FEs
  std::unique_ptr<FiniteElement<dim>> fe_bg_ptr;
  std::unique_ptr<FiniteElement<dim>> fe_fg_ptr;
  // Vector-valued multiplier FE. For "Q1-Q1-Q1" it coincides with
  // `fe_fg_ptr` (a copy is done so the two DoFHandlers stay independent).
  // For "Q1-Bubble-Q0" / "Q2-Q2-Q0" this is FESystem(FE_DGQ(0)).
  std::unique_ptr<FiniteElement<dim>> fe_mult_ptr;
  std::unique_ptr<Mapping<dim>> mapping_bg_ptr;
  std::unique_ptr<Mapping<dim>> mapping_fg_ptr;
  std::function<Quadrature<dim>(unsigned int)> make_quadrature_bg;
  std::function<Quadrature<dim>(unsigned int)> make_quadrature_fg;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;
  // Multiplier DoFHandler. Lives on the same immersed triangulation as
  // `dof_handler_fg` but uses `*fe_mult_ptr`.
  DoFHandler<dim> dof_handler_mult;

  IndexSet owned_bg, relevant_bg;
  IndexSet owned_fg, relevant_fg;
  IndexSet owned_mult, relevant_mult;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;
  AffineConstraints<double> constraints_mult;

  LA::MPI::SparseMatrix A_bg;
  // NOTE: the bare immersed stiffness A_fg is deliberately NOT stored. Only
  // the augmented block A_fg_plus_scaled_M = A_fg + gamma_2 * M W^{-1} M^T
  // is ever read (it is both the (1,1) block and the matrix the inner CG
  // inverts), so keeping A_fg around cost a full immersed matrix for
  // nothing.
  LA::MPI::SparseMatrix A_fg_plus_scaled_M;
  // Preconditioner twin of `A_fg_plus_scaled_M` used only to feed AMG for
  // the (2,2) block when the immersed element carries cell-interior
  // ("bubble") dofs. It holds the same matrix with the vertex<->bubble
  // couplings dropped, i.e. the 2x2 block-Jacobi splitting
  //     diag( A_vv , A_bb )
  // of A22 with respect to (vertex dofs, cell-interior dofs). A_bb is
  // block-diagonal (one small dim x dim block per cell, since bubbles have
  // no support on the cell boundary), so a point smoother resolves it
  // exactly enough while AMG now coarsens a clean, Q1-like A_vv whose
  // near-null space is exactly the rigid-body modes it is handed.
  // Empty (and unused) when the immersed element has no interior dofs.
  LA::MPI::SparseMatrix A22_prec;
  // Cross-mass (n_imm x n_mult): (phi_imm_i, phi_mult_j)_{Omega_2}.
  // For Q1-Q1-Q1 it equals the immersed primal mass matrix.
  LA::MPI::SparseMatrix mass_matrix_fg;
  // Multiplier mass W (n_mult x n_mult): (phi_mult_i, phi_mult_j)_{Omega_2}.
  // For Q1-Bubble-Q0 / Q2-Q2-Q0 it is diagonal with entries |K| per
  // component.
  LA::MPI::SparseMatrix mass_matrix_mult;
  // (n_bg x n_mult): Ct used in (0,2) / (2,0).
  LA::MPI::SparseMatrix coupling_matrix;
  // (n_bg x n_imm): variational off-diagonal used in (0,1).
  LA::MPI::SparseMatrix coupling_matrix_01;
  // (n_imm x n_bg): transpose of `coupling_matrix_01`, used in (1,0).
  LA::MPI::SparseMatrix coupling_matrix_10;
  // (n_mult x n_bg): per-(K,c) integrals of bg basis functions over each
  // immersed cell, i.e. Ct^T. Row (K,c) holds c_{K,c}[i] = int_K
  // phi_i^{bg,(c)}. Used only by the Q0 augmentation path to build the
  // outer-product form of the projected augmentation blocks.
  LA::MPI::SparseMatrix bg_mult_integral_T;
  // (n_mult x n_imm): the same thing on the immersed primal side, i.e.
  // M^T. Row (K,c) holds m_{K,c}[j] = int_K phi_j^{imm,(c)}. Q0 only.
  LA::MPI::SparseMatrix imm_mult_integral_T;
  // Snapshot of A_bg taken BEFORE the AL augmentation. Currently kept for
  // symmetry with `elliptic_interface_op.cc`, where it is used to recover
  // the AL term as (A_bg - K_bg_pre_aug) / gamma_1.
  LA::MPI::SparseMatrix K_bg_pre_aug;

  // True iff `fe_triple` selects a Q0 multiplier ("Q1-Bubble-Q0" or
  // "Q2-Q2-Q0"), in which case the whole AL augmentation (the (0,0),
  // (0,1), (1,0) and (1,1) blocks) is assembled as cellwise outer products
  // driven by the immersed cells instead of by the particle quadrature.
  bool use_q0_multiplier = false;

  // Local indices of the cell-interior ("bubble") dofs of the immersed
  // element, and the flag enabling the vertex/bubble block-Jacobi
  // preconditioner for the (2,2) block. Both are empty/false unless the
  // immersed element actually has interior dofs.
  std::vector<bool> imm_interior_dof;
  bool use_bubble_block_prec = false;

  LA::MPI::BlockVector system_rhs_block;
  LA::MPI::BlockVector system_solution_block;

  // Cumulative number of inner (P)CG iterations spent inverting the (1,1)
  // block A_bg of the preconditioner during the last outer FGMRES solve,
  // together with the number of times that block was inverted. Used to
  // report the average inner PCG iteration count per (1,1)-block solve.
  unsigned int last_inner_iters_A11 = 0;
  unsigned int last_n_A11_applications = 0;

  ConvergenceTable convergence_table;
};

template <int dim>
ElasticityInterfaceDLMParallel<dim>::ElasticityInterfaceDLMParallel(
    const ProblemParameters<dim> &prm)
    : parameters(prm), mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, this_mpi_process == 0),
      computing_timer(mpi_communicator, pcout,
                      TimerOutput::every_call_and_summary,
                      TimerOutput::wall_times),
      dof_handler_fg(), dof_handler_mult() {
  if (parameters.mesh_type_background == "hex") {
    tria_bg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
  } else {
    tria_bg = std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
  }
  if (parameters.mesh_type_immersed == "hex") {
    tria_fg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
  } else {
    tria_fg = std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
  }
  dof_handler_bg.reinit(*tria_bg);
  dof_handler_fg.reinit(*tria_fg);
  dof_handler_mult.reinit(*tria_fg);
  AssertThrow(
      parameters.mesh_type_background == "hex" ||
          parameters.mesh_type_background == "simplex",
      ExcMessage("Mesh type background must be \"hex\" or \"simplex\"."));
  AssertThrow(parameters.mesh_type_immersed == "hex" ||
                  parameters.mesh_type_immersed == "simplex",
              ExcMessage("Mesh type immersed must be \"hex\" or \"simplex\"."));

  auto build_fe_mapping_quad =
      [](const std::string &mesh_type, unsigned int fe_degree,
         std::unique_ptr<FiniteElement<dim>> &fe_out,
         std::unique_ptr<Mapping<dim>> &mapping_out,
         std::function<Quadrature<dim>(unsigned int)> &make_quad_out) {
        if (mesh_type == "hex") {
          fe_out = std::make_unique<FESystem<dim>>(FE_Q<dim>(fe_degree), dim);
          mapping_out = std::make_unique<MappingQ1<dim>>();
          make_quad_out = [](unsigned int n) -> Quadrature<dim> {
            return QGauss<dim>(n);
          };
        } else {
          fe_out =
              std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(fe_degree), dim);
          mapping_out = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
          make_quad_out = [](unsigned int n) -> Quadrature<dim> {
            return QGaussSimplex<dim>(std::min(4u, std::max(1u, n)));
          };
        }
      };

  build_fe_mapping_quad(parameters.mesh_type_background,
                        parameters.background_space_finite_element_degree,
                        fe_bg_ptr, mapping_bg_ptr, make_quadrature_bg);
  build_fe_mapping_quad(parameters.mesh_type_immersed,
                        parameters.immersed_space_finite_element_degree,
                        fe_fg_ptr, mapping_fg_ptr, make_quadrature_fg);

  // FE triple selection. The default ("Q1-Q1-Q1") leaves `fe_fg_ptr`
  // untouched and uses a twin of it as the multiplier basis.
  // "Q1-Bubble-Q0" overrides `fe_fg_ptr` with a bubble-enriched basis.
  // Both "Q1-Bubble-Q0" and "Q2-Q2-Q0" use a cellwise-constant DG multiplier.
  AssertThrow(parameters.fe_triple == "Q1-Q1-Q1" ||
                  parameters.fe_triple == "Q1-Bubble-Q0" ||
                  parameters.fe_triple == "Q2-Q2-Q0",
              ExcMessage("FE triple must be \"Q1-Q1-Q1\", "
                         "\"Q1-Bubble-Q0\" or \"Q2-Q2-Q0\"."));
  if (parameters.fe_triple == "Q1-Bubble-Q0") {
    AssertThrow(parameters.mesh_type_immersed == "hex",
                ExcMessage("FE triple \"Q1-Bubble-Q0\" requires "
                           "\"Mesh type immersed = hex\"."));

    AssertThrow(parameters.immersed_space_finite_element_degree == 1,
                ExcMessage("FE triple \"Q1-Bubble-Q0\" requires \"FE degree "
                           "immersed\" to be 1."));

    fe_fg_ptr = std::make_unique<FESystem<dim>>(FE_Q_Bubbles<dim>(1), dim);
    fe_mult_ptr = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), dim);
    use_q0_multiplier = true;
  } else if (parameters.fe_triple == "Q2-Q2-Q0") {
    AssertThrow(parameters.mesh_type_background == "hex" &&
                    parameters.mesh_type_immersed == "hex",
                ExcMessage("FE triple \"Q2-Q2-Q0\" requires both "
                           "\"Mesh type background\" and \"Mesh type "
                           "immersed\" to be \"hex\"."));
    AssertThrow(parameters.background_space_finite_element_degree == 2 &&
                    parameters.immersed_space_finite_element_degree == 2,
                ExcMessage("FE triple \"Q2-Q2-Q0\" requires both \"FE degree "
                           "background\" and \"FE degree immersed\" to be 2."));
    // fe_bg_ptr and fe_fg_ptr were already built as FESystem(FE_Q(2))
    // by build_fe_mapping_quad above. Only override the multiplier.
    fe_mult_ptr = std::make_unique<FESystem<dim>>(FE_DGQ<dim>(0), dim);
    use_q0_multiplier = true;
  } else {
    // Twin of the (possibly user-overridden) immersed primal basis.
    if (parameters.mesh_type_immersed == "hex")
      fe_mult_ptr = std::make_unique<FESystem<dim>>(
          FE_Q<dim>(parameters.immersed_space_finite_element_degree), dim);
    else
      fe_mult_ptr = std::make_unique<FESystem<dim>>(
          FE_SimplexP<dim>(parameters.immersed_space_finite_element_degree),
          dim);
    use_q0_multiplier = false;
  }

  // Mark the cell-interior dofs of the immersed element. On both hex and
  // simplex reference cells deal.II numbers dofs by increasing object
  // dimension (vertices, lines, quads, hexes), so the interior dofs are
  // exactly the trailing ones. For FESystem(FE_Q_Bubbles(1), dim) in 3d
  // these are the `dim` bubble dofs (local indices 24..26 out of 27);
  // for a plain FE_Q/FE_SimplexP of degree 1 there are none, in which case
  // the vertex/bubble splitting below degenerates to the identity and the
  // preconditioner is left exactly as it was.
  {
    const unsigned int n_imm_dofs_per_cell = fe_fg_ptr->n_dofs_per_cell();
    const unsigned int first_interior = (dim == 3)
                                            ? fe_fg_ptr->get_first_hex_index()
                                            : fe_fg_ptr->get_first_quad_index();
    imm_interior_dof.assign(n_imm_dofs_per_cell, false);
    for (unsigned int i = first_interior; i < n_imm_dofs_per_cell; ++i)
      imm_interior_dof[i] = true;
    // Restrict the splitting to genuinely bubble-enriched elements. What
    // makes those pathological for AMG is that FE_Q_Bubbles interpolation
    // sets the bubble coefficient to zero, so the rigid-body near-null
    // space handed to ML vanishes on the bubble dofs and the tentative
    // prolongator has zero rows there. A plain FE_Q(2) interior dof sits
    // at the cell centroid and DOES carry a non-zero rigid-body value, so
    // AMG copes with it unaided and Q2-Q2-Q0 is left untouched.
    const bool bubble_enriched_element = parameters.fe_triple == "Q1-Bubble-Q0";
    use_bubble_block_prec =
        bubble_enriched_element && (first_interior < n_imm_dofs_per_cell);
    if (use_bubble_block_prec)
      pcout << "Immersed element " << fe_fg_ptr->get_name() << " has "
            << (n_imm_dofs_per_cell - first_interior)
            << " cell-interior dofs per cell: using the vertex/bubble "
               "block-Jacobi preconditioner for the (2,2) block."
            << std::endl;
  }

  AssertThrow(parameters.mu_background > 0.,
              ExcMessage("mu background must be positive."));
  AssertThrow(parameters.mu_immersed > parameters.mu_background,
              ExcMessage("mu immersed must be greater than mu background."));
  AssertThrow(parameters.lambda_immersed >= parameters.lambda_background,
              ExcMessage("lambda immersed must be >= lambda background."));
  AssertThrow(parameters.gamma_AL_background > 0.,
              ExcMessage("gamma fluid must be positive."));
  AssertThrow(parameters.gamma_AL_immersed > 0.,
              ExcMessage("gamma solid must be positive."));

  if (this_mpi_process == 0 &&
      !std::filesystem::exists(parameters.output_directory)) {
    pcout << "Output directory does not exist, creating: "
          << parameters.output_directory << std::endl;
    std::filesystem::create_directories(parameters.output_directory);
  }
}

template <int dim>
void ElasticityInterfaceDLMParallel<dim>::generate_grids(
    unsigned int extra_refinement) {
  TimerOutput::Scope t(computing_timer, "Grid generation");

  // Incremental builder for fullydistributed triangulations: keeps a serial
  // `Triangulation` cached between cycles so that we only ever apply ONE extra
  // global refinement per cycle instead of redoing all previous refinements
  // from scratch.
  auto build_one_incremental =
      [this](parallel::fullydistributed::Triangulation<dim> &out,
             std::unique_ptr<Triangulation<dim>> &serial_cache,
             const std::string &name, const std::string &args,
             unsigned int base_refinements, const std::string &external_file,
             double scale_factor, const std::string &mesh_type) {
        if (!serial_cache) {
          // First call: build the base serial mesh and apply the initial
          // refinement. The base mesh is then kept alive for the lifetime of
          // the solver and reused (and further refined) at every cycle.
          serial_cache = std::make_unique<Triangulation<dim>>();
          auto &serial = *serial_cache;
          if (!external_file.empty()) {
            GridIn<dim> grid_in;
            grid_in.attach_triangulation(serial);
            std::ifstream input_file(external_file);
            AssertThrow(input_file,
                        ExcMessage("Could not open external grid file: " +
                                   external_file));
            grid_in.read_msh(input_file);
            if (scale_factor != 1.0)
              GridTools::scale(scale_factor, serial);
            serial.refine_global(base_refinements);
          } else if (mesh_type == "hex") {
            GridGenerator::generate_from_name_and_arguments(serial, name, args);
            if (scale_factor != 1.0)
              GridTools::scale(scale_factor, serial);
            serial.refine_global(base_refinements);
          } else {
            // For built-in simplex meshes we refine the *hex* template at
            // the base level and only then convert, so that cycle 0 matches
            // the previous behaviour exactly. Subsequent cycles instead
            // refine the simplex mesh in place (red refinement for tets).
            Triangulation<dim> hex_tmp;
            GridGenerator::generate_from_name_and_arguments(hex_tmp, name,
                                                            args);
            hex_tmp.refine_global(base_refinements);
            GridGenerator::convert_hypercube_to_simplex_mesh(hex_tmp, serial);
            if (scale_factor != 1.0)
              GridTools::scale(scale_factor, serial);
          }
        } else {
          // Subsequent calls: just refine the cached serial mesh by one
          // global level. For simplex meshes deal.II uses red refinement.
          serial_cache->refine_global(1);
        }

        auto &serial = *serial_cache;
        GridTools::partition_triangulation(
            Utilities::MPI::n_mpi_processes(mpi_communicator), serial);

        // The p::f::T must be cleared before `create_triangulation` is called
        // a second time on it.
        out.clear();
        for (const auto id : serial.get_manifold_ids())
          if (id != numbers::flat_manifold_id)
            out.set_manifold(id, serial.get_manifold(id));

        const auto desc = TriangulationDescription::Utilities::
            create_description_from_triangulation(serial, mpi_communicator);
        out.create_triangulation(desc);
      };

  // Hex-only path: build directly into a `parallel::distributed::Triangulation`
  // and use p4est's native `refine_global`. This is far cheaper at high
  // refinement than going through a serial triangulation + fullydistributed
  // description, and supports adaptive refinement out of the box.
  auto build_one_distributed =
      [](parallel::distributed::Triangulation<dim> &out,
         const std::string &name, const std::string &args,
         unsigned int n_refinements, const std::string &external_file,
         double scale_factor) {
        if (!external_file.empty()) {
          GridIn<dim> grid_in;
          grid_in.attach_triangulation(out);
          std::ifstream input_file(external_file);
          AssertThrow(input_file,
                      ExcMessage("Could not open external grid file: " +
                                 external_file));
          grid_in.read_msh(input_file);
        } else {
          GridGenerator::generate_from_name_and_arguments(out, name, args);
        }
        if (scale_factor != 1.0)
          GridTools::scale(scale_factor, out);
        out.refine_global(n_refinements);
      };

  bool bg_was_freshly_built = false;
  if (auto *tria_d = dynamic_cast<parallel::distributed::Triangulation<dim> *>(
          tria_bg.get())) {

    // Background is hex -> use parallel::distributed::Triangulation directly.
    // We do NOT clear and rebuild between cycles: at the first call we build
    // the coarse mesh and apply the initial global refinement; on every
    // subsequent call we just call `refine_global(1)`. This is much cheaper
    // than reconstructing the p4est forest from scratch and is the natural
    // way to drive a hierarchy of refined hex meshes.
    bg_was_freshly_built = (tria_d->n_global_active_cells() == 0);
    if (bg_was_freshly_built)
      build_one_distributed(*tria_d, parameters.name_of_background_grid,
                            parameters.arguments_for_background_grid,
                            parameters.initial_background_refinement,
                            parameters.name_external_background_grid,
                            parameters.background_scale_factor);
    else {
      // Free the large cycle-(N-1) sparse matrices to avoid OOM during the
      // subsequent `refine_global()` call.
      A_bg.clear();
      A_fg_plus_scaled_M.clear();
      A22_prec.clear();
      mass_matrix_fg.clear();
      mass_matrix_mult.clear();
      coupling_matrix.clear();
      coupling_matrix_01.clear();
      coupling_matrix_10.clear();
      bg_mult_integral_T.clear();
      imm_mult_integral_T.clear();
      K_bg_pre_aug.clear();

      tria_d->refine_global(1);
    }
  } else {
    // Background is simplex -> fall back to fullydistributed. The serial
    // triangulation is cached so that on every cycle we only refine ONE
    // extra level rather than rebuilding from scratch.
    // Free the previous-cycle matrices for the same reason as the hex path.
    A_bg.clear();
    A_fg_plus_scaled_M.clear();
    A22_prec.clear();
    mass_matrix_fg.clear();
    mass_matrix_mult.clear();
    coupling_matrix.clear();
    coupling_matrix_01.clear();
    coupling_matrix_10.clear();
    bg_mult_integral_T.clear();
    imm_mult_integral_T.clear();
    K_bg_pre_aug.clear();
    auto &tria_fd =
        dynamic_cast<parallel::fullydistributed::Triangulation<dim> &>(
            *tria_bg);
    build_one_incremental(
        tria_fd, serial_tria_bg_cache, parameters.name_of_background_grid,
        parameters.arguments_for_background_grid,
        parameters.initial_background_refinement,
        parameters.name_external_background_grid,
        parameters.background_scale_factor, parameters.mesh_type_background);
  }
  if (auto *tria_fg_d =
          dynamic_cast<parallel::distributed::Triangulation<dim> *>(
              tria_fg.get())) {
    // Hex immersed mesh -> p4est, refined in place exactly like the
    // background.
    const bool immersed_fresh = (tria_fg_d->n_global_active_cells() == 0);
    if (immersed_fresh)
      build_one_distributed(*tria_fg_d, parameters.name_of_immersed_grid,
                            parameters.arguments_for_immersed_grid,
                            parameters.initial_immersed_refinement,
                            parameters.name_external_immersed_grid,
                            parameters.immersed_scale_factor);
    else
      tria_fg_d->refine_global(1);
  } else {
    auto &tria_fg_fd =
        dynamic_cast<parallel::fullydistributed::Triangulation<dim> &>(
            *tria_fg);
    build_one_incremental(
        tria_fg_fd, serial_tria_fg_cache, parameters.name_of_immersed_grid,
        parameters.arguments_for_immersed_grid,
        parameters.initial_immersed_refinement,
        parameters.name_external_immersed_grid,
        parameters.immersed_scale_factor, parameters.mesh_type_immersed);
  }

  // Post-processing: refine background cells whose bounding box intersects
  // the surface of the immersed body. Performed only once, after the initial
  // background mesh was just built; on subsequent cycles the existing
  // refinement pattern is preserved by "refine_global(1)".
  if (bg_was_freshly_built)
    refine_background_at_interface(parameters.interface_refinement_levels);

  const double h_bg = GridTools::maximal_cell_diameter(*tria_bg);
  const double h_fg = GridTools::maximal_cell_diameter(*tria_fg);
  pcout << "h background = " << h_bg << "\n"
        << "h immersed   = " << h_fg << "\n"
        << "ratio (bg/imm) = " << h_bg / h_fg << std::endl;
}

// Post-processing local refinement of the (hex) background near the surface
// of the immersed body. For each of `n_levels` passes, every locally-owned
// background cell whose bounding box intersects the bounding box of any
// boundary face of the immersed mesh is flagged for refinement, then the
// mesh is refined.
template <int dim>
void ElasticityInterfaceDLMParallel<dim>::refine_background_at_interface(
    unsigned int n_levels) {
  if (n_levels == 0)
    return;

  auto *tria_d =
      dynamic_cast<parallel::distributed::Triangulation<dim> *>(tria_bg.get());
  if (tria_d == nullptr) {
    pcout << "Interface post-refinement of the background requested, but the "
             "background is not a parallel::distributed::Triangulation (hex); "
             "skipping."
          << std::endl;
    return;
  }

  TimerOutput::Scope t(computing_timer, "Interface refinement (bg)");

  // Collect bounding boxes of boundary faces of locally-owned immersed
  // cells, then gather across ranks so every rank has the full surface.
  std::vector<BoundingBox<dim>> local_surface_boxes;
  for (const auto &cell : tria_fg->active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int f : cell->face_indices())
        if (cell->face(f)->at_boundary())
          local_surface_boxes.emplace_back(cell->face(f)->bounding_box());

  const auto gathered =
      Utilities::MPI::all_gather(mpi_communicator, local_surface_boxes);
  std::vector<BoundingBox<dim>> surface_boxes;
  for (const auto &v : gathered)
    surface_boxes.insert(surface_boxes.end(), v.begin(), v.end());

  if (surface_boxes.empty()) {
    pcout << "No immersed surface faces found; skipping interface refinement."
          << std::endl;
    return;
  }

  const auto surface_tree = pack_rtree(surface_boxes);

  for (unsigned int l = 0; l < n_levels; ++l) {
    for (const auto &cell : tria_d->active_cell_iterators())
      if (cell->is_locally_owned()) {
        const auto bb = cell->bounding_box();
        if (surface_tree.qbegin(boost::geometry::index::intersects(bb)) !=
            surface_tree.qend())
          cell->set_refine_flag();
      }
    tria_d->execute_coarsening_and_refinement();
  }

  pcout << "After interface post-refinement: bg has "
        << tria_d->n_global_active_cells() << " active cells." << std::endl;
}

template <int dim> void ElasticityInterfaceDLMParallel<dim>::system_setup() {
  TimerOutput::Scope t(computing_timer, "System setup");
  const auto &fe_bg = *fe_bg_ptr;
  const auto &fe_fg = *fe_fg_ptr;

  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);
  dof_handler_mult.distribute_dofs(*fe_mult_ptr);

  owned_bg = dof_handler_bg.locally_owned_dofs();
  relevant_bg = DoFTools::extract_locally_relevant_dofs(dof_handler_bg);

  owned_fg = dof_handler_fg.locally_owned_dofs();
  relevant_fg = DoFTools::extract_locally_relevant_dofs(dof_handler_fg);

  owned_mult = dof_handler_mult.locally_owned_dofs();
  relevant_mult = DoFTools::extract_locally_relevant_dofs(dof_handler_mult);

  constraints_bg.clear();
  constraints_bg.reinit(owned_bg, relevant_bg);
  DoFTools::make_hanging_node_constraints(dof_handler_bg, constraints_bg);
  for (const auto id : parameters.dirichlet_ids)
    VectorTools::interpolate_boundary_values(
        *mapping_bg_ptr, dof_handler_bg, id, Functions::ZeroFunction<dim>(dim),
        constraints_bg);
  if (parameters.top_compression_boundary_id >= 0) {
    // u = (0, ..., 0, -delta): nonzero only on the last component.
    Vector<double> top_value(dim);
    if (parameters.name_of_immersed_grid.find("cylinder") !=
        std::string::npos) {
      pcout << "Immersed cylinder test case... modifying top boundary "
               "condition to +delta in x-direction"
            << std::endl;
      top_value = 0.;
      top_value[0] = +parameters.compression_delta;
    } else {
      top_value = 0.;
      top_value[dim - 1] = -parameters.compression_delta;
    }
    Functions::ConstantFunction<dim> top_bc(top_value);
    VectorTools::interpolate_boundary_values(
        *mapping_bg_ptr, dof_handler_bg,
        static_cast<types::boundary_id>(parameters.top_compression_boundary_id),
        top_bc, constraints_bg);
  }
  constraints_bg.close();

  constraints_fg.clear();
  constraints_fg.reinit(owned_fg, relevant_fg);
  DoFTools::make_hanging_node_constraints(dof_handler_fg, constraints_fg);
  constraints_fg.close();

  constraints_mult.clear();
  constraints_mult.reinit(owned_mult, relevant_mult);
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
    A_fg_plus_scaled_M.reinit(owned_fg, owned_fg, dsp, mpi_communicator);
    if (use_bubble_block_prec)
      A22_prec.reinit(owned_fg, owned_fg, dsp, mpi_communicator);
  }
  // Multiplier mass matrix W (n_mult x n_mult). For Q0 it is diagonal,
  // but we still use the standard make_sparsity_pattern so the same code
  // path works for the twin Q1-Q1-Q1 case.
  {
    DynamicSparsityPattern dsp(dof_handler_mult.n_dofs(),
                               dof_handler_mult.n_dofs(), relevant_mult);
    DoFTools::make_sparsity_pattern(dof_handler_mult, dsp, constraints_mult,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp, owned_mult,
                                               mpi_communicator, relevant_mult);
    mass_matrix_mult.reinit(owned_mult, owned_mult, dsp, mpi_communicator);
  }
  // Cross-mass M (n_imm x n_mult). For Q1-Q1-Q1 it coincides with the
  // immersed primal mass (square); for Q0 multiplier it is rectangular
  // with at most `fe_mult_ptr->n_dofs_per_cell()` nonzeros per immersed
  // row (one entry per (cell, component)).
  {
    DynamicSparsityPattern dsp(dof_handler_fg.n_dofs(),
                               dof_handler_mult.n_dofs(), relevant_fg);
    const unsigned int dpc_fg = fe_fg_ptr->n_dofs_per_cell();
    const unsigned int dpc_mult = fe_mult_ptr->n_dofs_per_cell();
    std::vector<types::global_dof_index> ldi_fg(dpc_fg);
    std::vector<types::global_dof_index> ldi_mult(dpc_mult);
    typename DoFHandler<dim>::active_cell_iterator
        cell_fg = dof_handler_fg.begin_active(),
        end_fg = dof_handler_fg.end(),
        cell_mult = dof_handler_mult.begin_active();
    for (; cell_fg != end_fg; ++cell_fg, ++cell_mult)
      if (cell_fg->is_locally_owned()) {
        cell_fg->get_dof_indices(ldi_fg);
        cell_mult->get_dof_indices(ldi_mult);
        for (unsigned int i = 0; i < dpc_fg; ++i)
          for (unsigned int j = 0; j < dpc_mult; ++j)
            dsp.add(ldi_fg[i], ldi_mult[j]);
      }
    SparsityTools::distribute_sparsity_pattern(dsp, owned_fg, mpi_communicator,
                                               relevant_fg);
    mass_matrix_fg.reinit(owned_fg, owned_mult, dsp, mpi_communicator);
  }

  std::vector<IndexSet> owned_blocks = {owned_bg, owned_fg, owned_mult};
  std::vector<IndexSet> relevant_blocks = {relevant_bg, relevant_fg,
                                           relevant_mult};
  system_rhs_block.reinit(owned_blocks, mpi_communicator);
  system_solution_block.reinit(owned_blocks, mpi_communicator);

  pcout << "N DoF background: " << dof_handler_bg.n_dofs() << "\n"
        << "N DoF immersed:   " << dof_handler_fg.n_dofs() << "\n"
        << "N DoF multiplier: " << dof_handler_mult.n_dofs() << std::endl;
}

// Linear elasticity stiffness assembly. Coefficients (lambda, mu) are passed
// explicitly so this routine can be reused for the background (full
// lambda_bg, mu_bg) and the immersed jump (lambda_imm - lambda_bg, mu_imm -
// mu_bg). The vector-valued rhs is assembled at the same time.
template <int dim>
static void assemble_elasticity_subsystem(
    const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<double> &constraints, double lambda, double mu,
    const Function<dim> &rhs_function,
    const std::function<Quadrature<dim>(unsigned int)> &make_quadrature,
    LA::MPI::SparseMatrix &A, LA::MPI::Vector *rhs_owned = nullptr) {
  const Quadrature<dim> quad = make_quadrature(fe.degree + 1);
  FEValues<dim> fe_values(mapping, fe, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cm(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);

  const FEValuesExtractors::Vector displacements(0);
  std::vector<Vector<double>> rhs_values(quad.size(), Vector<double>(dim));

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm = 0;
      cell_rhs = 0;
      if (rhs_owned != nullptr)
        rhs_function.vector_value_list(fe_values.get_quadrature_points(),
                                       rhs_values);
      for (unsigned int q = 0; q < quad.size(); ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const SymmetricTensor<2, dim> eps_i =
              fe_values[displacements].symmetric_gradient(i, q);
          const double div_i = fe_values[displacements].divergence(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const SymmetricTensor<2, dim> eps_j =
                fe_values[displacements].symmetric_gradient(j, q);
            const double div_j = fe_values[displacements].divergence(j, q);
            cm(i, j) += (lambda * div_i * div_j + 2. * mu * (eps_i * eps_j)) *
                        fe_values.JxW(q);
          }
          if (rhs_owned != nullptr) {
            const Tensor<1, dim> phi_i = fe_values[displacements].value(i, q);
            double rhs_dot_phi = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              rhs_dot_phi += rhs_values[q][d] * phi_i[d];
            cell_rhs(i) += rhs_dot_phi * fe_values.JxW(q);
          }
        }
      }
      cell->get_dof_indices(ldi);
      if (rhs_owned != nullptr)
        constraints.distribute_local_to_global(cm, cell_rhs, ldi, A,
                                               *rhs_owned);
      else
        constraints.distribute_local_to_global(cm, ldi, A);
    }
  A.compress(VectorOperation::add);
  if (rhs_owned != nullptr)
    rhs_owned->compress(VectorOperation::add);
}

template <int dim> void ElasticityInterfaceDLMParallel<dim>::assemble_bg() {
  TimerOutput::Scope t(computing_timer, "Assemble bg (A_omega1)");
  LA::MPI::Vector rhs_owned(owned_bg, mpi_communicator);
  rhs_owned = 0.;
  assemble_elasticity_subsystem<dim>(
      *mapping_bg_ptr, *fe_bg_ptr, dof_handler_bg, constraints_bg,
      parameters.lambda_background, parameters.mu_background, parameters.f_1,
      make_quadrature_bg, A_bg, &rhs_owned);
  system_rhs_block.block(0) = rhs_owned;
}

// Immersed assembly: A_fg uses the jump (lambda_imm - lambda_bg,
// mu_imm - mu_bg) with the immersed rhs (f_2 - f). At the same time we build
// the augmented (1,1) block A_fg_plus_scaled_M and its preconditioner twin
// A22_prec.
//
// The AL mass term added here is gamma_2 * M only for the triples whose
// multiplier coincides with the immersed primal space (W = M). When the
// multiplier is Q0 the term that actually belongs in the (1,1) block is the
// *projected* mass gamma_2 * M W^{-1} M^T; it is added afterwards, cellwise, by
// `add_projected_mass_fg`, so this routine leaves the mass out entirely.
template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_fg(double gamma_2_scaled) {
  TimerOutput::Scope t(computing_timer, "Assemble fg (A_omega2 and A22-prec)");
  const auto &fe_fg = *fe_fg_ptr;
  const Quadrature<dim> quad = make_quadrature_fg(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_fg_ptr, fe_fg, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  FullMatrix<double> cm_Aug(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cm_prec(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);

  // Q0 multiplier: the mass belongs to `add_projected_mass_fg` instead.
  const double mass_weight = use_q0_multiplier ? 0. : gamma_2_scaled;

  const double lambda_diff =
      parameters.lambda_immersed - parameters.lambda_background;
  const double mu_diff = parameters.mu_immersed - parameters.mu_background;

  const FEValuesExtractors::Vector displacements(0);
  std::vector<Vector<double>> rhs_values(quad.size(), Vector<double>(dim));

  LA::MPI::Vector rhs_owned(owned_fg, mpi_communicator);
  rhs_owned = 0.;

  for (const auto &cell : dof_handler_fg.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm_Aug = 0;
      cm_prec = 0;
      cell_rhs = 0;
      parameters.f_2_minus_f.vector_value_list(
          fe_values.get_quadrature_points(), rhs_values);
      for (unsigned int q = 0; q < quad.size(); ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<1, dim> phi_i = fe_values[displacements].value(i, q);
          const SymmetricTensor<2, dim> eps_i =
              fe_values[displacements].symmetric_gradient(i, q);
          const double div_i = fe_values[displacements].divergence(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const Tensor<1, dim> phi_j = fe_values[displacements].value(j, q);
            const SymmetricTensor<2, dim> eps_j =
                fe_values[displacements].symmetric_gradient(j, q);
            const double div_j = fe_values[displacements].divergence(j, q);
            const double a =
                (lambda_diff * div_i * div_j + 2. * mu_diff * (eps_i * eps_j)) *
                fe_values.JxW(q);
            const double m = (phi_i * phi_j) * fe_values.JxW(q);
            cm_Aug(i, j) += a + mass_weight * m;
            // Drop the vertex<->interior coupling in the preconditioner
            if (!use_bubble_block_prec ||
                imm_interior_dof[i] == imm_interior_dof[j])
              cm_prec(i, j) += a + mass_weight * m;
          }
          double rhs_dot_phi = 0.;
          for (unsigned int d = 0; d < dim; ++d)
            rhs_dot_phi += rhs_values[q][d] * phi_i[d];
          cell_rhs(i) += rhs_dot_phi * fe_values.JxW(q);
        }
      }
      cell->get_dof_indices(ldi);
      constraints_fg.distribute_local_to_global(cm_Aug, cell_rhs, ldi,
                                                A_fg_plus_scaled_M, rhs_owned);
      if (use_bubble_block_prec)
        constraints_fg.distribute_local_to_global(cm_prec, ldi, A22_prec);
    }
  A_fg_plus_scaled_M.compress(VectorOperation::add);
  if (use_bubble_block_prec)
    A22_prec.compress(VectorOperation::add);
  rhs_owned.compress(VectorOperation::add);
  system_rhs_block.block(1) = rhs_owned;
}

// Q0 multiplier only: add the *projected* AL mass
//
//   gamma_2 * M W^{-1} M^T = gamma_2 * sum_{K,c} (1/|K|) m_{K,c} m_{K,c}^T,
//   m_{K,c}[j] = int_K phi_j^{imm,(c)} dx,   |K| = int_K 1 dx,
//
// to the (1,1) block. Because W is diagonal for a cellwise-constant
// multiplier, every term is a rank-`dim` outer product over the dofs of a
// SINGLE immersed cell: no global triple product is ever formed and the
// result stays inside the standard immersed sparsity pattern.
//
// This is what makes the Q0 augmentation *consistent*: together with the
// matching (0,1)/(1,0)/(0,0) blocks assembled in
// `assemble_coupling_and_augment_bg`, the whole AL contribution is
// gamma * ||P_h(u_1 - u_2)||^2_{L^2(Omega_2)}, which vanishes identically
// on the discrete constraint Ct^T u_1 = M^T u_2.
template <int dim>
void ElasticityInterfaceDLMParallel<dim>::add_projected_mass_fg(
    double gamma_2_scaled) {
  if (!use_q0_multiplier)
    return;
  TimerOutput::Scope t(computing_timer, "Projected AL mass (1,1) block");
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;
  const Quadrature<dim> quad = make_quadrature_fg(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_fg_ptr, fe_fg, quad,
                          update_values | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  const unsigned int n_mult_dofs = fe_mult.n_dofs_per_cell();

  std::vector<unsigned int> comp(dofs_per_cell);
  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    comp[i] = fe_fg.system_to_component_index(i).first;
  std::vector<unsigned int> mult_comp(n_mult_dofs);
  for (unsigned int l = 0; l < n_mult_dofs; ++l)
    mult_comp[l] = fe_mult.system_to_component_index(l).first;

  // Also build M^T (row (K,c) = m_{K,c}), which the projected (0,1)/(1,0)
  // blocks need in `assemble_coupling_and_augment_bg`. Assembling it here
  // reuses this loop and keeps the constraint handling identical to the
  // background-side twin `bg_mult_integral_T`.
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
  FullMatrix<double> cm_prec(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cm_mT(n_mult_dofs, dofs_per_cell);
  std::vector<double> mvec(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  std::vector<types::global_dof_index> ldi_mult(n_mult_dofs);

  typename DoFHandler<dim>::active_cell_iterator cell_mult_it =
      dof_handler_mult.begin_active();
  for (const auto &cell : dof_handler_fg.active_cell_iterators()) {
    const auto cell_mult = cell_mult_it;
    ++cell_mult_it;
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      // All immersed elements used here are primitive, so shape_value(i, q)
      // is the value of the single non-zero component of phi_i.
      std::fill(mvec.begin(), mvec.end(), 0.);
      double measure = 0.;
      for (unsigned int q = 0; q < quad.size(); ++q) {
        measure += fe_values.JxW(q);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          mvec[i] += fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
      Assert(measure > 0.,
             ExcMessage("Immersed cell has non-positive measure."));
      const double scale = gamma_2_scaled / measure;

      cm = 0;
      cm_prec = 0;
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          if (comp[i] == comp[j]) {
            const double v = scale * mvec[i] * mvec[j];
            cm(i, j) = v;
            if (!use_bubble_block_prec ||
                imm_interior_dof[i] == imm_interior_dof[j])
              cm_prec(i, j) = v;
          }

      cm_mT = 0;
      for (unsigned int l = 0; l < n_mult_dofs; ++l)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          if (comp[j] == mult_comp[l])
            cm_mT(l, j) = mvec[j];

      cell->get_dof_indices(ldi);
      cell_mult->get_dof_indices(ldi_mult);
      constraints_fg.distribute_local_to_global(cm, ldi, A_fg_plus_scaled_M);
      if (use_bubble_block_prec)
        constraints_fg.distribute_local_to_global(cm_prec, ldi, A22_prec);
      constraints_mult.distribute_local_to_global(
          cm_mT, ldi_mult, constraints_fg, ldi, imm_mult_integral_T);
    }
  }
  A_fg_plus_scaled_M.compress(VectorOperation::add);
  if (use_bubble_block_prec)
    A22_prec.compress(VectorOperation::add);
  imm_mult_integral_T.compress(VectorOperation::add);
}

template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_mass_fg() {
  TimerOutput::Scope t(computing_timer,
                       "Assemble cross-mass (immersed x multiplier)");
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;
  // Use the immersed quadrature (matches the FE degree of the primal
  // basis); the Q0 multiplier shape is exact on it.
  const Quadrature<dim> quad = make_quadrature_fg(fe_fg.degree + 1);
  FEValues<dim> fe_values_fg(*mapping_fg_ptr, fe_fg, quad,
                             update_values | update_JxW_values);
  FEValues<dim> fe_values_mult(*mapping_fg_ptr, fe_mult, quad, update_values);
  const unsigned int dpc_fg = fe_fg.n_dofs_per_cell();
  const unsigned int dpc_mult = fe_mult.n_dofs_per_cell();
  FullMatrix<double> cm(dpc_fg, dpc_mult);
  std::vector<types::global_dof_index> ldi_fg(dpc_fg);
  std::vector<types::global_dof_index> ldi_mult(dpc_mult);
  const FEValuesExtractors::Vector displacements(0);
  typename DoFHandler<dim>::active_cell_iterator
      cell_fg = dof_handler_fg.begin_active(),
      end_fg = dof_handler_fg.end(),
      cell_mult = dof_handler_mult.begin_active();
  for (; cell_fg != end_fg; ++cell_fg, ++cell_mult)
    if (cell_fg->is_locally_owned()) {
      fe_values_fg.reinit(cell_fg);
      fe_values_mult.reinit(cell_mult);
      cm = 0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dpc_fg; ++i) {
          const Tensor<1, dim> phi_i = fe_values_fg[displacements].value(i, q);
          for (unsigned int j = 0; j < dpc_mult; ++j) {
            const Tensor<1, dim> phi_j =
                fe_values_mult[displacements].value(j, q);
            cm(i, j) += (phi_i * phi_j) * fe_values_fg.JxW(q);
          }
        }
      cell_fg->get_dof_indices(ldi_fg);
      cell_mult->get_dof_indices(ldi_mult);
      constraints_fg.distribute_local_to_global(cm, ldi_fg, constraints_mult,
                                                ldi_mult, mass_matrix_fg);
    }
  mass_matrix_fg.compress(VectorOperation::add);
}

// Diagonal (Q0) or standard immersed mass on the multiplier space. Used as
// W in the preconditioner (and to assemble the AL augmentation when the
// multiplier is Q0).
template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_mass_mult() {
  TimerOutput::Scope t(computing_timer, "Assemble multiplier mass W");
  const auto &fe_mult = *fe_mult_ptr;
  const Quadrature<dim> quad = make_quadrature_fg(fe_mult.degree + 1);
  FEValues<dim> fe_values(*mapping_fg_ptr, fe_mult, quad,
                          update_values | update_JxW_values);
  const unsigned int dpc = fe_mult.n_dofs_per_cell();
  FullMatrix<double> cm(dpc, dpc);
  std::vector<types::global_dof_index> ldi(dpc);
  const FEValuesExtractors::Vector displacements(0);
  for (const auto &cell : dof_handler_mult.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm = 0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dpc; ++i) {
          const Tensor<1, dim> phi_i = fe_values[displacements].value(i, q);
          for (unsigned int j = 0; j < dpc; ++j) {
            const Tensor<1, dim> phi_j = fe_values[displacements].value(j, q);
            cm(i, j) += (phi_i * phi_j) * fe_values.JxW(q);
          }
        }
      cell->get_dof_indices(ldi);
      constraints_mult.distribute_local_to_global(cm, ldi, mass_matrix_mult);
    }
  mass_matrix_mult.compress(VectorOperation::add);
}

// Particle-based assembly of:
//   * Ct = coupling_matrix          (n_bg x n_mult)
//   * coupling_matrix_01            (n_bg x n_imm)
//   * coupling_matrix_10            (n_imm x n_bg)   transpose of _01
//   * bg_mult_integral_T            (n_mult x n_bg)  Q0 only
// and, for Q1-Q1-Q1, the in-place AL augmentation of A_bg.
//
// For Q1-Bubble-Q0 / Q2-Q2-Q0 the augmentation of A_bg is performed
// outside this loop by a K-driven outer-product pass that re-uses the
// snapshot K_bg_pre_aug taken in solve() before this routine is called.
//
// For primitive FESystem-of-FE_Q (or FE_SimplexP, FE_Q_Bubbles, FE_DGQ,
// ...) finite elements, every shape function has a single non-zero
// component. Hence the dot products (phi_i_bg . phi_j_imm) and
// (phi_i_bg . phi_j_bg) reduce to scalar shape values multiplied
// together, conditional on the components matching.
//
// Particle properties layout (Q1-Q1-Q1):
//   [0]                                        : JxW
//   [1                  .. n_imm_dofs]         : immersed scalar shape values
//   [1+n_imm_dofs       .. 2*n_imm_dofs]       : immersed global dof indices
// Particle properties layout (Q1-Bubble-Q0 / Q2-Q2-Q0) appends:
//   [mult_off           .. n_mult_dofs]        : multiplier global dof indices
// where mult_off = 1 + 2*n_imm_dofs.

template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_coupling_and_augment_bg(
    double gamma_1_scaled) {
  TimerOutput::Scope t(computing_timer,
                       "Coupling assembly and bg augmentation (particles)");
  const auto &fe_bg = *fe_bg_ptr;
  const auto &fe_fg = *fe_fg_ptr;
  const auto &fe_mult = *fe_mult_ptr;

  const unsigned int n_imm_dofs = fe_fg.n_dofs_per_cell();
  const unsigned int n_bg_dofs = fe_bg.n_dofs_per_cell();
  const unsigned int n_mult_dofs = fe_mult.n_dofs_per_cell();
  const unsigned int mult_off = 1 + 2 * n_imm_dofs;
  const unsigned int n_props =
      mult_off + (use_q0_multiplier ? n_mult_dofs : 0u);

  // Cell-independent component maps (see the layout comment above).
  std::vector<unsigned int> imm_comp(n_imm_dofs);
  for (unsigned int i = 0; i < n_imm_dofs; ++i)
    imm_comp[i] = fe_fg.system_to_component_index(i).first;
  std::vector<unsigned int> mult_comp_fe(n_mult_dofs);
  for (unsigned int i = 0; i < n_mult_dofs; ++i)
    mult_comp_fe[i] = fe_mult.system_to_component_index(i).first;

  // Extract the multiplier global dof indices for a particle. For
  // Q1-Q1-Q1, the multiplier basis coincides with the immersed primal
  // basis on the same triangulation so the global dof indices are
  // numerically identical and we re-use the imm-dof slot in the particle
  // properties; for Q0 they are stored explicitly.
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

  Particles::ParticleHandler<dim> particles;
  {
    std::vector<BoundingBox<dim>> all_local_boxes;
    all_local_boxes.reserve(tria_bg->n_locally_owned_active_cells());
    for (const auto &cell : tria_bg->active_cell_iterators())
      if (cell->is_locally_owned())
        all_local_boxes.emplace_back(cell->bounding_box());
    const auto tree = pack_rtree(all_local_boxes);
    const auto local_boxes =
        extract_rtree_level(tree, 2); // TODO: parametrize level of extraction
    pcout << "Number of levels: " << n_levels(tree) << std::endl;
    const auto global_boxes =
        Utilities::MPI::all_gather(mpi_communicator, local_boxes);

    particles.initialize(*tria_bg, *mapping_bg_ptr, n_props);

    const Quadrature<dim> imm_quad =
        make_quadrature_fg(2 * std::max(fe_bg.degree, fe_fg.degree) + 1);
    FEValues<dim> fe_v_imm(*mapping_fg_ptr, fe_fg, imm_quad,
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

    pcout << "Started insertion of particles..." << std::endl;
    double start, stop;
    start = MPI_Wtime();
    particles.insert_global_particles(points, global_boxes, props);
    stop = MPI_Wtime();
    pcout << "Finalized insertion of particles, time for particle insertion: "
          << stop - start << " [s]." << std::endl;
  }

  pcout << "Inserted " << particles.n_global_particles()
        << " coupling particles." << std::endl;

  // Cache background dof component indices.
  std::vector<unsigned int> bg_comp(n_bg_dofs);
  for (unsigned int i = 0; i < n_bg_dofs; ++i)
    bg_comp[i] = fe_bg.system_to_component_index(i).first;

  std::vector<types::global_dof_index> bg_ldi(n_bg_dofs);

  // Build writable row IndexSets for the immersed and multiplier sides:
  // in parallel a particle owned by this rank may carry imm / mult dof
  // indices that are off-rank in the relevant_fg / relevant_mult sets;
  // without writable rows, DSP::add silently drops the entry and Trilinos
  // then errors out on the off-pattern write.
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

  // First pass: sparsity patterns.
  DynamicSparsityPattern dsp_C(dof_handler_bg.n_dofs(),
                               dof_handler_mult.n_dofs(), relevant_bg);
  DynamicSparsityPattern dsp_C_01(dof_handler_bg.n_dofs(),
                                  dof_handler_fg.n_dofs(), relevant_bg);
  DynamicSparsityPattern dsp_C_10(dof_handler_fg.n_dofs(),
                                  dof_handler_bg.n_dofs(), imm_writable);
  DynamicSparsityPattern dsp_bg_mult_T(dof_handler_mult.n_dofs(),
                                       dof_handler_bg.n_dofs(), mult_writable);
  {
    auto particle = particles.begin();
    while (particle != particles.end()) {
      const auto &cell = particle->get_surrounding_cell();
      typename DoFHandler<dim>::cell_iterator dh_cell(*cell, &dof_handler_bg);
      dh_cell->get_dof_indices(bg_ldi);
      const auto pic = particles.particles_in_cell(cell);

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
        if (!use_q0_multiplier)
          for (const auto imm_j : imm_cols)
            dsp_C_10.add(imm_j, bg_i);
        else
          for (const auto mu_k : mult_cols)
            dsp_bg_mult_T.add(mu_k, bg_i);
      }
      // Use constraint-aware additions so that hanging-node entries on the
      // bg (introduced by local interface refinement) are included in the
      // sparsity pattern of the coupling matrices.
      constraints_bg.add_entries_local_to_global(bg_ldi, constraints_mult,
                                                 mult_cols, dsp_C);
      // For a Q0 multiplier the (0,1)/(1,0) blocks are NOT the plain
      // variational couplings but their projected counterparts
      // Ct W^{-1} M^T / M W^{-1} Ct^T, whose sparsity is only known once
      // `bg_mult_integral_T` has been assembled. They are built further
      // down, cellwise, so nothing is added here.
      if (!use_q0_multiplier) {
        constraints_bg.add_entries_local_to_global(bg_ldi, constraints_fg,
                                                   imm_cols, dsp_C_01);
        // For coupling_matrix_10 (imm rows x bg cols) the bg columns may be
        // hanging nodes; their distribution introduces additional column
        // entries via the master dofs. add_entries_local_to_global handles
        // this symmetrically.
        constraints_fg.add_entries_local_to_global(imm_cols, constraints_bg,
                                                   bg_ldi, dsp_C_10);
      } else {
        // bg_mult_integral_T (mult rows x bg cols): same hanging-node
        // consideration on the bg columns.
        constraints_mult.add_entries_local_to_global(mult_cols, constraints_bg,
                                                     bg_ldi, dsp_bg_mult_T);
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
  } else {
    SparsityTools::distribute_sparsity_pattern(dsp_bg_mult_T, owned_mult,
                                               mpi_communicator, mult_writable);
    bg_mult_integral_T.reinit(owned_mult, owned_bg, dsp_bg_mult_T,
                              mpi_communicator);
  }

  // Second pass: assemble coupling matrix values and (Q1 only) AL.
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

        // Background scalar shape values at the particle ref location.
        std::vector<double> bg_vals(n_bg_dofs);
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          bg_vals[i] = fe_bg.shape_value(i, ref);

        // Q1-Q1-Q1: usual in-place AL augmentation.
        if (!use_q0_multiplier) {
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_bg_dofs; ++j)
              if (bg_comp[i] == bg_comp[j])
                local_AL(i, j) +=
                    gamma_1_scaled * bg_vals[i] * bg_vals[j] * JxW;
        }

        std::vector<types::global_dof_index> mult_idx(n_mult_dofs);
        mult_dofs_from_props(pp, mult_idx);
        const std::vector<unsigned int> &mult_comp =
            use_q0_multiplier ? mult_comp_fe : imm_comp;

        std::vector<types::global_dof_index> imm_cols(n_imm_dofs);
        for (unsigned int j = 0; j < n_imm_dofs; ++j)
          imm_cols[j] =
              static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + j]);

        // coupling_matrix (n_bg x n_mult).
        local_C = 0;
        if (use_q0_multiplier) {
          // Each mult basis fn has scalar shape = 1 in component
          // mult_comp[j]; the contribution is bg_vals[i] * 1 * JxW
          // when bg_comp[i] == mult_comp[j].
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_mult_dofs; ++j)
              if (bg_comp[i] == mult_comp[j])
                local_C(i, j) = bg_vals[i] * JxW;
        } else {
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_mult_dofs; ++j)
              if (bg_comp[i] == mult_comp[j])
                local_C(i, j) = bg_vals[i] * pp[1 + j] * JxW;
        }

        // coupling_matrix_01 / _10 (against the immersed primal basis).
        // Only for the triples whose multiplier equals the immersed primal
        // space; the Q0 path builds the projected versions cellwise below.
        if (!use_q0_multiplier) {
          local_C_01 = 0;
          local_C_10 = 0;
          for (unsigned int i = 0; i < n_bg_dofs; ++i)
            for (unsigned int j = 0; j < n_imm_dofs; ++j) {
              if (bg_comp[i] == imm_comp[j]) {
                const double val = bg_vals[i] * pp[1 + j] * JxW;
                local_C_01(i, j) = val;
                local_C_10(j, i) = val;
              }
            }
        }

        // bg_mult_integral_T (Q0 only). Entry [K_c, i] = int_K phi_i^{(c)}.
        if (use_q0_multiplier) {
          local_bg_mult_T = 0;
          for (unsigned int j = 0; j < n_mult_dofs; ++j)
            for (unsigned int i = 0; i < n_bg_dofs; ++i)
              if (bg_comp[i] == mult_comp[j])
                local_bg_mult_T(j, i) = bg_vals[i] * JxW;
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
      if (!use_q0_multiplier)
        constraints_bg.distribute_local_to_global(local_AL, bg_ldi, A_bg);
      particle = pic.end();
    }
  }
  coupling_matrix.compress(VectorOperation::add);
  if (!use_q0_multiplier) {
    A_bg.compress(VectorOperation::add);
    coupling_matrix_01.compress(VectorOperation::add);
    coupling_matrix_10.compress(VectorOperation::add);
  } else {
    bg_mult_integral_T.compress(VectorOperation::add);
  }

  // Q0 path: direct assembly of the three projected augmentation
  // blocks, each as a per-(cell, component) outer product. Writing
  //   c_{K,c}[i] = int_K phi_i^{bg,(c)}   (row (K,c) of bg_mult_integral_T)
  //   m_{K,c}[j] = int_K phi_j^{imm,(c)}  (row (K,c) of imm_mult_integral_T)
  //   |K|        = diagonal of the multiplier mass W at dof (K,c)
  // we assemble
  //   A_bg              += gamma_1 * sum (1/|K|) c_{K,c} c_{K,c}^T
  //   coupling_matrix_01 =          sum (1/|K|) c_{K,c} m_{K,c}^T
  //   coupling_matrix_10 =          sum (1/|K|) m_{K,c} c_{K,c}^T
  // i.e. exactly gamma_1 * Ct W^{-1} Ct^T, Ct W^{-1} M^T and M W^{-1} Ct^T
  // (the operator applies the -gamma_1 / -gamma_2 factors to the latter
  // two). Together with the projected (1,1) mass added by
  // `add_projected_mass_fg`, the augmentation is the variational form
  // gamma * ||P_h(u_1 - u_2)||^2_{L^2(Omega_2)}, which vanishes on the
  // discrete constraint manifold, so the augmented system has exactly
  // the same solution as the unaugmented saddle point. Since W is
  // diagonal, none of this ever forms a global triple product.

  if (use_q0_multiplier) {
    Assert(imm_mult_integral_T.m() == dof_handler_mult.n_dofs(),
           ExcMessage("imm_mult_integral_T (M^T) has not been assembled; "
                      "add_projected_mass_fg must run before this routine."));

    // Column supports of Ct^T and M^T for every locally owned multiplier
    // dof, i.e. the background / immersed dofs that see cell K.
    std::vector<std::vector<types::global_dof_index>> S_per_K, T_per_K;
    std::vector<types::global_dof_index> ordered_K;
    S_per_K.reserve(owned_mult.n_elements());
    T_per_K.reserve(owned_mult.n_elements());
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
        if (S_K.empty())
          continue;

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
      std::sort(touched_bg.begin(), touched_bg.end());
      touched_bg.erase(std::unique(touched_bg.begin(), touched_bg.end()),
                       touched_bg.end());
      bg_writable.add_indices(touched_bg.begin(), touched_bg.end());
      bg_writable.compress();
    }

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

    // Snapshot A_bg (=K_bg_pre_aug) and reinit with extended sparsity.
    A_bg.reinit(owned_bg, owned_bg, dsp_A_bg_ext, mpi_communicator);
    for (auto it = K_bg_pre_aug.begin(); it != K_bg_pre_aug.end(); ++it)
      A_bg.set(it->row(), it->column(), it->value());
    A_bg.compress(VectorOperation::insert);

    // The snapshot has done its job, release it before the AMG
    // hierarchies are built
    K_bg_pre_aug.clear();
    coupling_matrix_01.reinit(owned_bg, owned_fg, dsp_C_01_proj,
                              mpi_communicator);
    coupling_matrix_10.reinit(owned_fg, owned_bg, dsp_C_10_proj,
                              mpi_communicator);

    for (std::size_t idx = 0; idx < ordered_K.size(); ++idx) {
      const auto K = ordered_K[idx];
      const auto &S_K = S_per_K[idx];
      const auto &T_K = T_per_K[idx];
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

    // Ct^T and M^T were only needed to build the outer products above.
    bg_mult_integral_T.clear();
    imm_mult_integral_T.clear();
  }
}

template <int dim> unsigned int ElasticityInterfaceDLMParallel<dim>::solve() {
  const double h_immersed = GridTools::maximal_cell_diameter(*tria_fg);
  const double inv_h2 = 1. / (h_immersed * h_immersed);
  const double gamma_1 = parameters.gamma_AL_background * inv_h2;
  const double gamma_2 = parameters.gamma_AL_immersed * inv_h2;

  assemble_bg();
  assemble_fg(gamma_2);
  assemble_mass_fg();
  assemble_mass_mult();
  // Q0 multiplier only: completes the (1,1) block with the projected AL
  // mass gamma_2 * M W^{-1} M^T and builds M^T for the projected
  // off-diagonal blocks assembled below. No-op for Q1-Q1-Q1.
  add_projected_mass_fg(gamma_2);

  // Snapshot the bg stiffness BEFORE the AL augmentation, but ONLY for the
  // Q0 path, which is the only consumer: it re-fills A_bg after reinit-ing
  // with the extended sparsity.
  if (use_q0_multiplier) {
    K_bg_pre_aug.reinit(A_bg);
    K_bg_pre_aug.copy_from(A_bg);
  }

  assemble_coupling_and_augment_bg(gamma_1);

  LA::MPI::PreconditionJacobi mass_prec;
  mass_prec.initialize(mass_matrix_mult);
  using InvW_t = ALParallel::InvMassOp<LA::MPI::Vector, LA::MPI::SparseMatrix,
                                       LA::MPI::PreconditionJacobi>;
  InvW_t invW_op(mass_matrix_mult, mass_prec, 1000, 1e-12, owned_mult,
                 mpi_communicator);

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

  // For elasticity AMG, hint the near-null-space using the full set of
  // *rigid body modes* (dim translations + dim*(dim-1)/2 rotations).
  const LA::MPI::SparseMatrix &A22_amg_matrix =
      use_bubble_block_prec ? A22_prec : A_fg_plus_scaled_M;

  LA::MPI::PreconditionAMG amg_A11, amg_A22;
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.constant_modes_values = DoFTools::extract_rigid_body_modes(
        *mapping_bg_ptr, dof_handler_bg, ComponentMask());
    data.elliptic = true;

    data.higher_order_elements =
        parameters.background_space_finite_element_degree > 1;
    amg_A11.initialize(A_bg, data);
  }
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.constant_modes_values = DoFTools::extract_rigid_body_modes(
        *mapping_fg_ptr, dof_handler_fg, ComponentMask());
    data.elliptic = true;
    data.higher_order_elements =
        parameters.immersed_space_finite_element_degree > 1;

    // The immersed block carries no Dirichlet data, so its rigid-body
    // near-kernel is only weakly shifted by the AL mass term
    if (use_bubble_block_prec)
      data.smoother_sweeps = 3;
    amg_A22.initialize(A22_amg_matrix, data);
  }

  typename SolverFGMRES<LA::MPI::BlockVector>::AdditionalData data_fgmres;
  data_fgmres.max_basis_size = 30;
  SolverFGMRES<LA::MPI::BlockVector> solver_fgmres(
      parameters.outer_solver_control, data_fgmres);

  system_rhs_block.block(2) = 0.;
  system_solution_block = 0.;

  if (parameters.use_modified_AL_preconditioner) {
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

    {
      TimerOutput::Scope t(computing_timer, "Solve system");
      solver_fgmres.solve(system_operator, system_solution_block,
                          system_rhs_block, prec_AL);
    }

    inner_conn_A11.disconnect();
    last_inner_iters_A11 = inner_iters_A11;
    last_n_A11_applications = n_A11_applications;
  } else {
    AssertThrow(std::abs(parameters.gamma_AL_background -
                         parameters.gamma_AL_immersed) < 1e-12,
                ExcMessage("Ideal AL requires gamma_1 == gamma_2."));

    pcout << "\t *** USING IDEAL AL PRECONDITIONER (test only) ***"
          << std::endl;

    ALParallel::Block2x2<LA::MPI::Vector, LA::MPI::BlockVector,
                         LA::MPI::SparseMatrix, InvW_t>
        Aug_mat{A_bg,
                A_fg_plus_scaled_M,
                coupling_matrix_01,
                coupling_matrix_10,
                gamma_1,
                gamma_2,
                owned_bg,
                owned_fg,
                mpi_communicator};

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
  pcout << "||u_1|| = " << system_solution_block.block(0).l2_norm()
        << "   ||u_2|| = " << system_solution_block.block(1).l2_norm()
        << "   ||lambda|| = " << system_solution_block.block(2).l2_norm()
        << std::endl;

  // Average number of inner (P)CG iterations performed to invert the (1,1)
  // block A_bg of the preconditioner. Reported as iterations per
  // (1,1)-block solve.
  const double avg_inner_A11 =
      (last_n_A11_applications == 0)
          ? 0.0
          : double(last_inner_iters_A11) / double(last_n_A11_applications);
  pcout << "Average inner PCG iterations to invert the (1,1) block: "
        << avg_inner_A11 << " (" << last_inner_iters_A11 << " total over "
        << last_n_A11_applications << " solves)." << std::endl;

  convergence_table.add_value("cells", tria_bg->n_global_active_cells());
  convergence_table.add_value("DoF bg", dof_handler_bg.n_dofs());
  convergence_table.add_value("DoF imm", dof_handler_fg.n_dofs());
  convergence_table.add_value("gamma1/h^2", gamma_1);
  if (parameters.use_modified_AL_preconditioner)
    convergence_table.add_value("gamma2/h^2", gamma_2);
  convergence_table.add_value("outer it", n_outer);
  convergence_table.add_value("inner A11 tot", last_inner_iters_A11);
  convergence_table.add_value("inner A11/solve", avg_inner_A11);
  convergence_table.set_precision("inner A11/solve", 1);
  return n_outer;
}

template <int dim>
void ElasticityInterfaceDLMParallel<dim>::output_results(
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

  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      vec_interp(dim, DataComponentInterpretation::component_is_part_of_vector);
  const std::vector<std::string> u_names(dim, "u");
  const std::vector<std::string> u2_names(dim, "u2");
  const std::vector<std::string> lam_names(dim, "lambda");

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_bg);
    data_out.add_data_vector(u_bg_ghosted, u_names, DataOut<dim>::type_dof_data,
                             vec_interp);
    Vector<float> subdomain(tria_bg->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_bg->locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(*mapping_bg_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-bg", cycle, mpi_communicator,
                                        2, 0);
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_fg);
    data_out.add_data_vector(u_fg_ghosted, u2_names,
                             DataOut<dim>::type_dof_data, vec_interp);
    data_out.build_patches(*mapping_fg_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-imm", cycle, mpi_communicator,
                                        2, 0);
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_mult);
    data_out.add_data_vector(lam_ghosted, lam_names,
                             DataOut<dim>::type_dof_data, vec_interp);
    data_out.build_patches(*mapping_fg_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-mult", cycle,
                                        mpi_communicator, 2, 0);
  }

  if (this_mpi_process == 0) {
    convergence_table.write_text(
        std::cout, TableHandler::TextOutputFormat::org_mode_table);
  }
}

template <int dim> void ElasticityInterfaceDLMParallel<dim>::run() {
  for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
       ++cycle) {
    pcout << "==============================================================="
          << "\nRefinement cycle: " << cycle << std::endl;
    // Note: clearing/rebuilding of both triangulations is handled inside
    // `generate_grids` (which refines incrementally for hex p::d::T and for
    // cached-serial p::f::T simplex meshes).
    generate_grids(cycle);
    system_setup();
    solve();
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

    constexpr unsigned int dim = 3;
    ProblemParameters<dim> parameters;
    std::string parameter_file =
        (argc > 1) ? argv[1] : "parameters_elasticity_op.prm";
    ParameterAcceptor::initialize(parameter_file,
                                  "used_parameters_elasticity_op.prm");
    deallog.depth_console(
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ? 10 : 0);

    ElasticityInterfaceDLMParallel<dim> solver(parameters);
    solver.run();
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
