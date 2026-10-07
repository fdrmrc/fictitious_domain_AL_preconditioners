// Parallel (MPI) version of the elliptic interface fictitious-domain problem
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

#include <deal.II/fe/fe_q.h>
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
// We avoid the LinearOperator composition (op1 * op2) because in deal.II the
// resulting composite payload falls back to EmptyPayload, which crashes when
// it tries to allocate intermediate Trilinos vectors (uninitialised IndexSet).
//
// The augmented system reproduces exactly the one built in
// elliptic_interface.cc when "Use operator version = true":
//   (0,0) = A_bg                            (AL term already added in place)
//   (0,1) = -gamma1 * Ct * invW * M
//   (0,2) =  Ct
//   (1,0) = -gamma2 * M  * invW * C
//   (1,1) =  A_fg + gamma2 * M * invW * M   (A_fg is the un-augmented immersed)
//   (1,2) = -M
//   (2,0) =  C   = Ct^T
//   (2,1) = -M
//   (2,2) =  0
// where invW = M^{-1} (here applied via CG instead of a diagonal lump).

template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp>
struct System3x3 {
  const SparseMatrixType &A_bg; // already AL-augmented in place
  const SparseMatrixType &A_fg; // (beta_2 - beta_1) (grad u, grad v)
  const SparseMatrixType &M;    // immersed mass matrix
  const SparseMatrixType &Ct;   // n_bg x n_imm coupling matrix
  const InvWOp &invW;           // action of M^{-1}
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm), ti2(owned_fg, comm);
    VectorType tb(owned_bg, comm);

    // Row 0: A_bg * x0  - gamma1 * Ct * invW * M * x1  + Ct * x2
    A_bg.vmult(dst.block(0), src.block(0));
    M.vmult(ti, src.block(1));
    invW.vmult(ti2, ti);
    Ct.vmult(tb, ti2);
    dst.block(0).add(-gamma1, tb);
    Ct.vmult(tb, src.block(2));
    dst.block(0) += tb;

    // Row 1: -gamma2 * M * invW * Ct^T * x0
    //        + A_fg * x1 + gamma2 * M * invW * M * x1
    //        - M * x2
    Ct.Tvmult(ti, src.block(0));
    invW.vmult(ti2, ti);
    M.vmult(dst.block(1), ti2);
    dst.block(1) *= -gamma2;
    A_fg.vmult_add(dst.block(1), src.block(1));
    M.vmult(ti, src.block(1));
    invW.vmult(ti2, ti);
    M.vmult(ti, ti2);
    dst.block(1).add(gamma2, ti);
    M.vmult(ti, src.block(2));
    dst.block(1) -= ti;

    // Row 2: Ct^T * x0 - M * x1
    Ct.Tvmult(dst.block(2), src.block(0));
    M.vmult(ti, src.block(1));
    dst.block(2) -= ti;
  }
};

// 2x2 augmented block used by the ideal AL preconditioner (rows 0,1 of the
// system above, with x2 = 0).
template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp>
struct Block2x2 {
  const SparseMatrixType &A_bg;
  const SparseMatrixType &A_fg;
  const SparseMatrixType &M;
  const SparseMatrixType &Ct;
  const InvWOp &invW;
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm), ti2(owned_fg, comm);
    VectorType tb(owned_bg, comm);

    A_bg.vmult(dst.block(0), src.block(0));
    M.vmult(ti, src.block(1));
    invW.vmult(ti2, ti);
    Ct.vmult(tb, ti2);
    dst.block(0).add(-gamma1, tb);

    Ct.Tvmult(ti, src.block(0));
    invW.vmult(ti2, ti);
    M.vmult(dst.block(1), ti2);
    dst.block(1) *= -gamma2;
    A_fg.vmult_add(dst.block(1), src.block(1));
    M.vmult(ti, src.block(1));
    invW.vmult(ti2, ti);
    M.vmult(ti, ti2);
    dst.block(1).add(gamma2, ti);
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
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType lam(owned_fg, comm);
    invW.vmult(lam, src.block(2));
    lam *= -gamma;

    VectorType rhs1(owned_fg, comm);
    M.vmult(rhs1, lam);
    rhs1 += src.block(1);
    dst.block(1) = 0.;
    dst.block(1) = A22_inv * rhs1;

    VectorType rhs0(owned_bg, comm);
    VectorType ti(owned_fg, comm), ti2(owned_fg, comm);
    VectorType tb(owned_bg, comm);
    M.vmult(ti, dst.block(1));
    invW.vmult(ti2, ti);
    Ct.vmult(tb, ti2);
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
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType lam(owned_fg, comm);
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
      M.vmult(ti, lam);
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

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------
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

  bool use_modified_AL_preconditioner = false;

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
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
    : ParameterAcceptor("Elliptic Interface Problem<" +
                        Utilities::int_to_string(dim) + ">/"),
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
}

// -----------------------------------------------------------------------------
// Parallel elliptic interface solver (operator-form AL)
// -----------------------------------------------------------------------------
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
  void assemble_coupling_and_augment_bg(double gamma_1_h);
  unsigned int solve();
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
  // depends on a run-time parameter.
  std::unique_ptr<FiniteElement<dim>> fe_bg_ptr;
  std::unique_ptr<FiniteElement<dim>> fe_fg_ptr;
  std::unique_ptr<Mapping<dim>> mapping_ptr;
  // Quadrature factory: QGauss for hex, QGaussSimplex for simplex.
  std::function<Quadrature<dim>(unsigned int)> make_quadrature;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;

  IndexSet owned_bg, relevant_bg;
  IndexSet owned_fg, relevant_fg;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;

  // Background stiffness (will be augmented in-place by particle quadrature).
  LA::MPI::SparseMatrix A_bg;
  // Immersed stiffness (beta_2 - beta_1)(grad u, grad v) + gamma_2*M (when
  // assembling the augmented A22 preconditioner block).
  LA::MPI::SparseMatrix A_fg;
  LA::MPI::SparseMatrix A_fg_plus_scaled_M;
  // Immersed mass matrix M.
  LA::MPI::SparseMatrix mass_matrix_fg;
  // Coupling matrix Ct of size n_bg x n_imm (i.e. integral of phi_bg*psi_imm
  // over the immersed mesh). C is its transpose.
  LA::MPI::SparseMatrix coupling_matrix;

  LA::MPI::BlockVector system_rhs_block;
  LA::MPI::BlockVector system_solution_block;

  // Persistent particle handler. When `keep_immersed_grid_fixed` is true and
  // the background is distributed (hex), the particle handler is initialised
  // once at cycle 0 and then transferred across bg refinement; subsequent
  // calls to `assemble_coupling_and_augment_bg` only need to assemble Ct and
  // the AL augmentation (sparsity + values) without re-inserting particles.
  Particles::ParticleHandler<dim> particles;
  bool particles_built = false;

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
  if (parameters.mesh_type == "hex") {
    tria_bg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
    tria_fg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
    fe_bg_ptr = std::make_unique<FE_Q<dim>>(
        parameters.background_space_finite_element_degree);
    fe_fg_ptr = std::make_unique<FE_Q<dim>>(
        parameters.immersed_space_finite_element_degree);
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
    mapping_ptr = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    // QGaussSimplex is implemented for 1..4 points per direction only.
    make_quadrature = [](unsigned int n) -> Quadrature<dim> {
      return QGaussSimplex<dim>(std::min(4u, std::max(1u, n)));
    };
  }
  dof_handler_bg.reinit(*tria_bg);
  dof_handler_fg.reinit(*tria_fg);
  AssertThrow(parameters.beta_1 > 0., ExcMessage("Beta_1 must be positive."));
  AssertThrow(parameters.beta_2 > parameters.beta_1,
              ExcMessage("Beta_2 must be greater than Beta_1."));
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

  dof_handler_bg.distribute_dofs(fe_bg);
  dof_handler_fg.distribute_dofs(fe_fg);

  owned_bg = dof_handler_bg.locally_owned_dofs();
  relevant_bg = DoFTools::extract_locally_relevant_dofs(dof_handler_bg);

  owned_fg = dof_handler_fg.locally_owned_dofs();
  relevant_fg = DoFTools::extract_locally_relevant_dofs(dof_handler_fg);

  // Constraints on background: Dirichlet b.c.
  constraints_bg.clear();
  constraints_bg.reinit(owned_bg, relevant_bg);
  DoFTools::make_hanging_node_constraints(dof_handler_bg, constraints_bg);
  for (const auto id : parameters.dirichlet_ids)
    VectorTools::interpolate_boundary_values(*mapping_ptr, dof_handler_bg, id,
                                             Functions::ZeroFunction<dim>(),
                                             constraints_bg);
  constraints_bg.close();

  // No constraints on immersed: Lagrange multiplier and immersed unknown have
  // their natural conditions.
  constraints_fg.clear();
  constraints_fg.reinit(owned_fg, relevant_fg);
  DoFTools::make_hanging_node_constraints(dof_handler_fg, constraints_fg);
  constraints_fg.close();

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
    mass_matrix_fg.reinit(owned_fg, owned_fg, dsp, mpi_communicator);
  }

  // Block vectors
  std::vector<IndexSet> owned_blocks = {owned_bg, owned_fg, owned_fg};
  std::vector<IndexSet> relevant_blocks = {relevant_bg, relevant_fg,
                                           relevant_fg};
  system_rhs_block.reinit(owned_blocks, mpi_communicator);
  system_solution_block.reinit(owned_blocks, mpi_communicator);

  pcout << "N DoF background: " << dof_handler_bg.n_dofs() << "\n"
        << "N DoF immersed:   " << dof_handler_fg.n_dofs() << std::endl;
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

  const double rho = gamma_2_scaled; // coefficient of (u,v)
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

template <int dim> void EllipticInterfaceDLMParallel<dim>::assemble_mass_fg() {
  TimerOutput::Scope t(computing_timer, "Assemble mass matrix (immersed)");
  const auto &fe_fg = *fe_fg_ptr;
  const Quadrature<dim> quad = make_quadrature(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_ptr, fe_fg, quad,
                          update_values | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  FullMatrix<double> cm(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  for (const auto &cell : dof_handler_fg.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm = 0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cm(i, j) += fe_values.shape_value(i, q) *
                        fe_values.shape_value(j, q) * fe_values.JxW(q);
      cell->get_dof_indices(ldi);
      constraints_fg.distribute_local_to_global(cm, ldi, mass_matrix_fg);
    }
  mass_matrix_fg.compress(VectorOperation::add);
}

// Particle-based assembly of the coupling matrix Ct (n_bg x n_imm) and, at the
// same time, of the AL augmentation of A_bg (gamma_1_scaled * \int phi_i*phi_j
// over the immersed mesh). Each immersed quadrature point becomes a particle
// carrying as properties:
//   [ JxW , imm_shape_values (n_imm_dofs) , imm_dof_indices (n_imm_dofs) ]
template <int dim>
void EllipticInterfaceDLMParallel<dim>::assemble_coupling_and_augment_bg(
    double gamma_1_scaled) {
  TimerOutput::Scope t(computing_timer,
                       "Coupling assembly and bg augmentation (particles)");
  const auto &fe_bg = *fe_bg_ptr;
  const auto &fe_fg = *fe_fg_ptr;

  const unsigned int n_imm_dofs = fe_fg.n_dofs_per_cell();
  const unsigned int n_props = 1 + 2 * n_imm_dofs;

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
    std::vector<types::global_dof_index> ldi(n_imm_dofs);
    for (const auto &cell : dof_handler_fg.active_cell_iterators())
      if (cell->is_locally_owned()) {
        fe_v_imm.reinit(cell);
        cell->get_dof_indices(ldi);
        const auto &qpts = fe_v_imm.get_quadrature_points();
        for (unsigned int q = 0; q < qpts.size(); ++q) {
          points.push_back(qpts[q]);
          std::vector<double> p(n_props, 0.);
          p[0] = fe_v_imm.JxW(q);
          for (unsigned int i = 0; i < n_imm_dofs; ++i)
            p[1 + i] = fe_v_imm.shape_value(i, q);
          for (unsigned int i = 0; i < n_imm_dofs; ++i)
            p[1 + n_imm_dofs + i] = static_cast<double>(ldi[i]);
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

  // First pass: compute the sparsity pattern for the coupling matrix Ct.
  const unsigned int n_bg_dofs = fe_bg.n_dofs_per_cell();
  std::vector<types::global_dof_index> bg_ldi(n_bg_dofs);

  IndexSet relevant_imm_global(dof_handler_fg.n_dofs());
  // Build a DSP for Ct with row partition = owned_bg.
  // Columns can be any global index in [0, n_imm). We use a complete
  // index set as the "writable" rows are determined by ownership.
  DynamicSparsityPattern dsp_C(dof_handler_bg.n_dofs(), dof_handler_fg.n_dofs(),
                               relevant_bg);

  {
    auto particle = particles.begin();
    while (particle != particles.end()) {
      const auto &cell = particle->get_surrounding_cell();
      typename DoFHandler<dim>::cell_iterator dh_cell(*cell, &dof_handler_bg);
      dh_cell->get_dof_indices(bg_ldi);
      const auto pic = particles.particles_in_cell(cell);
      // Collect immersed dof indices appearing in this background cell.
      std::vector<types::global_dof_index> imm_cols;
      imm_cols.reserve(n_imm_dofs);
      for (const auto &p : pic) {
        const auto &pp = p.get_properties();
        for (unsigned int i = 0; i < n_imm_dofs; ++i)
          imm_cols.push_back(
              static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + i]));
      }
      std::sort(imm_cols.begin(), imm_cols.end());
      imm_cols.erase(std::unique(imm_cols.begin(), imm_cols.end()),
                     imm_cols.end());
      for (const auto bg_i : bg_ldi)
        for (const auto imm_j : imm_cols)
          dsp_C.add(bg_i, imm_j);
      particle = pic.end();
    }
  }
  SparsityTools::distribute_sparsity_pattern(dsp_C, owned_bg, mpi_communicator,
                                             relevant_bg);
  coupling_matrix.reinit(owned_bg, owned_fg, dsp_C, mpi_communicator);

  // Second pass: assemble Ct and add AL augmentation to A_bg.
  FullMatrix<double> local_C(n_bg_dofs, n_imm_dofs);
  FullMatrix<double> local_AL(n_bg_dofs, n_bg_dofs);
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

        // Augmentation contribution: gamma_1 * (phi_i, phi_j)_immersed
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          for (unsigned int j = 0; j < n_bg_dofs; ++j)
            local_AL(i, j) += gamma_1_scaled * bg_vals[i] * bg_vals[j] * JxW;

        // Coupling contribution: (phi_i_bg, psi_j_imm)_immersed
        local_C = 0;
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          for (unsigned int j = 0; j < n_imm_dofs; ++j)
            local_C(i, j) = bg_vals[i] * pp[1 + j] * JxW;

        // Distribute coupling matrix contributions per-particle.
        std::vector<types::global_dof_index> imm_cols(n_imm_dofs);
        for (unsigned int j = 0; j < n_imm_dofs; ++j)
          imm_cols[j] =
              static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + j]);
        constraints_bg.distribute_local_to_global(
            local_C, bg_ldi, constraints_fg, imm_cols, coupling_matrix);
      }
      // Distribute the AL augmentation (per background cell)
      constraints_bg.distribute_local_to_global(local_AL, bg_ldi, A_bg);
      particle = pic.end();
    }
  }
  A_bg.compress(VectorOperation::add);
  coupling_matrix.compress(VectorOperation::add);
}

template <int dim> unsigned int EllipticInterfaceDLMParallel<dim>::solve() {
  // h-scaling absorbed into gamma's (operator form W = h^2 M).
  const double h_immersed = GridTools::maximal_cell_diameter(*tria_fg);
  const double inv_h2 = 1. / (h_immersed * h_immersed);
  const double gamma_1 = parameters.gamma_AL_background * inv_h2;
  const double gamma_2 = parameters.gamma_AL_immersed * inv_h2;

  // Assemble all matrices for this cycle.
  assemble_bg();
  assemble_fg(gamma_2);
  assemble_mass_fg();

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
                   dof_handler_fg.n_dofs());
  }

  assemble_coupling_and_augment_bg(gamma_1);

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
                   dof_handler_fg.n_dofs());
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

  // Build invW = M^{-1} action via CG with a Jacobi preconditioner on the
  // immersed mass matrix. This replaces the previous diagonal lump.
  LA::MPI::PreconditionJacobi mass_prec;
  mass_prec.initialize(mass_matrix_fg);
  using InvW_t = ALParallel::InvMassOp<LA::MPI::Vector, LA::MPI::SparseMatrix,
                                       LA::MPI::PreconditionJacobi>;
  InvW_t invW_op(mass_matrix_fg, mass_prec, /*max_it*/ 1000, /*tol*/ 1e-12,
                 owned_fg, mpi_communicator);

  // LinearOperators only for single matrices (no composition).
  auto A11_aug_op = linear_operator<LA::MPI::Vector>(A_bg);
  auto A22_aug_op = linear_operator<LA::MPI::Vector>(A_fg_plus_scaled_M);

  ALParallel::System3x3<LA::MPI::Vector, LA::MPI::BlockVector,
                        LA::MPI::SparseMatrix, InvW_t>
      system_operator{
          A_bg,    A_fg,    mass_matrix_fg, coupling_matrix, invW_op,
          gamma_1, gamma_2, owned_bg,       owned_fg,        mpi_communicator};

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
    // std::vector<std::vector<bool>> constant_modes_fg;
    // DoFTools::extract_constant_modes(dof_handler_fg,
    //                                  ComponentMask(),
    //                                  constant_modes_fg);
    LA::MPI::PreconditionAMG::AdditionalData amg_data_a22;
    // amg_data_a22.constant_modes = constant_modes_fg;
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

    std::unique_ptr<SolverCG<LA::MPI::Vector>> inner_cg;
    if (parameters.use_fixed_iterations)
      inner_cg = std::make_unique<SolverCG<LA::MPI::Vector>>(
          parameters.iteration_number_control);
    else
      inner_cg = std::make_unique<SolverCG<LA::MPI::Vector>>(
          parameters.inner_solver_control);

    auto A11_aug_inv = inverse_operator(A11_aug_op, *inner_cg, amg_A11);
    auto A22_aug_inv = inverse_operator(A22_aug_op, *inner_cg, amg_A22);

    ALParallel::ModifiedALPreconditioner<
        LA::MPI::Vector, LA::MPI::BlockVector, LA::MPI::SparseMatrix, InvW_t,
        decltype(A11_aug_inv), decltype(A22_aug_inv)>
        prec_AL{mass_matrix_fg, coupling_matrix, invW_op,
                gamma_1,        A11_aug_inv,     A22_aug_inv,
                owned_bg,       owned_fg,        mpi_communicator};

    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, prec_AL);
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
        Aug_mat{A_bg,           A_fg,
                mass_matrix_fg, coupling_matrix,
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
        ideal_prec{mass_matrix_fg, coupling_matrix, invW_op,  gamma_1,
                   Aug_inv,        owned_bg,        owned_fg, mpi_communicator};

    solver_fgmres.solve(system_operator, system_solution_block,
                        system_rhs_block, ideal_prec);
  }

  constraints_bg.distribute(system_solution_block.block(0));
  constraints_fg.distribute(system_solution_block.block(1));
  constraints_fg.distribute(system_solution_block.block(2));

  const unsigned int n_outer = parameters.outer_solver_control.last_step();
  pcout << "Solved in " << n_outer << " outer FGMRES iterations." << std::endl;

  convergence_table.add_value("cells", tria_bg->n_global_active_cells());
  convergence_table.add_value("DoF bg", dof_handler_bg.n_dofs());
  convergence_table.add_value("DoF imm", dof_handler_fg.n_dofs());
  convergence_table.add_value("gamma1", gamma_1);
  if (parameters.use_modified_AL_preconditioner)
    convergence_table.add_value("gamma2", gamma_2);
  convergence_table.add_value("outer it", n_outer);
  return n_outer;
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
  LA::MPI::Vector lam_ghosted(owned_fg, relevant_fg, mpi_communicator);
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
    data_out.add_data_vector(lam_ghosted, "lambda");
    data_out.build_patches(*mapping_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-imm", cycle, mpi_communicator,
                                        2, 0);
  }

  if (this_mpi_process == 0) {
    convergence_table.write_text(
        std::cout, TableHandler::TextOutputFormat::org_mode_table);
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
