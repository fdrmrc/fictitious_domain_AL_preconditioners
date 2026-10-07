// Parallel (MPI) version of the elasticity fictitious-domain interface problem
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

#include <deal.II/fe/fe_q.h>
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
// assembled at the variational level (mirroring elliptic_interface_op.cc):
//
//   (0,0) = A_bg                            (AL term already added in place)
//   (0,1) = -gamma1 * Ct                   (u|_Omega_2, v_2)
//   (0,2) =  Ct
//   (1,0) = -gamma2 * Ct^T                 transpose of (0,1)
//   (1,1) =  A_fg + gamma2 * M             (already assembled in
//                                           A_fg_plus_scaled_M)
//   (1,2) = -M
//   (2,0) =  Ct^T
//   (2,1) = -M
//   (2,2) =  0
//
// where invW = M^{-1} (applied via CG) is used only by the AL
// preconditioners below. A_bg and A_fg are elasticity stiffness matrices,
// M is the vector L^2 mass matrix on the immersed domain, and Ct is the
// vector L^2 coupling matrix (n_bg x n_imm).

template <typename VectorType, typename BlockVectorType,
          typename SparseMatrixType, typename InvWOp>
struct System3x3 {
  const SparseMatrixType &A_bg;
  const SparseMatrixType &A_fg_plus_scaled_M; // A_fg + gamma2 * M
  const SparseMatrixType &M;
  const SparseMatrixType &Ct;
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm);
    VectorType tb(owned_bg, comm);

    // Row 0: A_bg * x0  - gamma1 * Ct * x1  + Ct * x2
    A_bg.vmult(dst.block(0), src.block(0));
    Ct.vmult(tb, src.block(1));
    dst.block(0).add(-gamma1, tb);
    Ct.vmult(tb, src.block(2));
    dst.block(0) += tb;

    // Row 1: -gamma2 * Ct^T * x0  +  A_fg_plus_scaled_M * x1  -  M * x2
    Ct.Tvmult(ti, src.block(0));
    ti *= -gamma2;
    dst.block(1) = ti;
    A_fg_plus_scaled_M.vmult_add(dst.block(1), src.block(1));
    M.vmult(ti, src.block(2));
    dst.block(1) -= ti;

    // Row 2: Ct^T * x0  -  M * x1
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
  const SparseMatrixType &A_fg_plus_scaled_M;
  const SparseMatrixType &Ct;
  double gamma1, gamma2;
  IndexSet owned_bg, owned_fg;
  MPI_Comm comm;

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    VectorType ti(owned_fg, comm);
    VectorType tb(owned_bg, comm);

    A_bg.vmult(dst.block(0), src.block(0));
    Ct.vmult(tb, src.block(1));
    dst.block(0).add(-gamma1, tb);

    Ct.Tvmult(ti, src.block(0));
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

  // If true, the immersed mesh is built once (at cycle 0) and re-used
  // unchanged in subsequent refinement cycles, while the background mesh
  // keeps being refined. This enables a step-70-style fast path in which
  // the coupling particles are inserted only once and then *transferred*
  // through background `refine_global` via
  // `Particles::ParticleHandler::prepare_for_coarsening_and_refinement` /
  // `unpack_after_coarsening_and_refinement`, instead of being recomputed
  // and re-routed across MPI every cycle. Only meaningful when the
  // background uses `parallel::distributed::Triangulation` (hex), since
  // simplex backgrounds rebuild their p::f::T from scratch every cycle.
  bool keep_immersed_grid_fixed = false;

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
  }
  leave_subsection();

  enter_subsection("Refinement and remeshing");
  {
    add_parameter("Initial background refinement",
                  initial_background_refinement);
    add_parameter("Initial immersed refinement", initial_immersed_refinement);
    add_parameter("Refinemented cycles", n_refinement_cycles);
    add_parameter("Keep immersed grid fixed", keep_immersed_grid_fixed,
                  "If true, build the immersed mesh only at cycle 0 and "
                  "re-use it unchanged in subsequent cycles, while the "
                  "background mesh is refined. Enables step-70-style "
                  "transfer of coupling particles across background "
                  "refinements (only effective with a hex/p::d::T "
                  "background).");
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

// -----------------------------------------------------------------------------
// Parallel elasticity interface solver (operator-form AL)
// -----------------------------------------------------------------------------
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
  // `parallel::distributed::Triangulation` (p4est, optimal for hexes and
  // supports arbitrary `refine_global` / future adaptive refinement) while
  // simplex backgrounds keep using `parallel::fullydistributed::Triangulation`
  // (the only parallel option that supports tets in deal.II).
  std::unique_ptr<parallel::DistributedTriangulationBase<dim>> tria_bg;
  parallel::fullydistributed::Triangulation<dim> tria_fg;

  // Persistent serial triangulations used to build the fullydistributed
  // simplex meshes incrementally across refinement cycles. Building a
  // (simplex) `parallel::fullydistributed::Triangulation` requires a serial
  // mesh as input; without caching, each cycle would redo all previous
  // refinement levels from scratch (cost growing as 8^N for tets). With the
  // cache, cycle 0 generates the base mesh and applies the initial
  // refinement, then every subsequent cycle simply calls `refine_global(1)`
  // on the cached serial mesh. The corresponding `p::f::T` is still rebuilt
  // from the (now larger) serial mesh every cycle since `p::f::T` does not
  // support in-place simplex refinement.
  std::unique_ptr<Triangulation<dim>> serial_tria_bg_cache;
  std::unique_ptr<Triangulation<dim>> serial_tria_fg_cache;

  // Vector-valued FEs (FESystem of FE_Q or FE_SimplexP, with `dim` copies).
  std::unique_ptr<FiniteElement<dim>> fe_bg_ptr;
  std::unique_ptr<FiniteElement<dim>> fe_fg_ptr;
  std::unique_ptr<Mapping<dim>> mapping_bg_ptr;
  std::unique_ptr<Mapping<dim>> mapping_fg_ptr;
  std::function<Quadrature<dim>(unsigned int)> make_quadrature_bg;
  std::function<Quadrature<dim>(unsigned int)> make_quadrature_fg;

  DoFHandler<dim> dof_handler_bg;
  DoFHandler<dim> dof_handler_fg;

  IndexSet owned_bg, relevant_bg;
  IndexSet owned_fg, relevant_fg;

  AffineConstraints<double> constraints_bg;
  AffineConstraints<double> constraints_fg;

  LA::MPI::SparseMatrix A_bg;
  LA::MPI::SparseMatrix A_fg;
  LA::MPI::SparseMatrix A_fg_plus_scaled_M;
  LA::MPI::SparseMatrix mass_matrix_fg;
  LA::MPI::SparseMatrix coupling_matrix;

  LA::MPI::BlockVector system_rhs_block;
  LA::MPI::BlockVector system_solution_block;

  // Persistent particle handler used by `assemble_coupling_and_augment_bg`.
  // When `parameters.keep_immersed_grid_fixed` is true and the background
  // mesh is a `parallel::distributed::Triangulation` (hex), the particles
  // are inserted once at cycle 0 and then transferred across the
  // subsequent background `refine_global(1)` calls inside `generate_grids`
  // using the step-70-style
  // `prepare_for_coarsening_and_refinement` / `unpack_after_...` pair.
  // Otherwise the handler is cleared and re-populated from scratch every
  // cycle (current behaviour).
  Particles::ParticleHandler<dim> particles;
  bool particles_built = false;

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
      tria_fg(mpi_communicator), dof_handler_fg(tria_fg) {
  if (parameters.mesh_type_background == "hex") {
    tria_bg = std::make_unique<parallel::distributed::Triangulation<dim>>(
        mpi_communicator);
  } else {
    tria_bg = std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
  }
  dof_handler_bg.reinit(*tria_bg);
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
  // from scratch. This is the dominant speedup for tet-heavy immersed meshes
  // at high refinement, where the cost of `refine_global(N)` grows as 8^N.
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
      // Step-70-style particle transfer: if particles have already been
      // inserted (cycle > 0 with `Keep immersed grid fixed = true`), pack
      // them into the p4est CellDataTransfer buffer before refinement and
      // unpack them afterwards. This avoids the expensive geometric
      // re-insertion (and MPI all-to-all routing) of every coupling
      // quadrature point at every cycle.
      // Free the large cycle-(N-1) sparse matrices before triggering the
      // p4est forest rebuild. During refine_global(1) p4est must hold both
      // the old and the new (8x) mesh simultaneously; if the previous-cycle
      // matrices (A_bg dominates at ~81 nnz/row × n_dofs × 8 B) are still
      // live the combined peak can exceed the per-node RAM budget and cause
      // an OOM before system_setup() ever gets to reinit() them. Clearing
      // here is safe because system_setup() unconditionally reinitializes
      // every matrix from scratch right after generate_grids() returns.
      A_bg.clear();
      A_fg.clear();
      A_fg_plus_scaled_M.clear();
      mass_matrix_fg.clear();
      coupling_matrix.clear();

      const bool transfer_particles =
          particles_built && parameters.keep_immersed_grid_fixed;
      if (transfer_particles)
        particles.prepare_for_coarsening_and_refinement();
      tria_d->refine_global(1);
      if (transfer_particles)
        particles.unpack_after_coarsening_and_refinement();
    }
  } else {
    // Background is simplex -> fall back to fullydistributed. The serial
    // triangulation is cached so that on every cycle we only refine ONE
    // extra level rather than rebuilding from scratch.
    // Note: any previously inserted coupling particles are invalidated by
    // the p::f::T rebuild below, so the step-70-style transfer fast path
    // is not available on this branch -- force a fresh re-insertion.
    // Free the previous-cycle matrices for the same reason as the hex path.
    A_bg.clear();
    A_fg.clear();
    A_fg_plus_scaled_M.clear();
    mass_matrix_fg.clear();
    coupling_matrix.clear();
    particles_built = false;
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
  // Immersed mesh: skip entirely on cycles > 0 when the user has requested
  // to keep it fixed. The cached serial mesh and the live p::f::T from
  // cycle 0 stay untouched, which is what the step-70-style particle
  // transfer above relies on (particle properties carry immersed shape
  // values, JxW and immersed dof indices and remain valid only as long as
  // the immersed FE space is unchanged).
  const bool skip_immersed_rebuild =
      parameters.keep_immersed_grid_fixed && serial_tria_fg_cache != nullptr;
  if (!skip_immersed_rebuild)
    build_one_incremental(
        tria_fg, serial_tria_fg_cache, parameters.name_of_immersed_grid,
        parameters.arguments_for_immersed_grid,
        parameters.initial_immersed_refinement,
        parameters.name_external_immersed_grid,
        parameters.immersed_scale_factor, parameters.mesh_type_immersed);

  // Post-processing: refine background cells whose bounding box intersects
  // the surface of the immersed body. Performed only once, after the initial
  // background mesh was just built; on subsequent cycles the existing
  // refinement pattern is preserved by `refine_global(1)`.
  if (bg_was_freshly_built)
    refine_background_at_interface(parameters.interface_refinement_levels);

  const double h_bg = GridTools::maximal_cell_diameter(*tria_bg);
  const double h_fg = GridTools::maximal_cell_diameter(tria_fg);
  pcout << "h background = " << h_bg << "\n"
        << "h immersed   = " << h_fg << "\n"
        << "ratio (bg/imm) = " << h_bg / h_fg << std::endl;
}

// Post-processing local refinement of the (hex) background near the surface
// of the immersed body. For each of `n_levels` passes, every locally-owned
// background cell whose bounding box intersects the bounding box of any
// boundary face of the immersed mesh is flagged for refinement, then the
// mesh is refined. Queries are accelerated by an R-tree built from the
// immersed boundary-face bounding boxes (gathered across MPI ranks).
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
  for (const auto &cell : tria_fg.active_cell_iterators())
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

  owned_bg = dof_handler_bg.locally_owned_dofs();
  relevant_bg = DoFTools::extract_locally_relevant_dofs(dof_handler_bg);

  owned_fg = dof_handler_fg.locally_owned_dofs();
  relevant_fg = DoFTools::extract_locally_relevant_dofs(dof_handler_fg);

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
    top_value = 0.;
    top_value[dim - 1] = -parameters.compression_delta;
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

  std::vector<IndexSet> owned_blocks = {owned_bg, owned_fg, owned_fg};
  std::vector<IndexSet> relevant_blocks = {relevant_bg, relevant_fg,
                                           relevant_fg};
  system_rhs_block.reinit(owned_blocks, mpi_communicator);
  system_solution_block.reinit(owned_blocks, mpi_communicator);

  pcout << "N DoF background: " << dof_handler_bg.n_dofs() << "\n"
        << "N DoF immersed:   " << dof_handler_fg.n_dofs() << std::endl;
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
// the augmented preconditioner block A_fg_plus_scaled_M = A_fg + gamma_2 * M.
template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_fg(double gamma_2_scaled) {
  TimerOutput::Scope t(computing_timer, "Assemble fg (A_omega2 and A22-prec)");
  const auto &fe_fg = *fe_fg_ptr;
  const Quadrature<dim> quad = make_quadrature_fg(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_fg_ptr, fe_fg, quad,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  FullMatrix<double> cm_A(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cm_Aug(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);

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
      cm_A = 0;
      cm_Aug = 0;
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
            cm_A(i, j) += a;
            cm_Aug(i, j) += a + gamma_2_scaled * m;
          }
          double rhs_dot_phi = 0.;
          for (unsigned int d = 0; d < dim; ++d)
            rhs_dot_phi += rhs_values[q][d] * phi_i[d];
          cell_rhs(i) += rhs_dot_phi * fe_values.JxW(q);
        }
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

template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_mass_fg() {
  TimerOutput::Scope t(computing_timer, "Assemble mass matrix (immersed)");
  const auto &fe_fg = *fe_fg_ptr;
  const Quadrature<dim> quad = make_quadrature_fg(fe_fg.degree + 1);
  FEValues<dim> fe_values(*mapping_fg_ptr, fe_fg, quad,
                          update_values | update_JxW_values);
  const unsigned int dofs_per_cell = fe_fg.n_dofs_per_cell();
  FullMatrix<double> cm(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> ldi(dofs_per_cell);
  const FEValuesExtractors::Vector displacements(0);
  for (const auto &cell : dof_handler_fg.active_cell_iterators())
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cm = 0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<1, dim> phi_i = fe_values[displacements].value(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const Tensor<1, dim> phi_j = fe_values[displacements].value(j, q);
            cm(i, j) += (phi_i * phi_j) * fe_values.JxW(q);
          }
        }
      cell->get_dof_indices(ldi);
      constraints_fg.distribute_local_to_global(cm, ldi, mass_matrix_fg);
    }
  mass_matrix_fg.compress(VectorOperation::add);
}

// Particle-based assembly of Ct (vector coupling) and AL augmentation of A_bg.
// For primitive FESystem-of-FE_Q (or FE_SimplexP) finite elements, every shape
// function has a single non-zero component. Hence the dot products
// (phi_i_bg . phi_j_imm) and (phi_i_bg . phi_j_bg) reduce to scalar shape
// values multiplied together, conditional on the components matching.
//
// Particle properties layout (n_imm_dofs = fe_fg.n_dofs_per_cell()):
//   [0]                                        : JxW
//   [1                  .. n_imm_dofs]         : immersed scalar shape values
//   [1+n_imm_dofs       .. 2*n_imm_dofs]       : immersed global dof indices
//   [1+2*n_imm_dofs     .. 3*n_imm_dofs]       : immersed dof components
//   (0..dim-1)
template <int dim>
void ElasticityInterfaceDLMParallel<dim>::assemble_coupling_and_augment_bg(
    double gamma_1_scaled) {
  TimerOutput::Scope t(computing_timer,
                       "Coupling assembly and bg augmentation (particles)");
  const auto &fe_bg = *fe_bg_ptr;
  const auto &fe_fg = *fe_fg_ptr;

  const unsigned int n_imm_dofs = fe_fg.n_dofs_per_cell();
  const unsigned int n_bg_dofs = fe_bg.n_dofs_per_cell();
  const unsigned int n_props = 1 + 3 * n_imm_dofs;

  // Fast path: the immersed mesh is unchanged since the last call and the
  // background only went through `refine_global` wrapped in particle
  // prepare/unpack (see `generate_grids`). In that case the persistent
  // member `particles` already contains the correct quadrature particles
  // sorted into the (refined) background cells, with valid properties
  // (immersed shape values, JxW, dof indices, components). We only need
  // to rebuild the coupling sparsity pattern and re-assemble the matrices.
  const bool reuse_particles =
      particles_built && parameters.keep_immersed_grid_fixed;

  if (!reuse_particles) {
    particles.clear_particles();

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
            p[1 + i] =
                fe_v_imm.shape_value(i, q); // scalar value of nonzero component
          for (unsigned int i = 0; i < n_imm_dofs; ++i)
            p[1 + n_imm_dofs + i] = static_cast<double>(ldi[i]);
          for (unsigned int i = 0; i < n_imm_dofs; ++i)
            p[1 + 2 * n_imm_dofs + i] =
                static_cast<double>(fe_fg.system_to_component_index(i).first);
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

    particles_built = true;
  } else {
    pcout << "Reusing " << particles.n_global_particles()
          << " coupling particles transferred across bg refinement."
          << std::endl;
  }

  pcout << "Inserted " << particles.n_global_particles()
        << " coupling particles." << std::endl;

  // Cache background dof component indices.
  std::vector<unsigned int> bg_comp(n_bg_dofs);
  for (unsigned int i = 0; i < n_bg_dofs; ++i)
    bg_comp[i] = fe_bg.system_to_component_index(i).first;

  std::vector<types::global_dof_index> bg_ldi(n_bg_dofs);

  // First pass: build sparsity pattern for Ct.
  DynamicSparsityPattern dsp_C(dof_handler_bg.n_dofs(), dof_handler_fg.n_dofs(),
                               relevant_bg);
  {
    auto particle = particles.begin();
    while (particle != particles.end()) {
      const auto &cell = particle->get_surrounding_cell();
      typename DoFHandler<dim>::cell_iterator dh_cell(*cell, &dof_handler_bg);
      dh_cell->get_dof_indices(bg_ldi);
      const auto pic = particles.particles_in_cell(cell);
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
      // Couple only component-matching pairs, but we keep the full block here
      // for sparsity (cheap and correct). We go through the bg and fg
      // constraint objects so that hanging-node entries (introduced by local
      // background refinement near the immersed surface) are included in the
      // sparsity pattern of the coupling matrix.
      constraints_bg.add_entries_local_to_global(bg_ldi, constraints_fg,
                                                 imm_cols, dsp_C);
      particle = pic.end();
    }
  }
  SparsityTools::distribute_sparsity_pattern(dsp_C, owned_bg, mpi_communicator,
                                             relevant_bg);
  coupling_matrix.reinit(owned_bg, owned_fg, dsp_C, mpi_communicator);

  // Second pass: assemble Ct + AL augmentation.
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

        std::vector<double> bg_vals(n_bg_dofs);
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          bg_vals[i] = fe_bg.shape_value(i, ref); // single nonzero component

        // AL: gamma_1 * (phi_i, phi_j)_immersed (component-wise)
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          for (unsigned int j = 0; j < n_bg_dofs; ++j)
            if (bg_comp[i] == bg_comp[j])
              local_AL(i, j) += gamma_1_scaled * bg_vals[i] * bg_vals[j] * JxW;

        // Coupling: (phi_i_bg, psi_j_imm)_immersed (component-wise)
        local_C = 0;
        std::vector<types::global_dof_index> imm_cols(n_imm_dofs);
        for (unsigned int j = 0; j < n_imm_dofs; ++j)
          imm_cols[j] =
              static_cast<types::global_dof_index>(pp[1 + n_imm_dofs + j]);
        for (unsigned int i = 0; i < n_bg_dofs; ++i)
          for (unsigned int j = 0; j < n_imm_dofs; ++j) {
            const unsigned int c_imm =
                static_cast<unsigned int>(pp[1 + 2 * n_imm_dofs + j]);
            if (bg_comp[i] == c_imm)
              local_C(i, j) = bg_vals[i] * pp[1 + j] * JxW;
          }
        constraints_bg.distribute_local_to_global(
            local_C, bg_ldi, constraints_fg, imm_cols, coupling_matrix);
      }
      constraints_bg.distribute_local_to_global(local_AL, bg_ldi, A_bg);
      particle = pic.end();
    }
  }
  A_bg.compress(VectorOperation::add);
  coupling_matrix.compress(VectorOperation::add);
}

template <int dim> unsigned int ElasticityInterfaceDLMParallel<dim>::solve() {
  const double h_immersed = GridTools::maximal_cell_diameter(tria_fg);
  const double inv_h2 = 1. / (h_immersed * h_immersed);
  const double gamma_1 = parameters.gamma_AL_background * inv_h2;
  const double gamma_2 = parameters.gamma_AL_immersed * inv_h2;

  assemble_bg();
  assemble_fg(gamma_2);
  assemble_mass_fg();
  assemble_coupling_and_augment_bg(gamma_1);

  LA::MPI::PreconditionJacobi mass_prec;
  mass_prec.initialize(mass_matrix_fg);
  using InvW_t = ALParallel::InvMassOp<LA::MPI::Vector, LA::MPI::SparseMatrix,
                                       LA::MPI::PreconditionJacobi>;
  InvW_t invW_op(mass_matrix_fg, mass_prec, 1000, 1e-12, owned_fg,
                 mpi_communicator);

  auto A11_aug_op = linear_operator<LA::MPI::Vector>(A_bg);
  auto A22_aug_op = linear_operator<LA::MPI::Vector>(A_fg_plus_scaled_M);

  ALParallel::System3x3<LA::MPI::Vector, LA::MPI::BlockVector,
                        LA::MPI::SparseMatrix, InvW_t>
      system_operator{
          A_bg,    A_fg_plus_scaled_M, mass_matrix_fg, coupling_matrix, gamma_1,
          gamma_2, owned_bg,           owned_fg,       mpi_communicator};

  // For elasticity AMG, hint the near-null-space using the full set of
  // *rigid body modes* (dim translations + dim*(dim-1)/2 rotations).
  // For the background block A11 the Dirichlet boundary conditions remove
  // the rigid-body modes from the kernel, but they still represent the
  // near-null space of the operator and improve AMG quality. For the
  // immersed block A22 they are essential because no BCs are imposed
  // there and A_fg + gamma_2 * M only weakly penalizes rigid motions.
  // We pass them through `constant_modes_values` (the double-valued
  // counterpart of `constant_modes`).
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
    amg_A22.initialize(A_fg_plus_scaled_M, data);
  }

  typename SolverFGMRES<LA::MPI::BlockVector>::AdditionalData data_fgmres;
  data_fgmres.max_basis_size = 30;
  SolverFGMRES<LA::MPI::BlockVector> solver_fgmres(
      parameters.outer_solver_control, data_fgmres);

  system_rhs_block.block(2) = 0.;
  system_solution_block = 0.;

  if (parameters.use_modified_AL_preconditioner) {
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

    {
      TimerOutput::Scope t(computing_timer, "Solve system");
      solver_fgmres.solve(system_operator, system_solution_block,
                          system_rhs_block, prec_AL);
    }
  } else {
    AssertThrow(std::abs(parameters.gamma_AL_background -
                         parameters.gamma_AL_immersed) < 1e-12,
                ExcMessage("Ideal AL requires gamma_1 == gamma_2."));

    pcout << "\t *** USING IDEAL AL PRECONDITIONER (test only) ***"
          << std::endl;

    ALParallel::Block2x2<LA::MPI::Vector, LA::MPI::BlockVector,
                         LA::MPI::SparseMatrix, InvW_t>
        Aug_mat{A_bg,    A_fg_plus_scaled_M, coupling_matrix, gamma_1,
                gamma_2, owned_bg,           owned_fg,        mpi_communicator};

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
void ElasticityInterfaceDLMParallel<dim>::output_results(
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
    data_out.add_data_vector(lam_ghosted, lam_names,
                             DataOut<dim>::type_dof_data, vec_interp);
    data_out.build_patches(*mapping_fg_ptr);
    data_out.write_vtu_with_pvtu_record(parameters.output_directory + "/",
                                        "solution-imm", cycle, mpi_communicator,
                                        2, 0);
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
