// Imposition of Dirichlet boundary conditions through a boundary-supported
// Lagrange multiplier (saddle-point formulation a' la Babuska), for the
// operator -Delta u + u as in Babuska's paper:
//
//   ( grad u, grad v )_Omega + ( u, v )_Omega
//                     + < lambda, v >_{partial Omega} = ( f, v )_Omega
//   < u, mu >_{partial Omega}                         = < g, mu
//   >_{partial Omega}
//
// The multiplier lives on the boundary mesh extracted from the bulk mesh, so
// the coupling matrix C with entries (mu_j, v_i)_{partial Omega} can be
// assembled by looping over the boundary cells and pairing them with the
// corresponding bulk faces via the surface-to-volume map returned by
// GridGenerator::extract_boundary_mesh.

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/particle_handler.h>

#include <deal.II/numerics/rtree.h>

#include <fstream>
#include <iostream>

#include "augmented_lagrangian_preconditioner.h"
#include "utilities.h"

namespace NitscheBCs {
using namespace dealii;

// True if the parameter selects tensor-product (quad/hex) cells.
inline bool is_hex_mesh(const std::string &mesh_type) {
  return mesh_type == "hex";
}

// Build a quadrature appropriate for the active mesh type.
// QGauss is implemented for any positive number of points; QGaussSimplex is
// only implemented for n_points_1D in {1,2,3,4} so let's cap the quad rule.
template <int d>
inline Quadrature<d> make_quadrature(const std::string &mesh_type,
                                     const unsigned int n) {
  if (is_hex_mesh(mesh_type))
    return QGauss<d>(std::max(1u, n));
  return QGaussSimplex<d>(std::min(4u, std::max(1u, n)));
}

// Manufactured solution used for the optional convergence study. The 0.3
// argument shift makes the boundary trace nonzero and non-constant on both
// the unit square and the disk, so the weak imposition of genuinely
// inhomogeneous data is exercised.
template <int spacedim> class ManufacturedSolution : public Function<spacedim> {
public:
  double value(const Point<spacedim> &p,
               const unsigned int /*component*/ = 0) const override {
    return std::sin(numbers::PI * (p[0] + 0.3)) *
           std::sin(numbers::PI * (p[1] + 0.3));
  }

  Tensor<1, spacedim>
  gradient(const Point<spacedim> &p,
           const unsigned int /*component*/ = 0) const override {
    Tensor<1, spacedim> g;
    g[0] = numbers::PI * std::cos(numbers::PI * (p[0] + 0.3)) *
           std::sin(numbers::PI * (p[1] + 0.3));
    g[1] = numbers::PI * std::sin(numbers::PI * (p[0] + 0.3)) *
           std::cos(numbers::PI * (p[1] + 0.3));
    return g;
  }
};

template <int spacedim> class ManufacturedRhs : public Function<spacedim> {
public:
  double value(const Point<spacedim> &p,
               const unsigned int /*component*/ = 0) const override {
    // -Delta u + u with u = sin(pi(x+0.3)) sin(pi(y+0.3)).
    return (2. * numbers::PI * numbers::PI + 1.) *
           std::sin(numbers::PI * (p[0] + 0.3)) *
           std::sin(numbers::PI * (p[1] + 0.3));
  }
};

template <int spacedim>
class ManufacturedDirichlet : public Function<spacedim> {
public:
  double value(const Point<spacedim> & /*p*/,
               const unsigned int /*component*/ = 0) const override {
    return 0.0; // u_ex vanishes on partial Omega = boundary of unit square.
  }
};

// Affine manufactured solution u = x + y: since -Delta u + u = f this gives
// f = x + y and Dirichlet data g = x + y on any domain. u is exactly
// representable in Q1/P1, so the measured error isolates the weak-BC /
// geometry consistency error of the multiplier formulation.
template <int spacedim> class AffineXY : public Function<spacedim> {
public:
  double value(const Point<spacedim> &p,
               const unsigned int /*component*/ = 0) const override {
    return p[0] + p[1];
  }

  Tensor<1, spacedim>
  gradient(const Point<spacedim> & /*p*/,
           const unsigned int /*component*/ = 0) const override {
    Tensor<1, spacedim> g;
    g[0] = 1.;
    g[1] = 1.;
    return g;
  }
};

template <int dim, int spacedim = dim> class BabuskaProblem {
public:
  class Parameters : public ParameterAcceptor {
  public:
    Parameters();

    unsigned int initial_refinement = 4;
    // Number of refinement levels by which the boundary (multiplier) mesh
    // is *coarser* than the bulk mesh. 0 (default) means matching meshes:
    // the multiplier mesh is the boundary trace of the bulk mesh. A value k > 0
    // builds a separate bulk mesh refined (n_bulk - k) times and extracts its
    // boundary as the multiplier mesh, which is non-matching and 2^k times
    // coarser. In that case the coupling matrix C is assembled via tracking
    // particles.
    unsigned int boundary_coarsening_level = 0;
    unsigned int n_refinement_cycles = 1;
    std::string name_of_grid = "hyper_cube";
    std::string arguments_for_grid = "0.0: 1.0: true";
    unsigned int bulk_space_finite_element_degree = 1;
    unsigned int multiplier_finite_element_degree = 1;
    unsigned int coupling_quadrature_order = 3;
    unsigned int verbosity_level = 4;
    bool use_discontinuous_multiplier = false;
    std::string name_external_grid = "homer.msh";
    double scale_factor = 1.0;
    // Cell type for both the bulk and boundary meshes: "hex" or "simplex".
    std::string mesh_type = "simplex";

    // The AL augmentation term gamma*(1/h)*(u,v)_{Gamma} is assembled
    // *consistently* with the coupling matrix C: when the multiplier
    // mesh matches the bulk boundary (boundary_coarsening_level == 0)
    // it is integrated via a boundary-face loop on the bulk grid; when
    // the multiplier mesh is coarser (boundary_coarsening_level > 0) it
    // is integrated via tracking particles placed at the same boundary
    // quadrature points used to build C. Mixing the two choices would
    // make K and C live on different surfaces, so it is no longer a
    // user knob.

    // If true, ignore the user-provided right-hand side and Dirichlet data
    // and use the hardcoded manufactured solution  u(x,y) = sin(pi x) sin(pi y)
    // on (0,1)^2,
    bool use_manufactured_solution = false;

    // Which manufactured solution to use (when the flag above is true):
    //  - "sine":     u = sin(pi(x+0.3)) sin(pi(y+0.3))  (nonzero,
    //    non-constant boundary trace);
    //  - "affine":   u = x + y, f = x + y, g = x + y  (any domain; u is in
    //    the FE space, so the error isolates the weak-BC consistency error).
    std::string manufactured_solution_type = "sine";

    bool iterative_inversion_mass_matrix = false;

    // Augmentation parameter gamma multiplying the (gamma/h) (u,v)_{Gamma}
    // term. Two independent values:
    //  - augmentation_gamma_assembly: factor actually added to the (0,0)
    //    block of the system matrix. Setting it to 0 recovers the raw
    //    un-augmented mixed problem and exposes classical inf-sup
    //    instabilities of the chosen bulk/multiplier pair.
    //  - augmentation_gamma_prec: factor used inside the AL block
    //    preconditioner only. Must stay > 0 so the Schur block of the
    //    preconditioner is well-defined.
    double augmentation_gamma_assembly = 10.0;
    double augmentation_gamma_prec = 10.0;

    // Power alpha in the augmentation scaling gamma * h^{-alpha} (h =
    // boundary-cell diameter). alpha = 1 (default) is the classical gamma/h
    // scaling; e.g. alpha = 0.5 gives the weaker gamma/sqrt(h). Used
    // consistently in the assembled augmentation and in the Schur block of
    // the AL preconditioner.
    double augmentation_h_power = 1.0;

    // If true, dump the three sparse matrices that define the saddle
    // point system (K = stiffness, C = coupling, M = boundary mass) in
    // MATLAB triplet format so they can be analysed offline (e.g. for
    // the discrete inf-sup eigenproblem (C^T K^{-1} C) v = beta^2 M v).
    // The export is intentionally restricted to small problems: it
    // throws when boundary_dh.n_dofs() exceeds matrix_export_max_dofs,
    // since larger cases would produce huge files and the dense
    // eigenproblem is not tractable anyway.
    bool export_matrices_for_matlab = false;
    unsigned int matrix_export_max_dofs = 5000;

    // If true (non-matching multiplier only), after the particle-based
    // augmentation of the (0,0) block also assemble it algebraically as
    // gamma*(1/h) * C diag(M)^{-1} C^T (lumped boundary mass) purely for a
    // timing / sparsity comparison. The algebraic result is discarded.
    bool compare_algebraic_augmentation = false;

    bool initialized = false;
  };

  BabuskaProblem(const Parameters &parameters);

  void run();

private:
  void setup_grids_and_dofs();
  void setup_coupling();
  void assemble_system();
  void solve();
  void output_results();

  const Parameters &parameters;

  Triangulation<spacedim> space_grid;
  std::unique_ptr<FiniteElement<spacedim>> space_fe;
  std::unique_ptr<Mapping<spacedim>> bulk_mapping;
  DoFHandler<spacedim> space_dh;

  // Boundary (multiplier) mesh and FE, extracted from space_grid.
  std::unique_ptr<Triangulation<dim, spacedim>> boundary_grid;
  std::unique_ptr<FiniteElement<dim, spacedim>> multiplier_fe;
  std::unique_ptr<Mapping<dim, spacedim>> surf_mapping;
  DoFHandler<dim, spacedim> boundary_dh;

  // Surface-cell -> bulk-face map filled by extract_boundary_mesh.
  std::map<typename Triangulation<dim, spacedim>::cell_iterator,
           typename Triangulation<spacedim, spacedim>::face_iterator>
      surface_to_volume_map;

  // Inverse lookup: bulk-face -> (bulk cell, local face number). Built once
  // after the bulk DoFs are distributed; reused by setup_coupling and
  // assemble_system
  std::map<typename Triangulation<spacedim, spacedim>::face_iterator,
           std::pair<typename DoFHandler<spacedim>::active_cell_iterator,
                     unsigned int>>
      face_to_bulk_cell;

  // Used only when boundary_coarsening_level > 0: particles laid down at
  // boundary-quadrature points and tracked on the (fine) bulk grid, with
  // properties [JxW, surf_active_cell_index, local_q_index]. Empty for the
  // matching-mesh path.
  Particles::ParticleHandler<spacedim> immersed_particle_handler;

  // True if boundary_coarsening_level > 0. Set in setup_grids_and_dofs().
  bool non_matching_multiplier = false;

  // Right-hand side and Dirichlet data imposed on the boundary.
  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs_function;
  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> g_function;

  // Solver controls (readable from the .prm file).
  ParameterAcceptorProxy<ReductionControl> outer_solver_control;
  ParameterAcceptorProxy<ReductionControl> inner_solver_control;

  AffineConstraints<double> constraints;

  SparsityPattern stiffness_sparsity;
  SparsityPattern coupling_sparsity;
  SparsityPattern boundary_mass_sparsity;

  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> coupling_matrix;
  SparseMatrix<double> boundary_mass_matrix;

  Vector<double> solution;
  Vector<double> embedding_rhs;

  Vector<double> lambda;
  Vector<double> embedded_rhs;

  // Convergence table for the optional convergence study
  ConvergenceTable convergence_table;

  TimerOutput monitor;
};

template <int dim, int spacedim>
BabuskaProblem<dim, spacedim>::Parameters::Parameters()
    : ParameterAcceptor("/NitscheLagrange<" + Utilities::int_to_string(dim) +
                        "," + Utilities::int_to_string(spacedim) + ">/") {
  add_parameter("Initial space refinement", initial_refinement);
  add_parameter(
      "Boundary coarsening level", boundary_coarsening_level,
      "Number of refinement levels by which the multiplier (boundary) mesh "
      "is coarser than the bulk mesh. 0 (default) means a matching boundary "
      "mesh extracted from the bulk grid; k > 0 builds a separately-refined "
      "non-matching multiplier mesh, 2^k times coarser, and uses tracking "
      "particles to assemble the coupling matrix C.");
  add_parameter("Number of refinement cycles", n_refinement_cycles);
  add_parameter("Name of the grid", name_of_grid);
  add_parameter("Arguments for the grid", arguments_for_grid);
  add_parameter("Bulk space finite element degree",
                bulk_space_finite_element_degree);
  add_parameter("Use discontinuous multiplier space",
                use_discontinuous_multiplier);
  add_parameter("Multiplier finite element degree",
                multiplier_finite_element_degree);
  add_parameter("Name of the external grid file", name_external_grid);
  add_parameter("Scale factor for the external grid", scale_factor);
  add_parameter("Coupling quadrature order", coupling_quadrature_order);
  add_parameter("Verbosity level", verbosity_level);
  add_parameter("Mesh type", mesh_type,
                "Cell type used for both the bulk and boundary meshes. "
                "Either \"hex\" (tensor-product quads/hexes) or \"simplex\" "
                "(tris/tets).",
                ParameterAcceptor::prm, Patterns::Selection("hex|simplex"));
  add_parameter(
      "Use manufactured solution", use_manufactured_solution,
      "If true, override the right-hand side and Dirichlet data with the "
      "selected manufactured solution and report L2/H1 errors plus "
      "convergence rates.");
  add_parameter("Manufactured solution type", manufactured_solution_type,
                "Manufactured solution used when 'Use manufactured solution' "
                "is true: \"sine\" (u = sin(pi(x+0.3)) sin(pi(y+0.3)), "
                "nonzero non-constant boundary trace), or "
                "\"affine\" (u = x + y with f = x + y and g = x + y, valid "
                "on any domain).",
                ParameterAcceptor::prm, Patterns::Selection("sine|affine"));
  add_parameter(
      "Iterative inversion mass matrix", iterative_inversion_mass_matrix,
      "If true, invert the boundary mass matrix (used inside the Schur "
      "preconditioner) iteratively with CG + Jacobi, instead of factorizing "
      "it with UMFPACK.");
  add_parameter(
      "Augmentation gamma assembly", augmentation_gamma_assembly,
      "Coefficient of the (gamma/h)(u,v)_{Gamma} term added to the system "
      "(0,0) block. Set to 0 to solve the raw un-augmented saddle-point "
      "so that classical inf-sup instabilities of the bulk/multiplier "
      "pair become visible.");
  add_parameter(
      "Augmentation gamma preconditioner", augmentation_gamma_prec,
      "Coefficient of (gamma/h) M^{-1} used in the AL block preconditioner. "
      "Must be > 0 for the preconditioner to be well-defined; can be kept "
      "at its default value independently of the assembly gamma.");
  add_parameter(
      "Augmentation h power", augmentation_h_power,
      "Power alpha in the augmentation scaling gamma * h^{-alpha}, where h "
      "is the boundary-cell diameter. alpha = 1 (default) reproduces the "
      "classical gamma/h scaling; alpha = 0.5 gives gamma/sqrt(h). Applied "
      "consistently to the assembled term and to the Schur block of the AL "
      "preconditioner.");
  add_parameter(
      "Export matrices for matlab", export_matrices_for_matlab,
      "If true, write K (stiffness), C (coupling) and M (boundary mass) "
      "as MATLAB-readable triplet (i,j,val) text files for offline "
      "analysis (e.g. the discrete inf-sup eigenproblem). Throws "
      "when the multiplier space exceeds 'Matrix export max DoFs'. "
      "To recover the genuine pair inf-sup constant the system must be "
      "un-augmented: set 'Augmentation gamma assembly' to 0.");
  add_parameter(
      "Matrix export max DoFs", matrix_export_max_dofs,
      "Maximum number of multiplier DoFs allowed when matrix export is "
      "enabled; the run aborts on cycles that exceed this size.");
  add_parameter(
      "Compare algebraic augmentation", compare_algebraic_augmentation,
      "If true, after the particle-based augmentation of the (0,0) block, "
      "also assemble it algebraically as gamma*(1/h)*C diag(M)^{-1} C^T "
      "(lumped boundary mass) and report timing, sparsity and an "
      "informational action difference. Requires a non-matching multiplier "
      "mesh ('Boundary coarsening level' > 0); the algebraic result is "
      "discarded.");

  parse_parameters_call_back.connect([&]() -> void { initialized = true; });
}

template <int dim, int spacedim>
BabuskaProblem<dim, spacedim>::BabuskaProblem(const Parameters &parameters)
    : parameters(parameters), space_dh(space_grid),
      rhs_function("Right hand side"), g_function("Dirichlet boundary data"),
      outer_solver_control("Outer solver control"),
      inner_solver_control("Inner solver control"),
      monitor(std::cout, TimerOutput::summary, TimerOutput::wall_times) {
  rhs_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });

  g_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Tolerance", "1.e-8");
    ParameterAcceptor::prm.set("Reduction", "0");
    ParameterAcceptor::prm.set("Log history", "true");
    ParameterAcceptor::prm.set("Log result", "true");
  });

  inner_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Tolerance", "1.e-12");
    ParameterAcceptor::prm.set("Reduction", "1.e-2");
    ParameterAcceptor::prm.set("Log history", "false");
    ParameterAcceptor::prm.set("Log result", "true");
  });
}

template <int dim, int spacedim>
void BabuskaProblem<dim, spacedim>::setup_grids_and_dofs() {
  TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

  // Rebuild the bulk grid from scratch each cycle
  static unsigned int extra_refinements = 0;
  space_grid.clear();

  // Number of global refinement levels actually applied to space_grid in
  // this cycle. For built-in grids: initial_refinement + extra. For
  // from_file: only extra (initial_refinement is ignored). When the
  // multiplier mesh is k levels coarser (boundary_coarsening_level == k),
  // we also bump space_grid's refinement count by k, so that already in the
  // first cycle the coarse bulk used to extract the boundary has 0
  // refinements (i.e. the loaded/built grid itself).
  const unsigned int k_off = parameters.boundary_coarsening_level;
  unsigned int n_bulk_refinements = 0;
  if (parameters.name_of_grid == "from_file") {
    GridIn<spacedim> grid_in;
    grid_in.attach_triangulation(space_grid);
    std::ifstream input_file(parameters.name_external_grid);
    AssertThrow(input_file, ExcMessage("Could not open grid file"));
    grid_in.read_msh(input_file);
    n_bulk_refinements = extra_refinements + k_off;
    space_grid.refine_global(n_bulk_refinements);
    GridTools::scale(parameters.scale_factor, space_grid);
  } else {
    n_bulk_refinements =
        parameters.initial_refinement + extra_refinements + k_off;
    if (is_hex_mesh(parameters.mesh_type)) {
      GridGenerator::generate_from_name_and_arguments(
          space_grid, parameters.name_of_grid, parameters.arguments_for_grid);
      space_grid.refine_global(n_bulk_refinements);
    } else {
      // convert to simplex mesh so the rest of the program can run on
      // tris/tets.
      Triangulation<spacedim> hex_grid;
      GridGenerator::generate_from_name_and_arguments(
          hex_grid, parameters.name_of_grid, parameters.arguments_for_grid);
      hex_grid.refine_global(n_bulk_refinements);
      GridGenerator::convert_hypercube_to_simplex_mesh(hex_grid, space_grid);
    }
  }
  ++extra_refinements;

  stiffness_matrix.clear();
  coupling_matrix.clear();
  boundary_mass_matrix.clear();
  face_to_bulk_cell.clear();
  surface_to_volume_map.clear();

  non_matching_multiplier = (parameters.boundary_coarsening_level > 0);

  // Extract the boundary mesh. With newer deal.II versions,
  // extract_boundary_mesh supports also simplex meshes
  boundary_grid = std::make_unique<Triangulation<dim, spacedim>>();

  // Decide which bulk grid the boundary mesh is extracted from:
  //   - matching case (k == 0): the boundary is extracted from space_grid,
  //     and surface_to_volume_map gives a 1-1 link between surface cells
  //     and bulk faces of space_grid;
  //   - non-matching case (k > 0): we build a separate, independently-refined
  //     bulk mesh `coarse_bulk` (refined initial+extra-k times), extract its
  //     boundary as the multiplier mesh, and discard surface_to_volume_map.
  //     In that case coupling assembly goes through tracking particles
  //     (see setup_coupling()).
  Triangulation<spacedim> coarse_bulk;
  Triangulation<spacedim> *bulk_for_extraction = &space_grid;
  if (non_matching_multiplier) {
    AssertThrow(parameters.boundary_coarsening_level <= n_bulk_refinements,
                ExcMessage("Boundary coarsening level cannot exceed the "
                           "current number of bulk refinements. "
                           "Increase 'Initial space refinement' "
                           "(or, for from_file grids, the number of "
                           "refinement cycles already performed)."));
    const unsigned int n_coarse_refinements =
        n_bulk_refinements - parameters.boundary_coarsening_level;

    // Build coarse_bulk the same way space_grid was built, but with k fewer
    // refinements. Supports built-in grids ("from_file" loads the same .msh)
    // and both hex and simplex mesh types.
    if (parameters.name_of_grid == "from_file") {
      GridIn<spacedim> grid_in;
      grid_in.attach_triangulation(coarse_bulk);
      std::ifstream input_file(parameters.name_external_grid);
      AssertThrow(input_file, ExcMessage("Could not open grid file"));
      grid_in.read_msh(input_file);
      coarse_bulk.refine_global(n_coarse_refinements);
      GridTools::scale(parameters.scale_factor, coarse_bulk);
    } else if (is_hex_mesh(parameters.mesh_type)) {
      GridGenerator::generate_from_name_and_arguments(
          coarse_bulk, parameters.name_of_grid, parameters.arguments_for_grid);
      coarse_bulk.refine_global(n_coarse_refinements);
    } else {
      Triangulation<spacedim> hex_grid;
      GridGenerator::generate_from_name_and_arguments(
          hex_grid, parameters.name_of_grid, parameters.arguments_for_grid);
      hex_grid.refine_global(n_coarse_refinements);
      GridGenerator::convert_hypercube_to_simplex_mesh(hex_grid, coarse_bulk);
    }
    bulk_for_extraction = &coarse_bulk;
  }

  // extract_boundary_mesh re-creates the surface mesh at its coarsest level
  // and refines it internally, so any curved manifold must be attached to
  // the surface triangulation BEFORE the extraction (see also documentaiton).
  // Otherwise the surface refinement is flat and the multiplier mesh lands on
  // the chords of the coarsest boundary (e.g. a square inscribed in the disk)
  // instead of the actual boundary. Built-in ball/sphere/shell grids use
  // manifold id 0 with a SphericalManifold centered at the first grid argument.
  if (parameters.name_of_grid.find("ball") != std::string::npos ||
      parameters.name_of_grid.find("sphere") != std::string::npos ||
      parameters.name_of_grid.find("shell") != std::string::npos) {
    Point<spacedim> center; // parsed from "x, y[, z] : radius : ..."
    const auto pieces =
        Utilities::split_string_list(parameters.arguments_for_grid, ':');
    if (!pieces.empty()) {
      const auto coords = Utilities::split_string_list(pieces[0], ',');
      for (unsigned int d = 0; d < spacedim && d < coords.size(); ++d)
        center[d] = Utilities::string_to_double(coords[d]);
    }
    boundary_grid->set_manifold(0, SphericalManifold<dim, spacedim>(center));
  }

#if DEAL_II_VERSION_GTE(9, 9, 0)
  surface_to_volume_map = GridGenerator::extract_boundary_mesh(
      *bulk_for_extraction, *boundary_grid);
#else
  if (is_hex_mesh(parameters.mesh_type)) {
    surface_to_volume_map = GridGenerator::extract_boundary_mesh(
        *bulk_for_extraction, *boundary_grid);
  } else {
    using BulkFaceIt =
        typename Triangulation<spacedim, spacedim>::face_iterator;

    std::vector<Point<spacedim>> surf_vertices;
    std::vector<CellData<dim>> surf_cells;
    std::vector<BulkFaceIt> face_per_surf_cell;

    std::map<unsigned int, unsigned int> bulk_to_surf_vertex;
    for (const auto &bulk_cell : bulk_for_extraction->active_cell_iterators())
      for (const unsigned int f : bulk_cell->face_indices())
        if (bulk_cell->at_boundary(f)) {
          const auto face = bulk_cell->face(f);
          const unsigned int n_v = face->n_vertices();
          CellData<dim> cd;
          cd.vertices.resize(n_v);
          for (unsigned int j = 0; j < n_v; ++j) {
            const unsigned int v_index = face->vertex_index(j);
            auto it = bulk_to_surf_vertex.find(v_index);
            if (it == bulk_to_surf_vertex.end()) {
              const unsigned int new_index = surf_vertices.size();
              surf_vertices.push_back(face->vertex(j));
              bulk_to_surf_vertex[v_index] = new_index;
              cd.vertices[j] = new_index;
            } else {
              cd.vertices[j] = it->second;
            }
          }
          cd.material_id = static_cast<types::material_id>(face->boundary_id());
          cd.manifold_id = face->manifold_id();
          surf_cells.push_back(std::move(cd));
          face_per_surf_cell.push_back(face);
        }

    AssertThrow(!surf_cells.empty(), ExcMessage("No boundary faces found"));

    boundary_grid->create_triangulation(surf_vertices, surf_cells,
                                        SubCellData());

    // create_triangulation preserves the order of input cells, so the i-th
    // active surface cell corresponds to the i-th boundary face we pushed.
    unsigned int idx = 0;
    for (const auto &surf_cell : boundary_grid->active_cell_iterators()) {
      surface_to_volume_map[surf_cell] = face_per_surf_cell[idx];
      ++idx;
    }
  }
#endif

  // Bulk DoFs and bulk mapping
  if (is_hex_mesh(parameters.mesh_type)) {
    space_fe = std::make_unique<FE_Q<spacedim>>(
        parameters.bulk_space_finite_element_degree);
    bulk_mapping = std::make_unique<MappingQ1<spacedim>>();
  } else {
    space_fe = std::make_unique<FE_SimplexP<spacedim>>(
        parameters.bulk_space_finite_element_degree);
    bulk_mapping =
        std::make_unique<MappingFE<spacedim>>(FE_SimplexP<spacedim>(1));
  }
  space_dh.distribute_dofs(*space_fe);

  // boundary conditions are imposed weakly through lambda.
  constraints.clear();
  DoFTools::make_hanging_node_constraints(space_dh, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(space_dh.n_dofs(), space_dh.n_dofs());
  DoFTools::make_sparsity_pattern(space_dh, dsp, constraints);
  stiffness_sparsity.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity);

  solution.reinit(space_dh.n_dofs());
  embedding_rhs.reinit(space_dh.n_dofs());

  // Multiplier DoFs on the boundary mesh, plus surface mapping.
  if (is_hex_mesh(parameters.mesh_type)) {
    if (parameters.use_discontinuous_multiplier == true)
      multiplier_fe = std::make_unique<FE_DGQ<dim, spacedim>>(
          parameters.multiplier_finite_element_degree);
    else // continuous
      multiplier_fe = std::make_unique<FE_Q<dim, spacedim>>(
          parameters.multiplier_finite_element_degree);
    surf_mapping = std::make_unique<MappingQ1<dim, spacedim>>();
  } else {
    if (parameters.use_discontinuous_multiplier == true)
      multiplier_fe = std::make_unique<FE_SimplexDGP<dim, spacedim>>(
          parameters.multiplier_finite_element_degree);
    else // continuous
      multiplier_fe = std::make_unique<FE_SimplexP<dim, spacedim>>(
          parameters.multiplier_finite_element_degree);
    surf_mapping = std::make_unique<MappingFE<dim, spacedim>>(
        FE_SimplexP<dim, spacedim>(1));
  }

  boundary_dh.reinit(*boundary_grid);
  boundary_dh.distribute_dofs(*multiplier_fe);

  lambda.reinit(boundary_dh.n_dofs());
  embedded_rhs.reinit(boundary_dh.n_dofs());

  // Mass matrix on the multiplier (boundary) space.
  DynamicSparsityPattern boundary_mass_dsp(boundary_dh.n_dofs(),
                                           boundary_dh.n_dofs());
  DoFTools::make_sparsity_pattern(boundary_dh, boundary_mass_dsp);
  boundary_mass_sparsity.copy_from(boundary_mass_dsp);
  boundary_mass_matrix.reinit(boundary_mass_sparsity);

  // In the non-matching case, surface_to_volume_map contains iterators into
  // a temporary `coarse_bulk` grid that is about to go out of scope; drop
  // them now so nothing dangles. The coupling matrix is assembled via
  // particles in that case.
  if (non_matching_multiplier)
    surface_to_volume_map.clear();

  // Construct the map: "face -> (bulk cell, local face index)", used to
  // recover the bulk owner of each surface cell during coupling assembly.
  for (const auto &bulk_cell : space_dh.active_cell_iterators())
    for (const unsigned int f : bulk_cell->face_indices())
      if (bulk_cell->at_boundary(f))
        face_to_bulk_cell[bulk_cell->face(f)] = {bulk_cell, f};

  deallog << "Bulk dofs: " << space_dh.n_dofs() << std::endl
          << "Multiplier dofs: " << boundary_dh.n_dofs() << std::endl;
}

template <int dim, int spacedim>
void BabuskaProblem<dim, spacedim>::setup_coupling() {
  TimerOutput::Scope timer_section(monitor, "Setup coupling");

  // build particles + sparsity pattern for the coupling matrix
  //
  // Particles are placed at the boundary quadrature points and stored on the
  // (fine) bulk grid. Since ParticleHandler groups particles by their host
  // bulk cell (insertion order is *not* preserved), we encode in each
  // particle's properties the index of the surface cell it came from and the
  // local q-index within that cell. This lets us assemble the coupling
  // matrix B in assemble_system() by iterating particles per bulk cell and
  // decoding the right surface cell and shape function for each particle.
  if (non_matching_multiplier) {
    immersed_particle_handler.clear();

    const unsigned int boundary_q =
        std::max({2u * space_fe->degree + 1u, 2u * multiplier_fe->degree + 1u,
                  parameters.coupling_quadrature_order});
    const auto face_quad =
        make_quadrature<dim>(parameters.mesh_type, boundary_q);

    // Bounding boxes of the bulk grid for global particle insertion.
    std::vector<std::vector<BoundingBox<spacedim>>> global_bboxes;
    {
      std::vector<BoundingBox<spacedim>> all_boxes;
      all_boxes.reserve(space_grid.n_active_cells());
      for (const auto &cell : space_grid.active_cell_iterators())
        if (cell->is_locally_owned())
          all_boxes.emplace_back(cell->bounding_box());
      const auto tree = pack_rtree(all_boxes);
      const auto local_boxes = extract_rtree_level(tree, 1);
      global_bboxes = Utilities::MPI::all_gather(
          space_grid.get_mpi_communicator(), local_boxes);
    }

    const unsigned int n_props = 3; // [JxW, surf_idx, q_idx]
    immersed_particle_handler.initialize(space_grid, *bulk_mapping, n_props);

    std::vector<Point<spacedim>> qpoints;
    std::vector<std::vector<double>> props;
    qpoints.reserve(face_quad.size() * boundary_grid->n_active_cells());
    props.reserve(face_quad.size() * boundary_grid->n_active_cells());

    FEValues<dim, spacedim> fe_v(*surf_mapping, *multiplier_fe, face_quad,
                                 update_JxW_values | update_quadrature_points);

    unsigned int surf_idx = 0;
    for (const auto &surface_cell : boundary_dh.active_cell_iterators()) {
      fe_v.reinit(surface_cell);
      const auto &p = fe_v.get_quadrature_points();
      const auto &J = fe_v.get_JxW_values();
      for (unsigned int q = 0; q < p.size(); ++q) {
        qpoints.push_back(p[q]);
        props.push_back(
            {J[q], static_cast<double>(surf_idx), static_cast<double>(q)});
      }
      ++surf_idx;
    }
    immersed_particle_handler.insert_global_particles(qpoints, global_bboxes,
                                                      props);

    // Verify that all inserted particles were actually located on the host
    // bulk grid. With curved 3D surface meshes that are merely *close* to,
    // but not coincident with, the boundary of space_grid (e.g. when the
    // coarse extracted boundary slightly cuts inside/outside the fine mesh
    // on highly curved geometries) some quadrature points might fail to
    // find a host cell, which would silently corrupt the coupling.
    {
      const types::particle_index n_inserted = qpoints.size();
      const types::particle_index n_located =
          immersed_particle_handler.n_global_particles();
      std::cout << "Particles inserted = " << n_inserted
                << ", located = " << n_located << std::endl;
      AssertThrow(
          n_located == n_inserted,
          ExcMessage("Some boundary quadrature points could not be located "
                     "in the bulk grid. The coupling matrix would then be "
                     "missing those contributions."));
    }

    // Surface-cell index lookup (deterministic ordering of boundary_dh).
    std::vector<typename DoFHandler<dim, spacedim>::active_cell_iterator>
        surf_cells;
    surf_cells.reserve(boundary_grid->n_active_cells());
    for (const auto &c : boundary_dh.active_cell_iterators())
      surf_cells.push_back(c);

    DynamicSparsityPattern dsp(space_dh.n_dofs(), boundary_dh.n_dofs());
    std::vector<types::global_dof_index> bulk_dof_indices(
        space_fe->n_dofs_per_cell());
    std::vector<types::global_dof_index> surf_dof_indices(
        multiplier_fe->n_dofs_per_cell());

    auto particle = immersed_particle_handler.begin();
    while (particle != immersed_particle_handler.end()) {
      const auto bulk_cell = particle->get_surrounding_cell();
      const typename DoFHandler<spacedim>::cell_iterator dh_cell(*bulk_cell,
                                                                 &space_dh);
      dh_cell->get_dof_indices(bulk_dof_indices);

      const auto pic = immersed_particle_handler.particles_in_cell(bulk_cell);
      for (const auto &p : pic) {
        const unsigned int s_idx =
            static_cast<unsigned int>(p.get_properties()[1]);
        surf_cells[s_idx]->get_dof_indices(surf_dof_indices);
        for (const auto i : bulk_dof_indices)
          for (const auto j : surf_dof_indices)
            dsp.add(i, j);
      }
      particle = pic.end();
    }

    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);
    return;
  }

  //  Matching case: face-based sparsity. This is simpler, because we know we
  //  are on the boundary of the grid.
  // Sparsity pattern for C: rows = bulk DoFs, cols = multiplier DoFs.
  DynamicSparsityPattern dsp(space_dh.n_dofs(), boundary_dh.n_dofs());

  std::vector<types::global_dof_index> bulk_dof_indices(
      space_fe->n_dofs_per_cell());
  std::vector<types::global_dof_index> surf_dof_indices(
      multiplier_fe->n_dofs_per_cell());

  for (const auto &surface_cell : boundary_dh.active_cell_iterators()) {
    const auto bulk_face = surface_to_volume_map.at(surface_cell);
    const auto &owner = face_to_bulk_cell.at(bulk_face);
    const auto &owner_cell = owner.first;

    owner_cell->get_dof_indices(bulk_dof_indices);
    surface_cell->get_dof_indices(surf_dof_indices);

    for (const auto i : bulk_dof_indices)
      for (const auto j : surf_dof_indices)
        dsp.add(i, j);
  }

  coupling_sparsity.copy_from(dsp);
  coupling_matrix.reinit(coupling_sparsity);
}

template <int dim, int spacedim>
void BabuskaProblem<dim, spacedim>::assemble_system() {
  TimerOutput::Scope timer_section(monitor, "Assemble system");

  {
    const auto quadrature = make_quadrature<spacedim>(parameters.mesh_type,
                                                      2 * space_fe->degree + 1);
    FEValues<spacedim> fe_values(*bulk_mapping, *space_fe, quadrature,
                                 update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);

    const unsigned int dofs_per_cell = space_fe->n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> rhs_values(n_q_points);

    // The matching path adds the gamma*(1/h)*(u,v)_{Gamma} term directly
    // here while looping over bulk cells, by visiting their boundary
    // faces. In the non-matching path the augmentation is integrated
    // particle-by-particle in solve(), on the same boundary quadrature
    // points used to build C, so we must NOT add it here -- otherwise K
    // and C would live on different surfaces. The constants below must
    // match those used in solve() for the Schur preconditioner.
    const bool augment_via_faces = !non_matching_multiplier;
    const double gamma_aug = parameters.augmentation_gamma_assembly;
    const double invW_scale_aug =
        augment_via_faces ? 1.0 / std::pow(GridTools::maximal_cell_diameter(
                                               *boundary_grid, *surf_mapping),
                                           parameters.augmentation_h_power)
                          : 0.0;
    const unsigned int boundary_q_aug =
        std::max({2u * space_fe->degree + 1u, 2u * multiplier_fe->degree + 1u,
                  parameters.coupling_quadrature_order});
    const auto face_quad_aug =
        make_quadrature<dim>(parameters.mesh_type, boundary_q_aug);
    FEFaceValues<spacedim> fe_face_values_aug(
        *bulk_mapping, *space_fe, face_quad_aug,
        update_values | update_quadrature_points | update_JxW_values);
    std::vector<double> g_values_aug(face_quad_aug.size());

    // Manufactured boundary data is the trace of the exact solution, so the
    // sine case is correct on any domain (it vanishes on the unit square).
    // For "affine", -Delta u = 0 and f = u,
    // so the same object serves as boundary data and as right-hand side.
    ManufacturedSolution<spacedim> manufactured_g_aug;
    ManufacturedRhs<spacedim> manufactured_rhs;
    AffineXY<spacedim> affine_xy;
    const Function<spacedim> *g_manufactured = &manufactured_g_aug;
    const Function<spacedim> *rhs_manufactured = &manufactured_rhs;
    if (parameters.manufactured_solution_type == "affine") {
      g_manufactured = &affine_xy;
      rhs_manufactured = &affine_xy;
    }
    const Function<spacedim> &g_for_aug =
        parameters.use_manufactured_solution
            ? *g_manufactured
            : static_cast<const Function<spacedim> &>(g_function);

    const Function<spacedim> &rhs_for_assembly =
        parameters.use_manufactured_solution
            ? *rhs_manufactured
            : static_cast<const Function<spacedim> &>(rhs_function);

    for (const auto &cell : space_dh.active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs = 0.;

      rhs_for_assembly.value_list(fe_values.get_quadrature_points(),
                                  rhs_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double JxW = fe_values.JxW(q);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                 fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) *
                JxW;
          cell_rhs(i) += fe_values.shape_value(i, q) * rhs_values[q] * JxW;
        }
      }

      // AL augmentation: add gamma*(1/h)*(u,v) on each boundary face of this
      // cell directly into cell_matrix, and the consistent gamma*(1/h)*(g,v)
      // contribution into cell_rhs.
      if (augment_via_faces && cell->at_boundary()) {
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary()) {
            fe_face_values_aug.reinit(cell, f);
            g_for_aug.value_list(fe_face_values_aug.get_quadrature_points(),
                                 g_values_aug);
            for (unsigned int q = 0; q < face_quad_aug.size(); ++q) {
              const double JxW = fe_face_values_aug.JxW(q);
              const double gq = g_values_aug[q];
              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double v_i = fe_face_values_aug.shape_value(i, q);
                cell_rhs(i) += gamma_aug * invW_scale_aug * v_i * gq * JxW;
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                      gamma_aug *                          // gamma
                      invW_scale_aug *                     // h^{-1}
                      v_i *                                // phi_i
                      fe_face_values_aug.shape_value(j, q) // phi_j
                      * JxW;
              }
            }
          }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                             local_dof_indices,
                                             stiffness_matrix, embedding_rhs);
    }
  }

  // Single boundary quadrature rule, shared by M, the multiplier RHS, the
  // coupling matrix C, and (in solve()) the particle-based AL augmentation.
  const unsigned int boundary_q =
      std::max({2u * space_fe->degree + 1u, 2u * multiplier_fe->degree + 1u,
                parameters.coupling_quadrature_order});
  const auto face_quad = make_quadrature<dim>(parameters.mesh_type, boundary_q);

  // RHS for the multiplier equation (g = trace of the exact solution)
  ManufacturedSolution<spacedim> manufactured_g;
  AffineXY<spacedim> affine_xy_g;
  const Function<spacedim> *g_manufactured_rhs = &manufactured_g;
  if (parameters.manufactured_solution_type == "affine")
    g_manufactured_rhs = &affine_xy_g;
  const Function<spacedim> &g_for_rhs =
      parameters.use_manufactured_solution
          ? *g_manufactured_rhs
          : static_cast<const Function<spacedim> &>(g_function);
  VectorTools::create_right_hand_side(*surf_mapping, boundary_dh, face_quad,
                                      g_for_rhs, embedded_rhs);

  // Mass matrix on the multiplier space (defined on boundary mesh).
  MatrixTools::create_mass_matrix(*surf_mapping, boundary_dh, face_quad,
                                  boundary_mass_matrix);

  // Assemble the coupling matrix C with entries C_{ij} = (phi_i^bulk,
  // phi_j^surf) integrated on the boundary face. pair each surface cell with
  // its bulk face and use the same quadrature on both sides so the points
  // coincide on the surface and on the bulk face.

  FEValues<dim, spacedim> fe_surface_values(
      *surf_mapping, *multiplier_fe, face_quad,
      update_values | update_quadrature_points | update_JxW_values);

  const unsigned int n_bulk_dofs = space_fe->n_dofs_per_cell();
  const unsigned int n_surf_dofs = multiplier_fe->n_dofs_per_cell();

  FullMatrix<double> local_C(n_bulk_dofs, n_surf_dofs);
  std::vector<types::global_dof_index> bulk_dof_indices(n_bulk_dofs);
  std::vector<types::global_dof_index> surf_dof_indices(n_surf_dofs);

  if (non_matching_multiplier) {
    // Particle-based assembly: iterate particles per host bulk cell and use
    // the surface-cell index encoded in the particle properties to find the
    // multiplier basis values at the right local q-index.
    std::vector<typename DoFHandler<dim, spacedim>::active_cell_iterator>
        surf_cells;
    surf_cells.reserve(boundary_grid->n_active_cells());
    for (const auto &c : boundary_dh.active_cell_iterators())
      surf_cells.push_back(c);

    auto particle = immersed_particle_handler.begin();
    while (particle != immersed_particle_handler.end()) {
      const auto bulk_cell = particle->get_surrounding_cell();
      const typename DoFHandler<spacedim>::cell_iterator dh_cell(*bulk_cell,
                                                                 &space_dh);
      dh_cell->get_dof_indices(bulk_dof_indices);

      const auto pic = immersed_particle_handler.particles_in_cell(bulk_cell);
      for (const auto &p : pic) {
        const Point<spacedim> ref_q = p.get_reference_location();
        const auto &props_p = p.get_properties();
        const double JxW = props_p[0];
        const unsigned int s_idx = static_cast<unsigned int>(props_p[1]);
        const unsigned int q_idx = static_cast<unsigned int>(props_p[2]);

        const auto &surface_cell = surf_cells[s_idx];
        surface_cell->get_dof_indices(surf_dof_indices);
        fe_surface_values.reinit(surface_cell);

        for (unsigned int i = 0; i < n_bulk_dofs; ++i) {
          const double v_i = space_fe->shape_value(i, ref_q);
          for (unsigned int j = 0; j < n_surf_dofs; ++j) {
            const double mu_j = fe_surface_values.shape_value(j, q_idx);
            coupling_matrix.add(bulk_dof_indices[i], surf_dof_indices[j],
                                v_i * mu_j * JxW);
          }
        }
      }
      particle = pic.end();
    }
  } else {
    // Matching path: drive quadrature from the surface cell and evaluate the
    // bulk shape values via transform_real_to_unit_cell. This is robust to
    // face orientation/flip/rotation, which would otherwise misalign the
    // local q-indices on the bulk face vs. the surface cell when the
    // multiplier basis is not piecewise constant (e.g. DGQ1).
    for (const auto &surface_cell : boundary_dh.active_cell_iterators()) {
      const auto bulk_face = surface_to_volume_map.at(surface_cell);
      const auto &owner = face_to_bulk_cell.at(bulk_face);
      const auto &bulk_cell = owner.first;

      fe_surface_values.reinit(surface_cell);

      bulk_cell->get_dof_indices(bulk_dof_indices);
      surface_cell->get_dof_indices(surf_dof_indices);

      const auto &q_phys = fe_surface_values.get_quadrature_points();

      local_C = 0;
      for (unsigned int q = 0; q < face_quad.size(); ++q) {
        const Point<spacedim> ref_q =
            bulk_mapping->transform_real_to_unit_cell(bulk_cell, q_phys[q]);
        const double JxW = fe_surface_values.JxW(q);
        for (unsigned int i = 0; i < n_bulk_dofs; ++i) {
          const double v_i = space_fe->shape_value(i, ref_q);
          for (unsigned int j = 0; j < n_surf_dofs; ++j) {
            const double mu_j = fe_surface_values.shape_value(j, q);
            local_C(i, j) += v_i * mu_j * JxW;
          }
        }
      }

      constraints.distribute_local_to_global(local_C, bulk_dof_indices,
                                             surf_dof_indices, coupling_matrix);
    }
  }

// Sanity check for coupling matrix against the boundary measure obtained by
// integrating 1 with the multiplier mass matrix
#ifdef DEBUG
  std::cout << "**** Sanity checks for coupling matrix: they run only in DEBUG "
               "mode. ****"
            << std::endl;
  {
    Vector<double> ones_mult(boundary_dh.n_dofs());
    ones_mult = 1.0;
    Vector<double> M_ones(boundary_dh.n_dofs());
    boundary_mass_matrix.vmult(M_ones, ones_mult);
    const double boundary_measure = ones_mult * M_ones; // 1^T M 1

    double C_total = 0.0;
    for (auto it = coupling_matrix.begin(); it != coupling_matrix.end(); ++it)
      C_total += it->value();

    deallog << "Coupling sanity check: sum(C) = " << C_total
            << ", |partial Omega| = " << boundary_measure
            << ", relative error = "
            << std::abs(C_total - boundary_measure) /
                   std::max(boundary_measure, 1e-30)
            << std::endl;
    AssertThrow(std::abs(C_total - boundary_measure) <
                    1e-10 * std::max(boundary_measure, 1.0),
                ExcMessage("Coupling matrix sum does not match the boundary "
                           "measure: assembly is likely incorrect."));
  }

  // Stronger sanity check, especially valuable in the non-matching case:
  // pick an *affine* test function phi (exactly representable in P1/Q1 on
  // any cell), interpolate it into the bulk space, and verify that
  //
  //     C^T * u_interp  ==  ( phi, mu_j )_{partial Omega}
  //
  // for every multiplier basis function mu_j. The right-hand side is just
  // an L2 boundary RHS, computed independently with create_right_hand_side.
  // Because phi is affine, the bulk Q1 interpolant equals phi pointwise on
  // the boundary trace, so the two sides must coincide up to machine
  // precision *if and only if* every (bulk_face, surface_cell) pair --
  // resp. every particle -- is matched correctly. Unlike the sum-check
  // above this test is sensitive to permutations of the columns of C and
  // therefore catches misindexed coupling assembly (the bug we hit on
  // the non-matching path before the per-particle properties fix).
  {
    class Affine : public Function<spacedim> {
    public:
      double value(const Point<spacedim> &p,
                   const unsigned int /*c*/ = 0) const override {
        double s = 0.5; // nonzero constant part
        for (unsigned int d = 0; d < spacedim; ++d)
          s += (d + 1) * p[d];
        return s;
      }
    } phi;

    Vector<double> u_interp(space_dh.n_dofs());
    VectorTools::interpolate(*bulk_mapping, space_dh, phi, u_interp);

    Vector<double> CT_u(boundary_dh.n_dofs());
    coupling_matrix.Tvmult(CT_u, u_interp);

    Vector<double> phi_rhs(boundary_dh.n_dofs());
    VectorTools::create_right_hand_side(*surf_mapping, boundary_dh, face_quad,
                                        phi, phi_rhs);

    Vector<double> diff(boundary_dh.n_dofs());
    diff = CT_u;
    diff -= phi_rhs;
    const double err = diff.l2_norm();
    const double ref = std::max(phi_rhs.l2_norm(), 1.e-30);
    deallog << "Coupling sanity (affine interp): || C^T u_interp - g_vec || = "
            << err << "   (relative = " << err / ref << ")" << std::endl;
    AssertThrow(err < 1.e-9 * ref,
                ExcMessage("Coupling matrix is inconsistent with an affine "
                           "test interpolant: particle/cell indexing in the "
                           "coupling assembly is likely wrong."));
  }
#endif

  std::cout << "Assembly complete." << std::endl;
}

template <int dim, int spacedim> void BabuskaProblem<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(monitor, "Solve system");

  // Build the saddle-point block operator
  //   [ K   C  ] [ u      ]   [ f ]
  //   [ C^T 0  ] [ lambda ] = [ g ]
  // and solve with (preconditioned) FGMRES
  auto K = linear_operator(
      stiffness_matrix); // notice that stiffness_matrix will be augmented later
  auto C = linear_operator(coupling_matrix);
  auto Ct = transpose_operator(C);

  // Zero (1,1) block: (multiplier,multiplier)
  const auto Zero = null_operator(Ct * C);

  // Inversion of the boundary mass matrix M, used inside the Schur
  // preconditioner. Two options:
  //  - direct (UMFPACK),
  //  - iterative (CG + Jacobi)
  SparseDirectUMFPACK M_inv_umfpack;
  PreconditionJacobi<SparseMatrix<double>> M_jacobi;
  ReductionControl mass_solver_control(1000, 1.e-12, 1.e-10, false, false);
  SolverCG<Vector<double>> mass_cg(mass_solver_control);
  LinearOperator<Vector<double>, Vector<double>> invM;
  if (parameters.iterative_inversion_mass_matrix) {
    std::cout << "Setting up CG + Jacobi for the multiplier mass matrix..."
              << std::endl;
    M_jacobi.initialize(boundary_mass_matrix);
    invM = inverse_operator(linear_operator(boundary_mass_matrix), mass_cg,
                            M_jacobi);
    std::cout << "Done." << std::endl;
  } else {
    std::cout << "Factorizing mass matrix on multiplier space..." << std::endl;
    M_inv_umfpack.initialize(boundary_mass_matrix);
    invM = linear_operator(boundary_mass_matrix, M_inv_umfpack);
    std::cout << "Done." << std::endl;
  }

  // Surface and bulk mappings are stored as members; reuse them here.
  const Mapping<dim, spacedim> &embedded_mapping = *surf_mapping;
  // Two distinct augmentation factors (see Parameters):
  //  - gamma_assembly is the actual augmentation written into K below;
  //    setting it to 0 yields the raw un-augmented saddle-point system.
  //  - gamma is the factor used inside the AL block preconditioner; it is
  //    kept independent so the preconditioner stays well-defined even when
  //    we deliberately disable system-level augmentation.
  const double gamma_assembly = parameters.augmentation_gamma_assembly;
  double gamma = parameters.augmentation_gamma_prec;
  // Per-cycle wall times of the two augmentation paths (comparison mode).
  double t_particle_aug = 0.;
  double t_alg_aug = 0.;
  TrilinosWrappers::PreconditionAMG amg_prec;
  auto prec_for_cg = null_operator(K);
  double invW_scale = 1.0; // = h^{-alpha}, set below once the boundary mesh
                           // size is known
  double h_boundary = 1.0; // boundary-cell diameter (for the matrix export)
  {
    const double h_immersed =
        GridTools::maximal_cell_diameter(*boundary_grid, embedded_mapping);
    invW_scale = 1.0 / std::pow(h_immersed, parameters.augmentation_h_power);
    h_boundary = h_immersed;

    if (parameters.compare_algebraic_augmentation)
      AssertThrow(
          non_matching_multiplier,
          ExcMessage("The option 'Compare algebraic augmentation' requires a "
                     "non-matching multiplier mesh "
                     "('Boundary coarsening level' > 0)."));

    if (non_matching_multiplier) {
      // Snapshot bare K: the optional algebraic comparison must start from
      // the same un-augmented matrix as the particle-based path.
      SparseMatrix<double> K_pre_aug;
      if (parameters.compare_algebraic_augmentation) {
        K_pre_aug.reinit(stiffness_sparsity);
        K_pre_aug.copy_from(stiffness_matrix);
      }

      Particles::ParticleHandler<spacedim> immersed_particle_handler;

      const unsigned int boundary_q =
          std::max({2u * space_fe->degree + 1u, 2u * multiplier_fe->degree + 1u,
                    parameters.coupling_quadrature_order});
      auto immersed_quadrature =
          make_quadrature<dim>(parameters.mesh_type, boundary_q);
      ALUtils::initialize_particles<spacedim, dim, spacedim>(
          immersed_particle_handler, space_dh, boundary_dh, *bulk_mapping,
          embedded_mapping, immersed_quadrature);

      // Assemble:
      // gamma*(1/h)*(u,v)_{Gamma} into the (0,0) block of the augmented matrix,
      // and the consistent gamma*(1/h)*(g,v)_{Gamma} contribution into the bulk
      // rhs
      ManufacturedSolution<spacedim> manufactured_g;
      AffineXY<spacedim> affine_xy_aug;
      const Function<spacedim> *g_manufactured_aug = &manufactured_g;
      if (parameters.manufactured_solution_type == "affine")
        g_manufactured_aug = &affine_xy_aug;
      const Function<spacedim> &g_for_aug =
          parameters.use_manufactured_solution
              ? *g_manufactured_aug
              : static_cast<const Function<spacedim> &>(g_function);

      std::vector<types::global_dof_index> background_dof_indices(
          space_fe->n_dofs_per_cell());

      FullMatrix<double> local_matrix(space_fe->n_dofs_per_cell(),
                                      space_fe->n_dofs_per_cell());
      Vector<double> local_rhs(space_fe->n_dofs_per_cell());

      monitor.enter_subsection("Particle-based (0,0)-block augmentation");
      Timer timer_particle;
      auto particle = immersed_particle_handler.begin();
      while (particle != immersed_particle_handler.end()) {
        local_matrix = 0;
        local_rhs = 0;
        const auto &cell = particle->get_surrounding_cell();
        const auto &dh_cell =
            typename DoFHandler<spacedim>::cell_iterator(*cell, &space_dh);
        dh_cell->get_dof_indices(background_dof_indices); // background dofs

        const auto pic = immersed_particle_handler.particles_in_cell(cell);
        Assert(pic.begin() == particle, ExcInternalError());
        for (const auto &p : pic) {
          const Point<spacedim> ref_q = p.get_reference_location();
          const double JxW = p.get_properties()[0];
          const double gq = g_for_aug.value(p.get_location());

          for (unsigned int i = 0; i < space_fe->n_dofs_per_cell(); ++i) {
            const double v_i = space_fe->shape_value(i, ref_q);
            local_rhs(i) += gamma_assembly * invW_scale * v_i * gq * JxW;
            for (unsigned int j = 0; j < space_fe->n_dofs_per_cell(); ++j) {
              local_matrix(i, j) += gamma_assembly *
                                    invW_scale * // gamma * (1/h)
                                    v_i * space_fe->shape_value(j, ref_q) * JxW;
            }
          }
        }

        constraints.distribute_local_to_global(
            local_matrix, background_dof_indices, stiffness_matrix);
        constraints.distribute_local_to_global(
            local_rhs, background_dof_indices, embedding_rhs);

        particle = pic.end();
      }
      timer_particle.stop();
      t_particle_aug = timer_particle.wall_time();
      monitor.leave_subsection();

      // Optional comparison: assemble the same augmentation algebraically as
      // gamma*(1/h) * C diag(M)^{-1} C^T (lumped boundary mass). Unlike the
      // particle path -- which stays inside the bulk cell stencil -- the
      // triple product couples all bulk DoFs sharing a multiplier DoF and
      // therefore adds fill-in to K. The result is discarded; only timing,
      // sparsity and an informational action difference are reported.
      if (parameters.compare_algebraic_augmentation) {
        monitor.enter_subsection(
            "Algebraic (0,0) block augmentation (C diag(M)^-1 C^T)");
        Timer timer_alg;

        const unsigned int n_lambda = boundary_dh.n_dofs();

        // Explicit transpose of C (multiplier x bulk), needed as the right
        // factor of the sparse triple product below.
        SparsityPattern coupling_T_sparsity;
        {
          DynamicSparsityPattern dsp_T(n_lambda, space_dh.n_dofs());
          for (auto it = coupling_matrix.begin(); it != coupling_matrix.end();
               ++it)
            dsp_T.add(it->column(), it->row());
          coupling_T_sparsity.copy_from(dsp_T);
        }
        SparseMatrix<double> coupling_T(coupling_T_sparsity);
        for (auto it = coupling_matrix.begin(); it != coupling_matrix.end();
             ++it)
          coupling_T.set(it->column(), it->row(), it->value());

        // Diagonal weights gamma*(1/h) * diag(M)^{-1}, baked into the middle
        // factor of the triple product.
        Vector<double> winv(n_lambda);
        for (unsigned int k = 0; k < n_lambda; ++k)
          winv[k] = gamma_assembly * invW_scale /
                    boundary_mass_matrix.diag_element(k);

        // Sparse triple product  aug_alg = C * diag(winv) * C^T; mmult builds
        // the (filled-in) sparsity pattern of the result itself.
        SparsityPattern aug_sparsity(space_dh.n_dofs(), space_dh.n_dofs(), 0);
        aug_sparsity.compress();
        SparseMatrix<double> aug_alg(aug_sparsity);
        coupling_matrix.mmult(aug_alg, coupling_T, winv);

        // K_alg = K + C diag(winv) C^T on the union of the two patterns.
        DynamicSparsityPattern dsp_alg(space_dh.n_dofs(), space_dh.n_dofs());
        for (auto it = stiffness_sparsity.begin();
             it != stiffness_sparsity.end(); ++it)
          dsp_alg.add(it->row(), it->column());
        for (auto it = aug_alg.begin(); it != aug_alg.end(); ++it)
          dsp_alg.add(it->row(), it->column());

        SparsityPattern alg_sparsity;
        alg_sparsity.copy_from(dsp_alg);
        SparseMatrix<double> K_alg(alg_sparsity);
        for (auto it = K_pre_aug.begin(); it != K_pre_aug.end(); ++it)
          K_alg.set(it->row(), it->column(), it->value());
        for (auto it = aug_alg.begin(); it != aug_alg.end(); ++it)
          K_alg.add(it->row(), it->column(), it->value());

        timer_alg.stop();
        t_alg_aug = timer_alg.wall_time();
        monitor.leave_subsection();

        // Informational only: with a continuous multiplier the lumped
        // triple product approximates, but does not equal, the exact
        // particle-quadrature boundary mass term.
        Vector<double> x(space_dh.n_dofs());
        Vector<double> y_particle(space_dh.n_dofs());
        Vector<double> y_alg(space_dh.n_dofs());
        x = 1.0;
        stiffness_matrix.vmult(y_particle, x);
        K_alg.vmult(y_alg, x);
        y_alg -= y_particle;
        deallog << "Algebraic augmentation comparison: "
                << "nnz(K_aug particles) = "
                << stiffness_matrix.n_nonzero_elements()
                << ", nnz(K + g/h C diag(M)^-1 C^T) = "
                << K_alg.n_nonzero_elements() << ", rel. action diff on ones = "
                << y_alg.l2_norm() / std::max(y_particle.l2_norm(), 1e-30)
                << std::endl;
        deallog << "Augmentation build times [s]: particles = "
                << t_particle_aug
                << ", algebraic triple product = " << t_alg_aug << std::endl;
      }
    }

    // In the matching case the augmentation has already been added to
    // stiffness_matrix during assemble_system() via the boundary-face
    // loop, so nothing to do here.
    std::cout
        << "Building AMG preconditioner for the augmented stiffness matrix..."
        << std::endl;
    amg_prec.initialize(stiffness_matrix);
    prec_for_cg = linear_operator(stiffness_matrix, amg_prec);
    std::cout << "Done." << std::endl;
  }

  auto Aug = linear_operator(stiffness_matrix);
  auto AA =
      block_operator<2, 2, BlockVector<double>>({{{{Aug, C}}, {{Ct, Zero}}}});

  BlockVector<double> solution_block;
  BlockVector<double> system_rhs_block;
  AA.reinit_domain_vector(solution_block, false);
  AA.reinit_range_vector(system_rhs_block, false);

  system_rhs_block.block(0) = embedding_rhs;
  system_rhs_block.block(1) = embedded_rhs;

  SolverCG<Vector<double>> solver_lagrangian(inner_solver_control);
  auto A_inv = inverse_operator(Aug, solver_lagrangian,
                                prec_for_cg); //! linear solver augmented
  //  invW = (1/h) * M^{-1}, used both as the Schur preconditioner
  // (-gamma * invW) and to build the augmentation/RHS terms.
  auto invW = invW_scale * invM;
  // The preconditioner's `Ct` argument is the (0,1) block (multiplier->bulk)
  BlockPreconditionerAugmentedLagrangian AL_prec{A_inv, Ct, C, invW, gamma};

  SolverFGMRES<BlockVector<double>> solver(outer_solver_control);

  // count total inner CG iterations across all applications of A_inv during the
  // outer FGMRES solve. Each outer step triggers one inner CG solve
  // preconditioned by AMG; this slot is called on every CG check() (including
  // the initial residual check), so the total is a faithful upper bound on the
  // cumulative CG work. Dividing by the outer iteration count gives the average
  // inner CG steps per outer step, which is the number we care about
  unsigned int total_inner_iters = 0;
  auto inner_signal_conn = solver_lagrangian.connect(
      [&total_inner_iters](
          const unsigned int /*step*/, const double /*res*/,
          const Vector<double> & /*current*/) -> SolverControl::State {
        ++total_inner_iters;
        return SolverControl::success;
      });

  solver.solve(AA, solution_block, system_rhs_block, AL_prec);

  inner_signal_conn.disconnect();
  const unsigned int outer_its = outer_solver_control.last_step();
  const double inner_per_outer =
      (outer_its == 0) ? double(total_inner_iters)
                       : double(total_inner_iters) / double(outer_its);

  // Record results for this refinement cycle in the convergence table.
  convergence_table.add_value("cells", space_grid.n_active_cells());
  convergence_table.add_value("dofs_u", space_dh.n_dofs());
  convergence_table.add_value("dofs_lambda", boundary_dh.n_dofs());
  convergence_table.add_value("outer_its", outer_its);
  // convergence_table.add_value("inner_cg_total", total_inner_iters);
  convergence_table.add_value("inner_cg/outer", inner_per_outer);
  convergence_table.set_precision("inner_cg/outer", 1);
  if (parameters.compare_algebraic_augmentation) {
    convergence_table.add_value("t_part_aug[s]", t_particle_aug);
    convergence_table.add_value("t_alg_aug[s]", t_alg_aug);
    convergence_table.set_scientific("t_part_aug[s]", true);
    convergence_table.set_scientific("t_alg_aug[s]", true);
    convergence_table.set_precision("t_part_aug[s]", 3);
    convergence_table.set_precision("t_alg_aug[s]", 3);
  }

  solution = solution_block.block(0);
  lambda = solution_block.block(1);
  constraints.distribute(solution);

  // Dump the three sparse matrices that define the saddle-point
  // system in MATLAB triplet (i, j, value) text format.
  if (parameters.export_matrices_for_matlab) {
    AssertThrow(boundary_dh.n_dofs() <= parameters.matrix_export_max_dofs,
                ExcMessage("Multiplier DoFs exceed 'Matrix export max DoFs'; "
                           "lower the refinement / cycle count, or raise the "
                           "cap in the parameter file."));

    static unsigned int export_counter = 0;
    const unsigned int cycle_idx = export_counter++;
    auto write_triplets =
        [cycle_idx](const SparseMatrix<double> &A, const std::string &name,
                    const unsigned int n_rows, const unsigned int n_cols) {
          const std::string fname =
              name + "_cycle" + Utilities::int_to_string(cycle_idx, 2) + ".txt";
          std::ofstream out(fname);
          out.precision(16);
          out << std::scientific;
          for (auto it = A.begin(); it != A.end(); ++it)
            out << (it->row() + 1) << ' ' << (it->column() + 1) << ' '
                << it->value() << '\n';
          // Sentinel triplet so spconvert / sparse(i,j,v) returns a
          // matrix of the correct size even if the last row/column is
          // structurally empty.
          out << n_rows << ' ' << n_cols << " 0\n";
          deallog << "Wrote " << fname << " (" << n_rows << " x " << n_cols
                  << ")" << std::endl;
        };

    if (std::fabs(parameters.augmentation_gamma_assembly) > 0.)
      write_triplets(stiffness_matrix, "K_aug", space_dh.n_dofs(),
                     space_dh.n_dofs());
    else
      write_triplets(stiffness_matrix, "K", space_dh.n_dofs(),
                     space_dh.n_dofs());
    write_triplets(coupling_matrix, "C", space_dh.n_dofs(),
                   boundary_dh.n_dofs());
    write_triplets(boundary_mass_matrix, "M", boundary_dh.n_dofs(),
                   boundary_dh.n_dofs());

    // Boundary mesh size h, so offline scripts can rebuild gamma = c/h.
    {
      const std::string fname =
          "h_cycle" + Utilities::int_to_string(cycle_idx, 2) + ".txt";
      std::ofstream out(fname);
      out.precision(16);
      out << std::scientific << h_boundary << '\n';
    }

    // Sanity check of the AL identity in the DGQ0 / matching-mesh case:
    // the bilinear form  gamma * ( P_h u_h, v_h )_Gamma  --  where P_h is
    // the L^2 projection onto the multiplier space -- has matrix
    //   B_proj_{ij} = sum_F (1/|F|) (int_F phi_i)(int_F phi_j)
    // (DGQ0 makes M block-diagonal with 1x1 blocks of size |F|, so the
    // L^2 projector is per-facet averaging). The AL literature claims
    //   B_proj  ==  C * M^{-1} * C^T,
    // which is what the algebraic augmentation A + gamma * C M^{-1} C^T
    // implicitly assembles. We dump B_proj here so that the Python
    // notebook can verify the identity numerically.
    if (!non_matching_multiplier && parameters.use_discontinuous_multiplier &&
        parameters.multiplier_finite_element_degree == 0) {
      SparseMatrix<double> projection_aug_matrix(stiffness_sparsity);

      const unsigned int boundary_q =
          std::max({2u * space_fe->degree + 1u, 2u * multiplier_fe->degree + 1u,
                    parameters.coupling_quadrature_order});
      const auto face_quad =
          make_quadrature<dim>(parameters.mesh_type, boundary_q);
      FEFaceValues<spacedim> fe_face_bulk(*bulk_mapping, *space_fe, face_quad,
                                          update_values | update_JxW_values);

      const unsigned int n_bulk_dofs = space_fe->n_dofs_per_cell();
      Vector<double> a_local(n_bulk_dofs);
      std::vector<types::global_dof_index> bulk_dof_indices(n_bulk_dofs);

      for (const auto &surface_cell : boundary_dh.active_cell_iterators()) {
        const auto bulk_face = surface_to_volume_map.at(surface_cell);
        const auto &owner = face_to_bulk_cell.at(bulk_face);
        const auto &bulk_cell = owner.first;
        const unsigned int local_face = owner.second;

        fe_face_bulk.reinit(bulk_cell, local_face);
        bulk_cell->get_dof_indices(bulk_dof_indices);

        a_local = 0.;
        double face_area = 0.;
        for (unsigned int q = 0; q < face_quad.size(); ++q) {
          const double JxW = fe_face_bulk.JxW(q);
          face_area += JxW;
          for (unsigned int i = 0; i < n_bulk_dofs; ++i)
            a_local(i) += fe_face_bulk.shape_value(i, q) * JxW;
        }

        for (unsigned int i = 0; i < n_bulk_dofs; ++i)
          for (unsigned int j = 0; j < n_bulk_dofs; ++j)
            projection_aug_matrix.add(bulk_dof_indices[i], bulk_dof_indices[j],
                                      a_local(i) * a_local(j) / face_area);
      }

      write_triplets(projection_aug_matrix, "B_proj", space_dh.n_dofs(),
                     space_dh.n_dofs());
    }
  }

  // If the manufactured solution is active, compute L2 / H1-seminorm errors
  // of u against u_ex on the bulk mesh
  const double h_bulk =
      GridTools::maximal_cell_diameter(space_grid, *bulk_mapping);
  convergence_table.add_value("h", h_bulk);

  if (parameters.use_manufactured_solution) {
    ManufacturedSolution<spacedim> u_ex_sine;
    AffineXY<spacedim> u_ex_affine;
    const Function<spacedim> *u_ex_ptr = &u_ex_sine;
    if (parameters.manufactured_solution_type == "affine")
      u_ex_ptr = &u_ex_affine;
    const Function<spacedim> &u_ex = *u_ex_ptr;
    const auto error_quad = make_quadrature<spacedim>(parameters.mesh_type,
                                                      2 * space_fe->degree + 2);

    Vector<double> per_cell_l2(space_grid.n_active_cells());
    VectorTools::integrate_difference(*bulk_mapping, space_dh, solution, u_ex,
                                      per_cell_l2, error_quad,
                                      VectorTools::L2_norm);
    const double l2_err = VectorTools::compute_global_error(
        space_grid, per_cell_l2, VectorTools::L2_norm);

    Vector<double> per_cell_h1(space_grid.n_active_cells());
    VectorTools::integrate_difference(*bulk_mapping, space_dh, solution, u_ex,
                                      per_cell_h1, error_quad,
                                      VectorTools::H1_seminorm);
    const double h1_err = VectorTools::compute_global_error(
        space_grid, per_cell_h1, VectorTools::H1_seminorm);

    convergence_table.add_value("L2_u", l2_err);
    convergence_table.add_value("H1_u", h1_err);

    // Multiplier error: from the first block row, <lambda, v>_Gamma =
    // (f,v) - a(u,v), so the exact multiplier is the negative normal flux
    // lambda_ex = -grad(u_ex).n. Integrate |lambda_h - lambda_ex|^2 on the
    // multiplier mesh (outward normals from the codim-1 FEValues).
    double lambda_l2_sq = 0.;
    {
      const auto lam_quad = make_quadrature<dim>(parameters.mesh_type,
                                                 2 * multiplier_fe->degree + 2);
      FEValues<dim, spacedim> fe_lam(*surf_mapping, *multiplier_fe, lam_quad,
                                     update_values | update_quadrature_points |
                                         update_JxW_values |
                                         update_normal_vectors);
      std::vector<double> lam_h(lam_quad.size());
      for (const auto &cell : boundary_dh.active_cell_iterators()) {
        fe_lam.reinit(cell);
        fe_lam.get_function_values(lambda, lam_h);
        for (unsigned int q = 0; q < lam_quad.size(); ++q) {
          const Tensor<1, spacedim> n = fe_lam.normal_vector(q);
          const double lam_ex =
              -(u_ex.gradient(fe_lam.quadrature_point(q)) * n);
          lambda_l2_sq +=
              (lam_h[q] - lam_ex) * (lam_h[q] - lam_ex) * fe_lam.JxW(q);
        }
      }
    }
    const double lambda_l2_err = std::sqrt(lambda_l2_sq);
    convergence_table.add_value("L2_lambda", lambda_l2_err);
    deallog << "|| u - u_ex ||_{L2} = " << l2_err
            << "   | u - u_ex |_{H1} = " << h1_err
            << "   || lambda - lambda_ex ||_{L2(Gamma)} = " << lambda_l2_err
            << std::endl;
  }
}

template <int dim, int spacedim>
void BabuskaProblem<dim, spacedim>::output_results() {
  TimerOutput::Scope timer_section(monitor, "Output results");

  DataOut<spacedim> bulk_out;
  bulk_out.attach_dof_handler(space_dh);
  bulk_out.add_data_vector(solution, "u");
  bulk_out.build_patches(*bulk_mapping,
                         parameters.bulk_space_finite_element_degree);
  std::ofstream bulk_file("solution_bulk.vtu");
  bulk_out.write_vtu(bulk_file);

  DataOut<dim, spacedim> surf_out;
  surf_out.attach_dof_handler(boundary_dh);
  surf_out.add_data_vector(lambda, "lambda",
                           DataOut<dim, spacedim>::type_dof_data);
  surf_out.build_patches(*surf_mapping);
  std::ofstream surf_file("multiplier.vtu");
  surf_out.write_vtu(surf_file);
}

template <int dim, int spacedim> void BabuskaProblem<dim, spacedim>::run() {
  AssertThrow(parameters.initialized, ExcNotInitialized());
  deallog.depth_console(parameters.verbosity_level);

  for (unsigned int cycle = 0; cycle < parameters.n_refinement_cycles;
       ++cycle) {
    deallog << "==== Refinement cycle " << cycle << " ====" << std::endl;
    setup_grids_and_dofs();
    setup_coupling();
    assemble_system();
    solve();
  }
  if (space_dh.n_dofs() < 1e6) // avoid outputting very large solutions
    output_results();

  // Print the iteration table to screen.
  std::cout << "\nRefinement study summary:\n";
  if (parameters.use_manufactured_solution) {
    convergence_table.set_precision("L2_u", 3);
    convergence_table.set_precision("H1_u", 3);
    convergence_table.set_precision("L2_lambda", 3);
    convergence_table.set_precision("h", 3);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.set_scientific("H1_u", true);
    convergence_table.set_scientific("L2_lambda", true);
    convergence_table.set_scientific("h", true);
    // Convergence rates are computed against h
    convergence_table.evaluate_convergence_rates(
        "L2_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
        "H1_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
        "L2_lambda", ConvergenceTable::reduction_rate_log2);
  }
  convergence_table.write_text(std::cout,
                               TableHandler::TextOutputFormat::org_mode_table);
}

} // namespace NitscheBCs

int main(int argc, char **argv) {
  try {
    using namespace dealii;
    using namespace NitscheBCs;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    const std::string parameter_file =
        (argc > 1) ? argv[1] : "parameters_nitsche.prm";

    // Detect the (dim, spacedim) pair from the parameter file by scanning
    // for the "NitscheLagrange<dim,spacedim>" subsection header. Two
    // configurations are supported: <1,2> and <2,3>.
    auto scan_prm_for_dims =
        [](const std::string &fname) -> std::pair<int, int> {
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
      const bool has_12 = c.find("NitscheLagrange<1,2>") != std::string::npos;
      const bool has_23 = c.find("NitscheLagrange<2,3>") != std::string::npos;
      AssertThrow(has_12 ^ has_23,
                  ExcMessage("The parameter file must contain exactly one of "
                             "'subsection NitscheLagrange<1,2>' or "
                             "'subsection NitscheLagrange<2,3>'."));
      if (has_12)
        return {1, 2};
      return {2, 3};
    };

    auto run = [&](auto dim_const, auto spacedim_const) {
      constexpr int dim = decltype(dim_const)::value;
      constexpr int spacedim = decltype(spacedim_const)::value;
      typename BabuskaProblem<dim, spacedim>::Parameters parameters;
      BabuskaProblem<dim, spacedim> problem(parameters);
      ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
      problem.run();
    };

    const auto dims = scan_prm_for_dims(parameter_file);
    if (dims.first == 1 && dims.second == 2)
      run(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
    else
      run(std::integral_constant<int, 2>{}, std::integral_constant<int, 3>{});
  } catch (std::exception &exc) {
    std::cerr << "\n\n----------------------------------------------------\n"
              << "Exception on processing: \n"
              << exc.what() << "\nAborting!\n"
              << "----------------------------------------------------\n";
    return 1;
  } catch (...) {
    std::cerr << "\n\nUnknown exception!\nAborting!\n";
    return 1;
  }
  return 0;
}