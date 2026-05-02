// Imposition of Dirichlet boundary conditions through a boundary-supported
// Lagrange multiplier (Nitsche-like saddle-point formulation):
//
//   ( grad u, grad v )_Omega + < lambda, v >_{partial Omega} = ( f, v )_Omega
//   < u, mu >_{partial Omega}                                = < g, mu
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

// Manufactured solution used for the optional convergence study.
template <int spacedim> class ManufacturedSolution : public Function<spacedim> {
public:
  double value(const Point<spacedim> &p,
               const unsigned int /*component*/ = 0) const override {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }

  Tensor<1, spacedim>
  gradient(const Point<spacedim> &p,
           const unsigned int /*component*/ = 0) const override {
    Tensor<1, spacedim> g;
    g[0] = numbers::PI * std::cos(numbers::PI * p[0]) *
           std::sin(numbers::PI * p[1]);
    g[1] = numbers::PI * std::sin(numbers::PI * p[0]) *
           std::cos(numbers::PI * p[1]);
    return g;
  }
};

template <int spacedim> class ManufacturedRhs : public Function<spacedim> {
public:
  double value(const Point<spacedim> &p,
               const unsigned int /*component*/ = 0) const override {
    // -Delta u + u with u = sin(pi x) sin(pi y).
    return (2. * numbers::PI * numbers::PI + 1.) *
           std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
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

template <int dim, int spacedim = dim> class NitscheLagrangeProblem {
public:
  class Parameters : public ParameterAcceptor {
  public:
    Parameters();

    unsigned int initial_refinement = 4;
    unsigned int n_refinement_cycles = 1;
    std::string name_of_grid = "hyper_cube";
    std::string arguments_for_grid = "0.0: 1.0: true";
    unsigned int bulk_space_finite_element_degree = 1;
    unsigned int multiplier_finite_element_degree = 1;
    unsigned int coupling_quadrature_order = 3;
    unsigned int verbosity_level = 4;
    bool use_discontinuous_multiplier = false;
    std::string name_external_grid = "idealized_lv.msh";
    double scale_factor = 1.0;
    // Cell type for both the bulk and boundary meshes: "hex" or "simplex".
    std::string mesh_type = "simplex";

    // If true, build the AL augmentation term gamma*(1/h)*(u,v)_{Gamma}
    // by inserting tracking particles on the boundary and looping
    // over them. If false, assemble the same term directly during the
    // boundary-face loop used for the coupling matrix C.
    bool use_particles_for_augmentation = true;

    // If true, ignore the user-provided right-hand side and Dirichlet data
    // and use the hardcoded manufactured solution  u(x,y) = sin(pi x) sin(pi y)
    // on (0,1)^2,
    bool use_manufactured_solution = false;

    bool iterative_inversion_mass_matrix = false;

    bool initialized = false;
  };

  NitscheLagrangeProblem(const Parameters &parameters);

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
NitscheLagrangeProblem<dim, spacedim>::Parameters::Parameters()
    : ParameterAcceptor("/Nitsche Lagrange<" + Utilities::int_to_string(dim) +
                        "," + Utilities::int_to_string(spacedim) + ">/") {
  add_parameter("Initial space refinement", initial_refinement);
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
      "Use particles to impose constraints", use_particles_for_augmentation,
      "If true, the AL augmentation term gamma*(1/h)*(u,v)_{Gamma} is "
      "assembled via tracking particles placed at quadrature points on the "
      "immersed boundary. If false, it is assembled directly through a "
      "boundary-face loop on the bulk space, reusing the surface-to-bulk "
      "face map already used to build the coupling matrix C.");
  add_parameter(
      "Use manufactured solution", use_manufactured_solution,
      "If true, override the right-hand side and Dirichlet data with the "
      "hardcoded manufactured solution u = sin(pi x) sin(pi y) on the unit "
      "square, and report L2/H1 errors plus convergence rates.");
  add_parameter(
      "Iterative inversion mass matrix", iterative_inversion_mass_matrix,
      "If true, invert the boundary mass matrix (used inside the Schur "
      "preconditioner) iteratively with CG + Jacobi, instead of factorizing "
      "it with UMFPACK.");

  parse_parameters_call_back.connect([&]() -> void { initialized = true; });
}

template <int dim, int spacedim>
NitscheLagrangeProblem<dim, spacedim>::NitscheLagrangeProblem(
    const Parameters &parameters)
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
void NitscheLagrangeProblem<dim, spacedim>::setup_grids_and_dofs() {
  TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

  // Rebuild the bulk grid from scratch each cycle
  static unsigned int extra_refinements = 0;
  space_grid.clear();

  if (parameters.name_of_grid == "from_file") {
    GridIn<spacedim> grid_in;
    grid_in.attach_triangulation(space_grid);
    std::ifstream input_file(parameters.name_external_grid);
    AssertThrow(input_file, ExcMessage("Could not open grid file"));
    grid_in.read_msh(input_file);
    space_grid.refine_global(extra_refinements);
    GridTools::scale(parameters.scale_factor, space_grid);
  } else {
    if (is_hex_mesh(parameters.mesh_type)) {
      GridGenerator::generate_from_name_and_arguments(
          space_grid, parameters.name_of_grid, parameters.arguments_for_grid);
      space_grid.refine_global(parameters.initial_refinement +
                               extra_refinements);
    } else {
      // convert to simplex mesh so the rest of the program can run on
      // tris/tets.
      Triangulation<spacedim> hex_grid;
      GridGenerator::generate_from_name_and_arguments(
          hex_grid, parameters.name_of_grid, parameters.arguments_for_grid);
      hex_grid.refine_global(parameters.initial_refinement + extra_refinements);
      GridGenerator::convert_hypercube_to_simplex_mesh(hex_grid, space_grid);
    }
  }
  ++extra_refinements;

  stiffness_matrix.clear();
  coupling_matrix.clear();
  boundary_mass_matrix.clear();
  face_to_bulk_cell.clear();
  surface_to_volume_map.clear();

  // Extract the boundary mesh on the bulk grid. With newer deal.II versions,
  // extract_boundary_mesh supports also simplex meshes

  boundary_grid = std::make_unique<Triangulation<dim, spacedim>>();
#if DEAL_II_VERSION_GTE(9, 9, 0)
  surface_to_volume_map =
      GridGenerator::extract_boundary_mesh(space_grid, *boundary_grid);
#else
  if (is_hex_mesh(parameters.mesh_type)) {
    surface_to_volume_map =
        GridGenerator::extract_boundary_mesh(space_grid, *boundary_grid);
  } else {
    using BulkFaceIt =
        typename Triangulation<spacedim, spacedim>::face_iterator;

    std::vector<Point<spacedim>> surf_vertices;
    std::vector<CellData<dim>> surf_cells;
    std::vector<BulkFaceIt> face_per_surf_cell;

    std::map<unsigned int, unsigned int> bulk_to_surf_vertex;
    for (const auto &bulk_cell : space_grid.active_cell_iterators())
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
void NitscheLagrangeProblem<dim, spacedim>::setup_coupling() {
  TimerOutput::Scope timer_section(monitor, "Setup coupling");

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
void NitscheLagrangeProblem<dim, spacedim>::assemble_system() {
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

    // When the augmentation is not done with particles, we add the
    // gamma*(1/h)*(u,v)_{Gamma} term directly here while looping over
    // bulk cells, by visiting their boundary faces. The constants below
    // must match those used in solve() for the Schur preconditioner.

    const bool augment_via_faces = !parameters.use_particles_for_augmentation;
    const double gamma_aug = 10.0;
    const double invW_scale_aug = augment_via_faces
                                      ? 1.0 / GridTools::maximal_cell_diameter(
                                                  *boundary_grid, *surf_mapping)
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

    ManufacturedDirichlet<spacedim> manufactured_g_aug;
    const Function<spacedim> &g_for_aug =
        parameters.use_manufactured_solution
            ? static_cast<const Function<spacedim> &>(manufactured_g_aug)
            : static_cast<const Function<spacedim> &>(g_function);

    for (const auto &cell : space_dh.active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs = 0.;

      // Pick either the user-provided RHS or the manufactured one.
      if (parameters.use_manufactured_solution) {
        ManufacturedRhs<spacedim> manufactured_rhs;
        manufactured_rhs.value_list(fe_values.get_quadrature_points(),
                                    rhs_values);
      } else {
        rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);
      }

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

  // RHS for the multiplier equation
  ManufacturedDirichlet<spacedim> manufactured_g;
  const Function<spacedim> &g_for_rhs =
      parameters.use_manufactured_solution
          ? static_cast<const Function<spacedim> &>(manufactured_g)
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

  FEFaceValues<spacedim> fe_face_values(
      *bulk_mapping, *space_fe, face_quad,
      update_values | update_quadrature_points | update_JxW_values);
  FEValues<dim, spacedim> fe_surface_values(*surf_mapping, *multiplier_fe,
                                            face_quad, update_values);

  const unsigned int n_bulk_dofs = space_fe->n_dofs_per_cell();
  const unsigned int n_surf_dofs = multiplier_fe->n_dofs_per_cell();

  FullMatrix<double> local_C(n_bulk_dofs, n_surf_dofs);
  std::vector<types::global_dof_index> bulk_dof_indices(n_bulk_dofs);
  std::vector<types::global_dof_index> surf_dof_indices(n_surf_dofs);

  for (const auto &surface_cell : boundary_dh.active_cell_iterators()) {
    const auto bulk_face = surface_to_volume_map.at(surface_cell);
    const auto &owner = face_to_bulk_cell.at(bulk_face);
    const auto &bulk_cell = owner.first;
    const unsigned int bulk_face_no = owner.second;

    fe_face_values.reinit(bulk_cell, bulk_face_no);
    fe_surface_values.reinit(surface_cell);

    bulk_cell->get_dof_indices(bulk_dof_indices);
    surface_cell->get_dof_indices(surf_dof_indices);

    local_C = 0;
    for (unsigned int q = 0; q < face_quad.size(); ++q) {
      const double JxW = fe_face_values.JxW(q);
      for (unsigned int i = 0; i < n_bulk_dofs; ++i) {
        const double v_i = fe_face_values.shape_value(i, q);
        for (unsigned int j = 0; j < n_surf_dofs; ++j) {
          const double mu_j = fe_surface_values.shape_value(j, q);
          local_C(i, j) += v_i * mu_j * JxW;
        }
      }
    }

    constraints.distribute_local_to_global(local_C, bulk_dof_indices,
                                           surf_dof_indices, coupling_matrix);
  }

// Sanity check for coupling matrix against the boundary measure obtained by
// integrating 1 with the multiplier mass matrix
#ifdef DEBUG
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
#endif
  std::cout << "Assembly complete." << std::endl;
}

template <int dim, int spacedim>
void NitscheLagrangeProblem<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(monitor, "Solve system");

  // Build the saddle-point block operator
  //   [ K   C  ] [ u      ]   [ f ]
  //   [ C^T 0  ] [ lambda ] = [ g ]
  // and solve with (preconditioned) FGMRES
  auto K = linear_operator(
      stiffness_matrix); // notice that stiffness_matrix will be augmented
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
  double gamma = 10.;
  TrilinosWrappers::PreconditionAMG amg_prec;
  auto prec_for_cg = null_operator(K);
  double invW_scale = 1.0; // = 1/h, set below once the boundary mesh size is
                           // known
  {
    const double h_immersed =
        GridTools::maximal_cell_diameter(*boundary_grid, embedded_mapping);
    invW_scale = 1.0 / h_immersed;

    if (parameters.use_particles_for_augmentation) {
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
      ManufacturedDirichlet<spacedim> manufactured_g;
      const Function<spacedim> &g_for_aug =
          parameters.use_manufactured_solution
              ? static_cast<const Function<spacedim> &>(manufactured_g)
              : static_cast<const Function<spacedim> &>(g_function);

      std::vector<types::global_dof_index> background_dof_indices(
          space_fe->n_dofs_per_cell());

      FullMatrix<double> local_matrix(space_fe->n_dofs_per_cell(),
                                      space_fe->n_dofs_per_cell());
      Vector<double> local_rhs(space_fe->n_dofs_per_cell());

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
            local_rhs(i) += gamma * invW_scale * v_i * gq * JxW;
            for (unsigned int j = 0; j < space_fe->n_dofs_per_cell(); ++j) {
              local_matrix(i, j) += gamma * invW_scale * // gamma * (1/h)
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
    }

    // When use_particles_for_augmentation is false, the augmentation has
    // already been added to stiffness_matrix during assemble_system().
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
  convergence_table.add_value("inner_cg_total", total_inner_iters);
  convergence_table.add_value("inner_cg/outer", inner_per_outer);
  convergence_table.set_precision("inner_cg/outer", 1);

  solution = solution_block.block(0);
  lambda = solution_block.block(1);
  constraints.distribute(solution);

  // If the manufactured solution is active, compute L2 / H1-seminorm errors
  // of u against u_ex on the bulk mesh
  const double h_bulk =
      GridTools::maximal_cell_diameter(space_grid, *bulk_mapping);
  convergence_table.add_value("h", h_bulk);
  if (parameters.use_manufactured_solution) {
    ManufacturedSolution<spacedim> u_ex;
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
    deallog << "|| u - u_ex ||_{L2} = " << l2_err
            << "   | u - u_ex |_{H1} = " << h1_err << std::endl;
  }
}

template <int dim, int spacedim>
void NitscheLagrangeProblem<dim, spacedim>::output_results() {
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

template <int dim, int spacedim>
void NitscheLagrangeProblem<dim, spacedim>::run() {
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
    convergence_table.set_precision("h", 3);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.set_scientific("H1_u", true);
    convergence_table.set_scientific("h", true);
    // Convergence rates are computed against h
    convergence_table.evaluate_convergence_rates(
        "L2_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
        "H1_u", ConvergenceTable::reduction_rate_log2);
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

    constexpr unsigned int dim = 1;
    constexpr unsigned int spacedim = 2;

    NitscheLagrangeProblem<dim, spacedim>::Parameters parameters;
    NitscheLagrangeProblem<dim, spacedim> problem(parameters);

    const std::string parameter_file =
        (argc > 1) ? argv[1] : "parameters_nitsche.prm";

    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
    problem.run();
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
