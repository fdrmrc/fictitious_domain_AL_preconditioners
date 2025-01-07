#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>

#include "augmented_lagrangian_preconditioner.h"
#include "rational_preconditioner.h"
#include "utilities.h"

namespace IBStokes {

using namespace dealii;

//  Struct used to store iteration counts
struct ResultsData {
  types::global_dof_index dofs_background;
  types::global_dof_index dofs_immersed;
  unsigned int outer_iterations;
};

// Struct to pipe parameters to the AL solver.
struct ALControl {
  double gamma;  // gamma parameter for the augmented lagrangian formulation
  double gamma_grad_div;        // gamma parameter for grad-div stabilization
  bool grad_div_stabilization;  // true if you want to assemble grad-div
                                // stabilization
  bool inverse_diag_square;     // true if you want to use the inverse diagonal
                                // squared (immersed)
  bool GMG_preconditioner_augmented;  // true if you want to use GMG (geometric
                                      // multigrid) preconditioner for augmented
                                      // block
  bool AMG_preconditioner_augmented;  // true if you want to build AMG
                                      // preconditioner for augmented block
  double tol_AL;
  unsigned int max_iterations_AL;
  bool log_result;

  void declare_parameters(ParameterHandler &param) {
    param.declare_entry("Gamma", "10", Patterns::Double());
    param.declare_entry("Gamma Grad-div", "10", Patterns::Double());
    param.declare_entry("Grad-div stabilization", "true", Patterns::Bool());
    param.declare_entry("Diagonal mass immersed", "true", Patterns::Bool());
    param.declare_entry("GMG for augmented block", "true", Patterns::Bool());
    param.declare_entry("AMG for augmented block", "false", Patterns::Bool());
    param.declare_entry("Tolerance for Augmented Lagrangian", "1e-4",
                        Patterns::Double());
    param.declare_entry("Max steps", "100", Patterns::Integer());
    param.declare_entry("Log result", "true", Patterns::Bool());
  }
  void parse_parameters(ParameterHandler &param) {
    gamma = param.get_double("Gamma");
    gamma_grad_div = param.get_double("Gamma Grad-div");
    grad_div_stabilization = param.get_bool("Grad-div stabilization");
    inverse_diag_square = param.get_bool("Diagonal mass immersed");
    GMG_preconditioner_augmented = param.get_bool("GMG for augmented block");
    AMG_preconditioner_augmented = param.get_bool("AMG for augmented block");
    tol_AL = param.get_double("Tolerance for Augmented Lagrangian");
    log_result = param.get_bool("Log result");
    max_iterations_AL = param.get_integer("Max steps");
  }
};

// Struct to pipe parameters for the embedded configuration. As we assume to
// work with an immersed sphere, you can select radius and coordinates.
template <int spacedim>
struct ConfigurationControl {
  double radius;  // gamma parameter for the augmented lagrangian formulation
  double x_center;
  double y_center;
  double z_center;

  void declare_parameters(ParameterHandler &param) {
    param.declare_entry("Radius", "0.21", Patterns::Double());
    param.declare_entry("x_c", "0.", Patterns::Double());
    param.declare_entry("y_c", "0.", Patterns::Double());
    if constexpr (spacedim == 3)
      param.declare_entry("z_c", "0.", Patterns::Double());
  }
  void parse_parameters(ParameterHandler &param) {
    radius = param.get_double("Radius");
    x_center = param.get_double("x_c");
    y_center = param.get_double("y_c");
    if constexpr (spacedim == 3) z_center = param.get_double("z_c");
  }
};

template <int dim, int spacedim = dim>
class IBStokesProblem {
 public:
  class Parameters : public ParameterAcceptor {
   public:
    Parameters();

    unsigned int initial_refinement = 6;

    unsigned int delta_refinement = 1;

    unsigned int initial_embedded_refinement = 5;

    std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

    unsigned int velocity_finite_element_degree =
        2;  // Pressure will be 1 (Taylor-Hood pair)

    unsigned int embedded_space_finite_element_degree = 1;  // multiplier space

    unsigned int embedded_configuration_finite_element_degree = 1;

    unsigned int coupling_quadrature_order = 3;

    unsigned int verbosity_level = 10;

    bool initialized = false;

    std::string solver = "Stokes";
  };

  ResultsData results_data;
  IBStokesProblem(const Parameters &parameters);

  void run();

  void set_filename(const std::string &filename);

 private:
  const Parameters &parameters;

  void setup_grids_and_dofs();

  void setup_background_dofs();

  void setup_embedded_dofs();

  void setup_coupling();

  void setup_multigrid();

  void assemble_stokes();

  void assemble_multigrid();

  void assemble_preconditioner();

  void solve();

  void output_results();

  std::unique_ptr<parallel::distributed::Triangulation<spacedim, spacedim>>
      space_grid;
  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_grid_tools_cache;
  std::unique_ptr<FESystem<spacedim>> velocity_fe;
  std::unique_ptr<FiniteElement<spacedim>> space_fe;
  std::unique_ptr<DoFHandler<spacedim>> space_dh;
  std::unique_ptr<DoFHandler<spacedim>> velocity_dh;

  std::unique_ptr<parallel::shared::Triangulation<dim, spacedim>> embedded_grid;
  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_dh;

  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;

  std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_value_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      dirichlet_bc_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      body_force_function;

  ParameterAcceptorProxy<ReductionControl> outer_solver_control;

  ParameterAcceptorProxy<ALControl> augmented_lagrangian_control;
  ParameterAcceptorProxy<ConfigurationControl<spacedim>>
      embedded_configuration_control;

  SparsityPattern coupling_sparsity;
  SparsityPattern coupling_sparsity_t;
  SparsityPattern mass_sparsity;

  TrilinosWrappers::SparseMatrix mass_matrix_immersed;
  TrilinosWrappers::SparseMatrix coupling_matrix;
  TrilinosWrappers::SparseMatrix coupling_matrix_t;

  // Matrix for augmented term: A + gamma C^T W^{-1} C
  TrilinosWrappers::SparseMatrix augmented_matrix;
  TrilinosWrappers::SparseMatrix diag_inverse_square_trilinos;

  TrilinosWrappers::BlockSparsityPattern sparsity_pattern_stokes;
  TrilinosWrappers::BlockSparseMatrix stokes_matrix;

  TrilinosWrappers::BlockSparsityPattern preconditioner_sparsity_pattern;
  TrilinosWrappers::BlockSparseMatrix preconditioner_matrix;

  TrilinosWrappers::SparseMatrix diag_inverse_pressure_matrix_trilinos;
  TrilinosWrappers::SparsityPattern diag_inverse_pressure_sparsity;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_velocity;

  TrilinosWrappers::MPI::BlockVector solution;
  TrilinosWrappers::MPI::BlockVector locally_relevant_solution;
  TrilinosWrappers::MPI::BlockVector stokes_rhs;

  TrilinosWrappers::MPI::Vector lambda;
  TrilinosWrappers::MPI::Vector embedded_rhs;
  TrilinosWrappers::MPI::Vector embedded_value;

  // Needed to invert pressure and multiplier mass matrices
  TrilinosWrappers::PreconditionILU Mp_inv_ilu;
  TrilinosWrappers::PreconditionILU M_immersed_inv_ilu;
  TrilinosWrappers::MPI::Vector inverse_squares_multiplier;  // M^{-2}

  std::vector<IndexSet> stokes_partitioning;
  std::vector<IndexSet> stokes_relevant_partitioning;

  std::string parameters_filename;

  IndexSet embedded_locally_owned_dofs;
  IndexSet embedded_locally_relevant_dofs;

  MPI_Comm mpi_comm;
  ConditionalOStream pcout;
  TimerOutput monitor;

  // GMG related types

  MGLevelObject<TrilinosWrappers::SparseMatrix> mg_matrix;
  MGLevelObject<TrilinosWrappers::SparseMatrix> mg_mass_matrix_immersed;
  MGLevelObject<TrilinosWrappers::SparseMatrix> mg_matrix_coupling;
  MGLevelObject<TrilinosWrappers::SparseMatrix>
      mg_matrix_coupling_t;  // For transpose
  MGLevelObject<TrilinosWrappers::MPI::Vector> mg_inverse_squares_multiplier;

  MGLevelObject<TrilinosWrappers::SparseMatrix> mg_interface_in;
  MGConstrainedDoFs mg_constrained_dofs;
};

template <int dim, int spacedim>
IBStokesProblem<dim, spacedim>::Parameters::Parameters()
    : ParameterAcceptor("/Distributed Lagrange<" +
                        Utilities::int_to_string(dim) + "," +
                        Utilities::int_to_string(spacedim) + ">/") {
  add_parameter("Initial background space refinement", initial_refinement);

  add_parameter("Initial embedded space refinement",
                initial_embedded_refinement);

  add_parameter("Local refinements steps near embedded domain",
                delta_refinement);

  add_parameter("Dirichlet boundary ids", dirichlet_ids);

  add_parameter("Velocity space finite element degree",
                velocity_finite_element_degree);

  add_parameter("Embedded space finite element degree",
                embedded_space_finite_element_degree);

  add_parameter("Coupling quadrature order", coupling_quadrature_order);

  add_parameter("Verbosity level", verbosity_level);

  add_parameter("Solver", solver);

  parse_parameters_call_back.connect([&]() -> void { initialized = true; });
}

template <int dim, int spacedim>
IBStokesProblem<dim, spacedim>::IBStokesProblem(const Parameters &parameters)
    : parameters(parameters),
      embedded_value_function("Embedded value", spacedim),
      dirichlet_bc_function("Dirichlet boundary condition", spacedim + 1),
      body_force_function("Body force", spacedim),
      outer_solver_control("Outer solver control"),
      augmented_lagrangian_control("Augmented Lagrangian control"),
      embedded_configuration_control("Embedded Configuration control"),
      mpi_comm(MPI_COMM_WORLD),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
      monitor(pcout, TimerOutput::summary, TimerOutput::wall_times) {
  embedded_value_function.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "1; 1");
  });

  dirichlet_bc_function.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "0;0;0");
  });

  body_force_function.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "0;0");
  });

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Reduction", "1.e-12");
    ParameterAcceptor::prm.set("Tolerance", "1.e-10");
  });

  augmented_lagrangian_control.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Gamma", "10");
        ParameterAcceptor::prm.set("Grad-div stabilization", "true");
        ParameterAcceptor::prm.set("Diagonal mass immersed", "true");
        ParameterAcceptor::prm.set("AMG for augmented block", "true");
        ParameterAcceptor::prm.set("Log result", "true");
        ParameterAcceptor::prm.set("Max steps", "100");
        ParameterAcceptor::prm.set("Tolerance for Augmented Lagrangian",
                                   "1.e-4");
      });

  embedded_configuration_control.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Radius", "0.21");
        ParameterAcceptor::prm.set("x_c", "0.");
        ParameterAcceptor::prm.set("y_c", "0.");
        if constexpr (spacedim == 3) ParameterAcceptor::prm.set("z_c", "0.");
      });
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::set_filename(const std::string &filename) {
  Assert(!filename.empty(), ExcMessage("Set an invalid filename"));
  parameters_filename = filename;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_grids_and_dofs() {
  TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

  space_grid = std::make_unique<parallel::distributed::Triangulation<spacedim>>(
      mpi_comm, Triangulation<spacedim>::limit_level_difference_at_vertices,
      (augmented_lagrangian_control.AMG_preconditioner_augmented == true)
          ? parallel::distributed::Triangulation<spacedim>::default_setting
          : parallel::distributed::Triangulation<
                spacedim>::construct_multigrid_hierarchy);

  GridGenerator::hyper_cube(*space_grid, 0., 1, true);

  space_grid->refine_global(parameters.initial_refinement);
  space_grid_tools_cache =
      std::make_unique<GridTools::Cache<spacedim, spacedim>>(*space_grid);

  embedded_grid =
      std::make_unique<parallel::shared::Triangulation<dim, spacedim>>(
          mpi_comm);

  Point<spacedim> center =
      (spacedim == 3)
          ? Point<spacedim>{embedded_configuration_control.x_center,
                            embedded_configuration_control.y_center,
                            embedded_configuration_control.z_center}
          : Point<spacedim>{embedded_configuration_control.x_center,
                            embedded_configuration_control.y_center};
  GridGenerator::hyper_sphere(*embedded_grid, center,
                              embedded_configuration_control.radius);
  embedded_grid->refine_global(parameters.initial_embedded_refinement - 2);
  const unsigned int n_mpi_processes =
      Utilities::MPI::n_mpi_processes(mpi_comm);
  GridTools::partition_triangulation(n_mpi_processes, *embedded_grid);

  embedded_configuration_fe = std::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(
          parameters.embedded_configuration_finite_element_degree) ^
      spacedim);

  embedded_mapping = std::make_unique<MappingQ1<dim, spacedim>>();

  setup_embedded_dofs();

  std::vector<Point<spacedim>> support_points(embedded_dh->n_dofs());
  if (parameters.delta_refinement != 0) {
    DoFTools::map_dofs_to_support_points(*embedded_mapping, *embedded_dh,
                                         support_points);

    // We now perform a localized refinement around the interface of the
    // immersed domain. In case of a distributed tria, we exchange with other
    // processes a rough description of the local portion of the domain using
    // bounding boxes
    IteratorFilters::LocallyOwnedCell locally_owned_cell_predicate;
    std::vector<BoundingBox<spacedim>> local_bbox =
        GridTools::compute_mesh_predicate_bounding_box(
            *space_grid,
            std::function<bool(const typename Triangulation<
                               spacedim>::active_cell_iterator &)>(
                locally_owned_cell_predicate),
            1, false, 4);

    // Obtaining the global mesh description through an all to all communication
    std::vector<std::vector<BoundingBox<spacedim>>> global_bboxes;
    global_bboxes = Utilities::MPI::all_gather(mpi_comm, local_bbox);

    for (unsigned int i = 0; i < parameters.delta_refinement; ++i) {
      // Notice how we call the distributed version of this function.
      const auto point_locations =
          GridTools::distributed_compute_point_locations(
              *space_grid_tools_cache, support_points, global_bboxes);
      const auto &cells = std::get<0>(point_locations);
      for (auto &cell : cells) {
        if (cell->is_locally_owned()) {
          cell->set_refine_flag();
          for (const auto face_no : cell->face_indices())
            if (!cell->at_boundary(face_no))
              cell->neighbor(face_no)->set_refine_flag();
        }
      }
      space_grid->execute_coarsening_and_refinement();
    }
  }

  const double embedded_space_maximal_diameter =
      GridTools::maximal_cell_diameter(*embedded_grid, *embedded_mapping);
  double background_space_minimal_diameter =
      GridTools::minimal_cell_diameter(*space_grid);

  pcout << "Background minimal diameter: " << background_space_minimal_diameter
        << ", embedded maximal diameter: " << embedded_space_maximal_diameter
        << ", ratio: "
        << embedded_space_maximal_diameter / background_space_minimal_diameter
        << std::endl;

  if constexpr (spacedim == 2)
    AssertThrow(
        embedded_space_maximal_diameter < background_space_minimal_diameter,
        ExcMessage("The background grid is too refined (or the embedded grid "
                   "is too coarse). Adjust the parameters so that the minimal"
                   "grid size of the background grid is larger "
                   "than the maximal grid size of the embedded grid."));

  setup_background_dofs();
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_background_dofs() {
  // Define background FE
  space_dh = std::make_unique<DoFHandler<spacedim>>(*space_grid);
  velocity_dh = std::make_unique<DoFHandler<spacedim>>(*space_grid);
  velocity_fe = std::make_unique<FESystem<spacedim>>(
      FE_Q<spacedim>(parameters.velocity_finite_element_degree) ^ spacedim);

  space_fe = std::make_unique<FESystem<spacedim>>(
      FE_Q<spacedim>(parameters.velocity_finite_element_degree) ^ spacedim,
      FE_Q<spacedim>(parameters.velocity_finite_element_degree - 1));

  space_dh->distribute_dofs(*space_fe);
  velocity_dh->distribute_dofs(*velocity_fe);

  // Do DoFRenumbering only in serial
  if (Utilities::MPI::n_mpi_processes(mpi_comm) == 1) {
    DoFRenumbering::Cuthill_McKee(*space_dh);
    DoFRenumbering::Cuthill_McKee(
        *velocity_dh);  // we need to renumber in the same way we renumbered
                        // DoFs for velocity
  }

  std::vector<unsigned int> block_component(spacedim + 1, 0);
  block_component[spacedim] = 1;
  DoFRenumbering::component_wise(*space_dh, block_component);

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(*space_dh, block_component);
  const types::global_dof_index n_u = dofs_per_block[0];
  const types::global_dof_index n_p = dofs_per_block[1];
  pcout << "Number of degrees of freedom: " << space_dh->n_dofs() << " (" << n_u
        << '+' << n_p << ')' << std::endl;

  const IndexSet &stokes_locally_owned_index_set =
      space_dh->locally_owned_dofs();
  const IndexSet stokes_locally_relevant_set =
      DoFTools::extract_locally_relevant_dofs(*space_dh);

  {
    constraints.clear();
    constraints.reinit(stokes_locally_owned_index_set,
                       stokes_locally_relevant_set);

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(*space_dh, constraints);
    for (const unsigned int id : parameters.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
          *space_dh, id, dirichlet_bc_function, constraints,
          space_fe->component_mask(velocities));
  }
  constraints.close();

  const IndexSet &velocity_locally_owned_index_set =
      velocity_dh->locally_owned_dofs();
  const IndexSet velocity_locally_relevant_set =
      DoFTools::extract_locally_relevant_dofs(*velocity_dh);
  {
    constraints_velocity.clear();
    constraints_velocity.reinit(velocity_locally_owned_index_set,
                                velocity_locally_relevant_set);

    DoFTools::make_hanging_node_constraints(*velocity_dh, constraints_velocity);
    for (const unsigned int id : parameters.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
          *velocity_dh, id, Functions::ZeroFunction<spacedim>(spacedim),
          constraints_velocity);
  }
  constraints_velocity.close();

  stokes_partitioning.push_back(
      stokes_locally_owned_index_set.get_view(0, n_u));
  stokes_partitioning.push_back(
      stokes_locally_owned_index_set.get_view(n_u, n_u + n_p));

  stokes_relevant_partitioning.push_back(
      stokes_locally_relevant_set.get_view(0, n_u));
  stokes_relevant_partitioning.push_back(
      stokes_locally_relevant_set.get_view(n_u, n_u + n_p));

  // Define blocksparsityPattern

  {
    BlockDynamicSparsityPattern dsp_stokes(dofs_per_block, dofs_per_block);
    Table<2, DoFTools::Coupling> coupling_table(spacedim + 1, spacedim + 1);
    for (unsigned int c = 0; c < spacedim + 1; ++c)
      for (unsigned int d = 0; d < spacedim + 1; ++d)
        if (!((c == spacedim) && (d == spacedim)))
          coupling_table[c][d] = DoFTools::always;
        else
          coupling_table[c][d] = DoFTools::none;

    sparsity_pattern_stokes.reinit(stokes_partitioning, stokes_partitioning,
                                   stokes_relevant_partitioning, mpi_comm);

    DoFTools::make_sparsity_pattern(
        *space_dh, coupling_table, sparsity_pattern_stokes, constraints, false,
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    sparsity_pattern_stokes.compress();
  }

  {
    Table<2, DoFTools::Coupling> preconditioner_coupling(spacedim + 1,
                                                         spacedim + 1);
    for (unsigned int c = 0; c < spacedim + 1; ++c)
      for (unsigned int d = 0; d < spacedim + 1; ++d)
        if (((c == spacedim) && (d == spacedim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;

    preconditioner_sparsity_pattern.reinit(
        stokes_partitioning, stokes_partitioning, stokes_relevant_partitioning,
        mpi_comm);

    DoFTools::make_sparsity_pattern(
        *space_dh, preconditioner_coupling, preconditioner_sparsity_pattern,
        constraints, false, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    preconditioner_sparsity_pattern.compress();
  }

  // Initialize matrices
  stokes_matrix.reinit(sparsity_pattern_stokes);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

  TrilinosWrappers::SparsityPattern mass_dsp(embedded_locally_owned_dofs,
                                             mpi_comm);
  DoFTools::make_sparsity_pattern(*embedded_dh, mass_dsp);
  mass_dsp.compress();
  mass_matrix_immersed.reinit(mass_dsp);  // M_immersed

  // Initialize vectors
  solution.reinit(stokes_partitioning, mpi_comm);
  locally_relevant_solution.reinit(stokes_partitioning,
                                   stokes_relevant_partitioning, mpi_comm);
  stokes_rhs.reinit(stokes_partitioning, stokes_relevant_partitioning, mpi_comm,
                    true);
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_embedded_dofs() {
  embedded_dh = std::make_unique<DoFHandler<dim, spacedim>>(*embedded_grid);

  if (parameters.embedded_space_finite_element_degree > 0) {
    embedded_fe = std::make_unique<FESystem<dim, spacedim>>(
        FE_Q<dim, spacedim>(parameters.embedded_space_finite_element_degree) ^
        spacedim);
  } else if (parameters.embedded_space_finite_element_degree == 0) {
    // otherwise, DG(0) elements for the multiplier
    embedded_fe = std::make_unique<FESystem<dim, spacedim>>(
        FE_DGQ<dim, spacedim>(parameters.embedded_space_finite_element_degree) ^
        spacedim);
  } else {
    AssertThrow(false, ExcNotImplemented());
  }
  embedded_dh->distribute_dofs(*embedded_fe);

  embedded_locally_owned_dofs = embedded_dh->locally_owned_dofs();
  embedded_locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(*embedded_dh);

  lambda.reinit(embedded_locally_owned_dofs, mpi_comm);
  embedded_rhs.reinit(embedded_locally_owned_dofs, mpi_comm);
  embedded_value.reinit(embedded_locally_owned_dofs, mpi_comm);

  pcout << "Embedded dofs: " << embedded_dh->n_dofs() << std::endl;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_coupling() {
  TimerOutput::Scope timer_section(monitor, "Setup coupling");

  const QGauss<dim> quad(parameters.coupling_quadrature_order);

  TrilinosWrappers::SparsityPattern dsp(velocity_dh->locally_owned_dofs(),
                                        embedded_dh->locally_owned_dofs(),
                                        mpi_comm);

  TrilinosWrappers::SparsityPattern dsp_t(embedded_dh->locally_owned_dofs(),
                                          velocity_dh->locally_owned_dofs(),
                                          mpi_comm);
  // Here, we use velocity_dh: we want to couple DoF for velocity with the
  // ones of the multiplier.
  UtilitiesAL::create_coupling_sparsity_patterns(
      *space_grid_tools_cache, *velocity_dh, *embedded_dh, quad, dsp_t, dsp,
      constraints, ComponentMask(), ComponentMask(), *embedded_mapping,
      AffineConstraints<double>());
  dsp.compress();
  dsp_t.compress();
  pcout << "Sparsity coupling: done" << std::endl;
  coupling_matrix.reinit(dsp);
  coupling_matrix_t.reinit(dsp_t);
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_multigrid() {
  Assert((augmented_lagrangian_control.GMG_preconditioner_augmented == true),
         ExcMessage(
             "You should not assemble level matrices if GMG was disabled."));

  TimerOutput::Scope timing(monitor, "Setup multigrid");

  velocity_dh->distribute_mg_dofs();

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(*velocity_dh);
  mg_constrained_dofs.make_zero_boundary_constraints(
      *velocity_dh, {0, 1, 2, 3});  // TODO: take from parameters file

  const unsigned int n_levels = space_grid->n_global_levels();

  mg_matrix.resize(0, n_levels - 1);
  mg_matrix.clear_elements();
  mg_interface_in.resize(0, n_levels - 1);
  mg_interface_in.clear_elements();
  mg_matrix_coupling.resize(0, n_levels - 1);  // For coupling matrices
  mg_matrix_coupling.clear_elements();
  mg_mass_matrix_immersed.resize(0, n_levels - 1);  // Multiplier mass matrix
  mg_mass_matrix_immersed.clear_elements();

  for (unsigned int level = 0; level < n_levels; ++level) {
    const IndexSet dof_set =
        DoFTools::extract_locally_relevant_level_dofs(*velocity_dh, level);

    {
      TrilinosWrappers::SparsityPattern dsp(
          velocity_dh->locally_owned_mg_dofs(level),
          velocity_dh->locally_owned_mg_dofs(level), dof_set, mpi_comm);
      MGTools::make_sparsity_pattern(*velocity_dh, dsp, level);

      dsp.compress();
      mg_matrix[level].reinit(dsp);
    }

    {
      TrilinosWrappers::SparsityPattern dsp(
          velocity_dh->locally_owned_mg_dofs(level),
          velocity_dh->locally_owned_mg_dofs(level), dof_set, mpi_comm);

      MGTools::make_interface_sparsity_pattern(*velocity_dh,
                                               mg_constrained_dofs, dsp, level);
      dsp.compress();
      mg_interface_in[level].reinit(dsp);
    }

    // Coupling terms
    const QGauss<dim> quad(parameters.coupling_quadrature_order);

    TrilinosWrappers::SparsityPattern dsp(
        velocity_dh->locally_owned_mg_dofs(level),
        embedded_dh->locally_owned_dofs(), mpi_comm);

    TrilinosWrappers::SparsityPattern dsp_t(
        embedded_dh->locally_owned_dofs(),
        velocity_dh->locally_owned_mg_dofs(level), mpi_comm);

    // We couple DoF for velocity with the ones of the multiplier **on each
    // level**
    UtilitiesAL::create_coupling_sparsity_patterns(
        *space_grid_tools_cache, *velocity_dh, *embedded_dh, quad, dsp_t, dsp,
        constraints, ComponentMask(), ComponentMask(), *embedded_mapping,
        AffineConstraints<double>());
    dsp.compress();
    dsp_t.compress();
    pcout << "Sparsity coupling on level " << level << ": done" << std::endl;
    mg_matrix_coupling[level].reinit(dsp);
    mg_matrix_coupling_t[level].reinit(dsp_t);
  }
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::assemble_multigrid() {
  Assert((augmented_lagrangian_control.GMG_preconditioner_augmented == true),
         ExcMessage(
             "You should not assemble level matrices if GMG was disabled."));
  TimerOutput::Scope timing(monitor, "Assemble multigrid");

  // Assemble coupling matrices on each multigrid level

  const unsigned int n_levels = space_grid->n_global_levels();
  const QGauss<dim> quad(parameters.coupling_quadrature_order);
  std::vector<AffineConstraints<double>> boundary_constraints(n_levels);

  for (unsigned int level = 0; level < n_levels; ++level) {
    boundary_constraints[level].reinit(
        velocity_dh->locally_owned_mg_dofs(level),
        DoFTools::extract_locally_relevant_level_dofs(*velocity_dh, level));

    // Assemble C and Ct simultaneously
    UtilitiesAL::create_coupling_mass_matrices(
        *space_grid_tools_cache, *velocity_dh, *embedded_dh, quad,
        mg_matrix_coupling_t[level], mg_matrix_coupling[level],
        boundary_constraints[level], ComponentMask(), ComponentMask(),
        *embedded_mapping, AffineConstraints<double>());
    mg_matrix_coupling[level].compress(VectorOperation::add);
    mg_matrix_coupling_t[level].compress(VectorOperation::add);
  }

  // Assemble the augmented product on each level
  UtilitiesAL::assemble_multilevel_matrices(
      *velocity_dh, mg_matrix_coupling, mg_matrix_coupling_t,
      inverse_squares_multiplier, mg_constrained_dofs, mg_matrix,
      mg_interface_in, augmented_lagrangian_control.gamma);
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::assemble_stokes() {
  stokes_matrix = 0;
  stokes_rhs = 0;
  preconditioner_matrix = 0;
  const QGauss<spacedim> quadrature_formula(space_fe->degree + 2);

  FEValues<spacedim> fe_values(*space_fe, quadrature_formula,
                               update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = space_fe->n_dofs_per_cell();

  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs(dofs_per_cell);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(spacedim);

  std::vector<Vector<double>> body_force_values(n_q_points,
                                                Vector<double>(spacedim));

  // Precompute stuff for Stokes' weak form
  std::vector<SymmetricTensor<2, spacedim>> symgrad_phi_u(dofs_per_cell);
  std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, spacedim>> phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);

  for (const auto &cell : space_dh->active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      local_matrix = 0;
      local_rhs = 0;
      local_preconditioner_matrix = 0;

      body_force_function.vector_value_list(fe_values.get_quadrature_points(),
                                            body_force_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        Tensor<1, spacedim> body_force_values_tensor{ArrayView{
            body_force_values[q].begin(), body_force_values[q].size()}};

        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
          grad_phi_u[k] = fe_values[velocities].gradient(k, q);
          div_phi_u[k] = fe_values[velocities].divergence(k, q);
          phi_u[k] = fe_values[velocities].value(k, q);
          phi_p[k] = fe_values[pressure].value(k, q);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j <= i; ++j) {
            if (augmented_lagrangian_control.grad_div_stabilization == true) {
              local_matrix(i, j) +=
                  (1. * scalar_product(grad_phi_u[i],
                                       grad_phi_u[j])  // symgrad-symgrad
                   - div_phi_u[i] * phi_p[j]           // div u_i p_j
                   - phi_p[i] * div_phi_u[j]           // p_i div u_j
                   +
                   augmented_lagrangian_control.gamma_grad_div * div_phi_u[i] *
                       div_phi_u[j]) *  // grad-div stabilization
                  fe_values.JxW(q);
            } else {
              // no grad-div stabilization, usual formulation
              local_matrix(i, j) +=
                  (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])  // symgrad-symgrad
                   - div_phi_u[i] * phi_p[j]                  // div u_i p_j
                   - phi_p[i] * div_phi_u[j]) *               // p_i div u_j
                  fe_values.JxW(q);
            }

            local_preconditioner_matrix(i, j) +=
                (phi_p[i] * phi_p[j]) * fe_values.JxW(q);  // p_i p_j
          }

          // local_rhs(i) += phi_u[i] * rhs_values[q] * fe_values.JxW(q);
          local_rhs(i) +=
              phi_u[i] * body_force_values_tensor * fe_values.JxW(q);
        }
      }

      // exploit symmetry
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j) {
          local_matrix(i, j) = local_matrix(j, i);
          local_preconditioner_matrix(i, j) = local_preconditioner_matrix(j, i);
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_matrix, local_rhs,
                                             local_dof_indices, stokes_matrix,
                                             stokes_rhs);

      constraints.distribute_local_to_global(local_preconditioner_matrix,
                                             local_dof_indices,
                                             preconditioner_matrix);
    }
  }
  stokes_matrix.compress(VectorOperation::add);
  preconditioner_matrix.compress(VectorOperation::add);
  pcout << "Assembled Stokes terms" << std::endl;

  {
    TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

    const QGauss<dim> quad(parameters.coupling_quadrature_order);

    // Assemble C and Ct simultaneously
    UtilitiesAL::create_coupling_mass_matrices(
        *space_grid_tools_cache, *velocity_dh, *embedded_dh, quad,
        coupling_matrix_t, coupling_matrix, constraints, ComponentMask(),
        ComponentMask(), *embedded_mapping, AffineConstraints<double>());
    coupling_matrix.compress(VectorOperation::add);
    coupling_matrix_t.compress(VectorOperation::add);

    MatrixTools::create_mass_matrix(*embedded_mapping, *embedded_dh,
                                    QGauss<dim>(2 * embedded_fe->degree + 1),
                                    mass_matrix_immersed);

    VectorTools::create_right_hand_side(
        *embedded_mapping, *embedded_dh,
        QGauss<dim>(2 * embedded_fe->degree + 2), embedded_value_function,
        embedded_rhs);

    VectorTools::interpolate(*embedded_mapping, *embedded_dh,
                             embedded_value_function, embedded_value);
  }
  pcout << "A dimensions (" << stokes_matrix.block(0, 0).m() << ","
        << stokes_matrix.block(0, 0).n() << ")" << std::endl;
  pcout << "B dimensions (" << stokes_matrix.block(1, 0).m() << ","
        << stokes_matrix.block(1, 0).n() << ")" << std::endl;
  pcout << "C dimensions (" << coupling_matrix.n() << "," << coupling_matrix.m()
        << ")" << std::endl;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::assemble_preconditioner() {
  TimerOutput::Scope timer_section(monitor, "Assemble preconditioner terms");

  {
    // Inverse of pressure mass matrix using the diagonal only. If this option
    // is not selected, we will an ILU decomposition.
    if (augmented_lagrangian_control.inverse_diag_square) {
      diag_inverse_pressure_sparsity.reinit(stokes_partitioning[1], mpi_comm);

      for (auto dof : stokes_partitioning[1]) {
        diag_inverse_pressure_sparsity.add(dof,
                                           dof);  // Add diagonal entry only
      }
      diag_inverse_pressure_sparsity.compress();

      diag_inverse_pressure_matrix_trilinos.reinit(
          diag_inverse_pressure_sparsity);

      SolverControl solver_control(100, 1e-16, false, false);
      SolverCG<TrilinosWrappers::MPI::Vector> cg_solver(solver_control);

      if (augmented_lagrangian_control.inverse_diag_square) {
        TrilinosWrappers::MPI::Vector pressure_diagonal_inv;
        pressure_diagonal_inv.reinit(stokes_partitioning[1], mpi_comm);
        // for (types::global_dof_index i = 0; i < n_cols_Mp; ++i)
        for (const types::global_dof_index local_idx : stokes_partitioning[1])
          pressure_diagonal_inv(local_idx) =
              1. / preconditioner_matrix.block(1, 1).diag_element(local_idx);

        pressure_diagonal_inv.compress(VectorOperation::insert);

        // diag_inverse_pressure_matrix.reinit(pressure_diagonal_inv);

        for (auto row : stokes_partitioning[1]) {
          diag_inverse_pressure_matrix_trilinos.set(row, row,
                                                    pressure_diagonal_inv[row]);
        }
        diag_inverse_pressure_matrix_trilinos.compress(VectorOperation::insert);

      } else {
        Mp_inv_ilu.initialize(preconditioner_matrix.block(1, 1));
      }
      pcout << "Defined inverse of pressure mass matrix" << std::endl;
    }
  }

  // Multiplier mass matrix
  {
    TrilinosWrappers::MPI::Vector inverse_squares;
    TrilinosWrappers::SparsityPattern diag_inverse_square_sparsity(
        embedded_locally_owned_dofs, mpi_comm);
    for (auto dof : embedded_locally_owned_dofs) {
      diag_inverse_square_sparsity.add(dof, dof);  // Add diagonal entry only
    }
    diag_inverse_square_sparsity.compress();
    diag_inverse_square_trilinos.reinit(diag_inverse_square_sparsity);

    inverse_squares.reinit(embedded_locally_owned_dofs, mpi_comm);
    for (const types::global_dof_index local_idx : embedded_locally_owned_dofs)
      inverse_squares(local_idx) =
          1. / (mass_matrix_immersed.diag_element(local_idx) *
                mass_matrix_immersed.diag_element(local_idx));

    inverse_squares.compress(VectorOperation::insert);

    for (auto row : embedded_locally_owned_dofs) {
      diag_inverse_square_trilinos.set(row, row, inverse_squares[row]);
    }
    diag_inverse_square_trilinos.compress(VectorOperation::insert);
  }

  // Augmented Lagrangian term
  {
    inverse_squares_multiplier.reinit(embedded_locally_owned_dofs, mpi_comm);

    // for (types::global_dof_index i = 0; i < mass_matrix_immersed.m(); ++i)
    for (const types::global_dof_index local_idx : embedded_locally_owned_dofs)
      inverse_squares_multiplier(local_idx) =
          1. / (mass_matrix_immersed.diag_element(local_idx) *
                mass_matrix_immersed.diag_element(local_idx));

    inverse_squares_multiplier.compress(VectorOperation::insert);

    if (augmented_lagrangian_control.AMG_preconditioner_augmented == true) {
      pcout << "Computing augmented block for AMG..." << std::endl;
      UtilitiesAL::create_augmented_block(
          *velocity_dh, coupling_matrix_t, coupling_matrix,
          inverse_squares_multiplier, constraints_velocity,
          augmented_lagrangian_control.gamma, augmented_matrix);
      pcout << "Assembled augmented block for AMG." << std::endl;
    }
  }
}

void output_double_number(double input, const std::string &text) {
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << text << input << std::endl;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(monitor, "Solve system");

  using PayloadType = dealii::TrilinosWrappers::internal::
      LinearOperatorImplementation::TrilinosPayload;
  if (std::strcmp(parameters.solver.c_str(), "IBStokes") == 0) {
    // Immersed boundary, **without preconditioner**
    // Extract blocks from Stokes

    auto A = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(0, 0));
    auto Bt = linear_operator<TrilinosWrappers::MPI::Vector,
                              TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(0, 1));
    auto B = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(1, 0));
    auto Ct = linear_operator<TrilinosWrappers::MPI::Vector,
                              TrilinosWrappers::MPI::Vector, PayloadType>(
        coupling_matrix);
    auto C = transpose_operator(Ct);

    // SparseDirectUMFPACK A_inv_umfpack;
    // A_inv_umfpack.initialize(stokes_matrix.block(0, 0));
    TrilinosWrappers::PreconditionILU A_inv_direct;
    A_inv_direct.initialize(stokes_matrix.block(0, 0));
    auto A_inv = linear_operator<TrilinosWrappers::MPI::Vector,
                                 TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(0, 0), A_inv_direct);

    // Define inverse operators

    SolverControl solver_control(100 * solution.block(1).size(), 1e-10, false,
                                 false);
    SolverCG<TrilinosWrappers::MPI::Vector> cg_solver(solver_control);
    auto SBB = B * A_inv * Bt;
    auto SBC = B * A_inv * Ct;
    auto SCB = C * A_inv * Bt;
    auto SCC = C * A_inv * Ct;

    TrilinosWrappers::PreconditionIdentity prec_id;
    auto SBB_inv = inverse_operator(SBB, cg_solver, prec_id);
    auto S_lambda = SCC - SCB * SBB_inv * SBC;
    auto S_lambda_inv = inverse_operator(S_lambda, cg_solver, prec_id);

    auto A_inv_f = A_inv * stokes_rhs.block(0);
    lambda = S_lambda_inv *
             (C * A_inv_f - embedded_rhs - SCB * SBB_inv * B * A_inv_f);
    pcout << "Computed multiplier" << std::endl;

    auto &p = solution.block(1);
    p = SBB_inv * (B * A_inv_f - SBC * lambda);
    constraints.distribute(solution);
    pcout << "Computed pressure" << std::endl;
    auto &u = solution.block(0);
    u = A_inv * (stokes_rhs.block(0) - Bt * solution.block(1) - Ct * lambda);
    constraints.distribute(solution);
    pcout << "Computed velocity" << std::endl;
  } else if (std::strcmp(parameters.solver.c_str(), "IBStokesAL") == 0) {
    // Immersed boundary, with Augmented Lagrangian preconditioner

    // As before, extract blocks from Stokes
    auto A = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(0, 0));
    auto Bt = linear_operator<TrilinosWrappers::MPI::Vector,
                              TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(0, 1));
    auto B = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector, PayloadType>(
        stokes_matrix.block(1, 0));

    auto Ct = linear_operator<TrilinosWrappers::MPI::Vector,
                              TrilinosWrappers::MPI::Vector, PayloadType>(
        coupling_matrix);
    auto C = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector, PayloadType>(
        coupling_matrix_t);

    auto M = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector, PayloadType>(
        mass_matrix_immersed);
    const auto Zero = M * 0.0;
    auto Mp = linear_operator<TrilinosWrappers::MPI::Vector,
                              TrilinosWrappers::MPI::Vector, PayloadType>(
        preconditioner_matrix.block(1, 1));
    auto Mp_inv = null_operator(Mp);

    // Solver for the inversion of the pressure mass matrix.
    SolverControl solver_control(100, 1e-16, false, false);
    SolverCG<TrilinosWrappers::MPI::Vector> cg_solver(solver_control);
    // Inverse of the pressure mass matrix
    if (augmented_lagrangian_control.inverse_diag_square) {
      Mp_inv = linear_operator<TrilinosWrappers::MPI::Vector,
                               TrilinosWrappers::MPI::Vector>(
          diag_inverse_pressure_matrix_trilinos);
    } else {
      Mp_inv_ilu.initialize(preconditioner_matrix.block(1, 1));
      Mp_inv = inverse_operator(Mp, cg_solver, Mp_inv_ilu);
    }

    // Inverse of the multiplier mass matrix
    M_immersed_inv_ilu.initialize(mass_matrix_immersed);
    auto invW = null_operator(M);
    auto invW1 = inverse_operator(M, cg_solver, M_immersed_inv_ilu);
    if (augmented_lagrangian_control.inverse_diag_square)
      invW = linear_operator<TrilinosWrappers::MPI::Vector,
                             TrilinosWrappers::MPI::Vector>(
          diag_inverse_square_trilinos);
    else
      invW = invW1 * invW1;

    // Next, we define the augmented block. If we selected grad-div
    // stabilization, we will use the augmented matrix we have built in
    // assemble_preconditioner(). Otherwise, we will add the term "Ct * invW *
    // C" as an operator, but this prevents the setup of a preconditioner.
    const double gamma = augmented_lagrangian_control.gamma;
    const double gamma_grad_div = augmented_lagrangian_control.gamma_grad_div;
    pcout << "gamma (Grad-div): " << gamma_grad_div << std::endl;
    pcout << "gamma (AL): " << gamma << std::endl;
    auto Aug = null_operator(A);
    if (augmented_lagrangian_control.grad_div_stabilization)
      Aug = linear_operator<TrilinosWrappers::MPI::Vector,
                            TrilinosWrappers::MPI::Vector, PayloadType>(
          augmented_matrix);
    else
      Aug = A + gamma * Ct * invW * C;  // no preconditioner will be employed.

    // We define the block operator in the standard way...
    auto AA = block_operator<3, 3, TrilinosWrappers::MPI::BlockVector>(
        {{{{Aug, Bt, Ct}},
          {{B, Zero, Zero}},
          {{C, Zero, Zero}}}});  //! Augmented the (1,1) block

    //... and the layout solution vector.
    TrilinosWrappers::MPI::BlockVector solution_block;
    TrilinosWrappers::MPI::BlockVector system_rhs_block;
    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);
    solution_block.block(0) = solution.block(0);  // velocity
    solution_block.block(1) = solution.block(1);  // pressure
    solution_block.block(2) = lambda;             // multiplier

    // Add Lagrangian term to rhs:
    TrilinosWrappers::MPI::Vector tmp;
    tmp.reinit(stokes_partitioning[0], stokes_relevant_partitioning[0],
               mpi_comm);
    tmp = gamma * Ct * invW * embedded_rhs;
    // We now define the augmented rhs of the system.
    system_rhs_block.block(0) = stokes_rhs.block(0);
    system_rhs_block.block(0).add(1., tmp);  // ! augmented
    system_rhs_block.block(1) = stokes_rhs.block(1);
    system_rhs_block.block(2) = embedded_rhs;

    // Next, we define the parameters for the inner solve associated to the
    // augmented block.
    SolverControl control_lagrangian(
        augmented_lagrangian_control.max_iterations_AL,
        augmented_lagrangian_control.tol_AL, false,
        augmented_lagrangian_control.log_result);
    SolverCG<TrilinosWrappers::MPI::Vector> solver_lagrangian(
        control_lagrangian);

    // Depending on the parameters file, we will initialize an AMG or GMG
    // preconditioner, or nothing.
    auto Aug_inv = null_operator(A);
    TrilinosWrappers::PreconditionAMG prec_amg_aug;
    TrilinosWrappers::PreconditionIdentity prec_id;
    std::unique_ptr<
        PreconditionMG<spacedim, TrilinosWrappers::MPI::Vector,
                       MGTransferPrebuilt<TrilinosWrappers::MPI::Vector>>>
        gmg_preconditioner;
    if (augmented_lagrangian_control.AMG_preconditioner_augmented == true &&
        augmented_lagrangian_control.grad_div_stabilization == true) {
      const FEValuesExtractors::Vector velocity_components(0);
      const std::vector<std::vector<bool>> constant_modes =
          DoFTools::extract_constant_modes(
              *space_dh,
              space_dh->get_fe().component_mask(velocity_components));

      pcout << "Initializing AMG preconditioner..." << std::endl;
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.constant_modes = constant_modes;
      amg_data.elliptic = true;
      amg_data.higher_order_elements = true;
      amg_data.smoother_sweeps = 2;
      amg_data.aggregation_threshold = 0.02;

      prec_amg_aug.initialize(augmented_matrix,
                              amg_data);  //! actually fill the preconditioner

      Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_amg_aug);
      pcout << "Initialized AMG preconditioner" << std::endl;
    } else if (augmented_lagrangian_control.AMG_preconditioner_augmented ==
                   false &&
               augmented_lagrangian_control.GMG_preconditioner_augmented ==
                   true) {
      // Define GMG preconditioner

      MGTransferPrebuilt<TrilinosWrappers::MPI::Vector> mg_transfer(
          mg_constrained_dofs);
      mg_transfer.build(*velocity_dh);

      // Control for the coarse solver
      SolverControl coarse_solver_control(1000, 1e-12, false, false);
      SolverCG<TrilinosWrappers::MPI::Vector> coarse_solver(
          coarse_solver_control);
      PreconditionIdentity identity;  // no preconditioner for the coarse solver
      MGCoarseGridIterativeSolver<TrilinosWrappers::MPI::Vector,
                                  SolverCG<TrilinosWrappers::MPI::Vector>,
                                  TrilinosWrappers::SparseMatrix,
                                  PreconditionIdentity>
          coarse_grid_solver(coarse_solver, mg_matrix[0], identity);

      using Smoother = TrilinosWrappers::PreconditionJacobi;
      MGSmootherPrecondition<TrilinosWrappers::SparseMatrix, Smoother,
                             TrilinosWrappers::MPI::Vector>
          smoother;

      smoother.initialize(mg_matrix, 1.0);  // damping factor = 1
      smoother.set_steps(1);                // smoothing steps = 1

      mg::Matrix<TrilinosWrappers::MPI::Vector> mg_m(mg_matrix);
      mg::Matrix<TrilinosWrappers::MPI::Vector> mg_in(mg_interface_in);
      mg::Matrix<TrilinosWrappers::MPI::Vector> mg_out(mg_interface_in);

      Multigrid<TrilinosWrappers::MPI::Vector> mg(
          mg_m, coarse_grid_solver, mg_transfer, smoother, smoother);
      mg.set_edge_matrices(mg_out, mg_in);

      gmg_preconditioner = std::make_unique<
          PreconditionMG<spacedim, TrilinosWrappers::MPI::Vector,
                         MGTransferPrebuilt<TrilinosWrappers::MPI::Vector>>>(
          *velocity_dh, mg,
          mg_transfer);  // create the preconditioner object...

      // and use it to invert the augmented block
      Aug_inv = inverse_operator(Aug, solver_lagrangian, *gmg_preconditioner);
    } else if (augmented_lagrangian_control.AMG_preconditioner_augmented ==
                   false &&
               augmented_lagrangian_control.GMG_preconditioner_augmented ==
                   false &&
               augmented_lagrangian_control.grad_div_stabilization == false) {
      // No preconditioner and no grad-div
      Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_id);
    } else {
      AssertThrow(false, ExcNotImplemented());
    }

    // The next class takes the operators and computes the action of the
    // preconditioner.
    BlockPreconditionerAugmentedLagrangianStokes<
        TrilinosWrappers::MPI::Vector, TrilinosWrappers::MPI::BlockVector>
        augmented_lagrangian_preconditioner_Stokes{
            Aug_inv, Bt, Ct, invW, Mp_inv, gamma, gamma_grad_div};

    // Finally, solve the resulting system
    SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver_fgmres(
        outer_solver_control);
    solver_fgmres.solve(AA, solution_block, system_rhs_block,
                        augmented_lagrangian_preconditioner_Stokes);

    solution.block(0) = solution_block.block(0);
    solution.block(1) = solution_block.block(1);
    constraints.distribute(solution);
    locally_relevant_solution = solution;

  } else {
    AssertThrow(false, ExcNotImplemented());
  }

  // Store iteration counts and DoF
  results_data.dofs_background = space_dh->n_dofs();
  results_data.dofs_immersed = embedded_dh->n_dofs();
  results_data.outer_iterations = outer_solver_control.last_step();
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::output_results() {
  TimerOutput::Scope timer_section(monitor, "Output results");

  {
    DataOut<dim, spacedim> embedded_out;

    std::ofstream embedded_out_file("embedded.vtu");

    std::vector<std::string> solution_names(spacedim, "g");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            spacedim, DataComponentInterpretation::component_is_part_of_vector);

    embedded_out.attach_dof_handler(*embedded_dh);
    const auto dg_or_not = parameters.embedded_space_finite_element_degree == 0
                               ? DataOut<dim, spacedim>::type_cell_data
                               : DataOut<dim, spacedim>::type_automatic;
    embedded_out.add_data_vector(lambda, "lambda", dg_or_not);
    embedded_out.add_data_vector(embedded_value, solution_names, dg_or_not,
                                 data_component_interpretation);
    embedded_out.build_patches(*embedded_mapping, 1.);
    embedded_out.write_vtu(embedded_out_file);
  }

  {
    std::vector<std::string> solution_names(spacedim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            spacedim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    DataOut<spacedim> data_out_stokes;
    data_out_stokes.attach_dof_handler(*space_dh);
    data_out_stokes.add_data_vector(locally_relevant_solution, solution_names,
                                    DataOut<spacedim>::type_dof_data,
                                    data_component_interpretation);

    Vector<float> subdomain(space_grid->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = space_grid->locally_owned_subdomain();
    data_out_stokes.add_data_vector(subdomain, "subdomain");

    data_out_stokes.build_patches();

    // std::ofstream output("solution-stokes.vtk");
    // data_out_stokes.write_vtk(output);
    data_out_stokes.write_vtu_with_pvtu_record(
        "", "solution-stokes", 1, mpi_comm, numbers::invalid_unsigned_int, 0);
  }

  // Estimate condition number:
  pcout << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
  pcout << "Estimate condition number of CCt using CG" << std::endl;
  SolverControl solver_control(lambda.size(), 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_control);

  solver_cg.connect_condition_number_slot(
      std::bind(output_double_number, std::placeholders::_1,
                "Condition number estimate: "));
  using PayloadType = dealii::TrilinosWrappers::internal::
      LinearOperatorImplementation::TrilinosPayload;
  auto Ct = linear_operator<TrilinosWrappers::MPI::Vector,
                            TrilinosWrappers::MPI::Vector, PayloadType>(
      coupling_matrix);
  auto C = transpose_operator(Ct);

  auto CCt = C * Ct;

  TrilinosWrappers::MPI::Vector u(lambda);
  u = 0.;
  TrilinosWrappers::MPI::Vector f(lambda);
  f = 1.;
  TrilinosWrappers::PreconditionIdentity prec_no;
  try {
    solver_cg.solve(CCt, u, f, prec_no);
  } catch (...) {
    pcout << "***CCt solve not successfull (see condition number above)***"
          << std::endl;
  }
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::run() {
  AssertThrow(parameters.initialized, ExcNotInitialized());

  pcout << "Running with Trilinos on "
        << Utilities::MPI::n_mpi_processes(mpi_comm) << " MPI rank(s)..."
        << std::endl;

  setup_grids_and_dofs();
  setup_coupling();
  if (augmented_lagrangian_control.GMG_preconditioner_augmented)
    setup_multigrid();
  assemble_stokes();
  if (std::strcmp(parameters.solver.c_str(), "IBStokesAL") == 0)
    assemble_preconditioner();
  if (augmented_lagrangian_control.GMG_preconditioner_augmented)
    assemble_multigrid();
  solve();
  output_results();
}
}  // namespace IBStokes

int main(int argc, char **argv) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    using namespace dealii;
    using namespace IBStokes;

    const unsigned int dim = 1, spacedim = 2;

    IBStokesProblem<dim, spacedim>::Parameters parameters;
    IBStokesProblem<dim, spacedim> problem(parameters);

    mpi_initlog(true, parameters.verbosity_level);
    std::string parameter_file;
    if (argc > 1)
      parameter_file = argv[1];
    else
      parameter_file = "parameters.prm";

    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
    problem.set_filename(parameter_file);
    problem.run();
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