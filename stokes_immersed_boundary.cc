#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
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
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>

#include "augmented_lagrangian_preconditioner.h"
#include "rational_preconditioner.h"
#include "utilities.h"

#ifdef DEAL_II_WITH_TRILINOS
#include <Epetra_CrsMatrix.h>
#include <Epetra_RowMatrixTransposer.h>
#endif

namespace IBStokes {

using namespace dealii;

template <int dim>
struct InnerPreconditioner;

template <>
struct InnerPreconditioner<2> {
  using type = SparseDirectUMFPACK;
};

template <>
struct InnerPreconditioner<3> {
  using type = SparseILU<double>;
};

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor {
 public:
  InverseMatrix(const MatrixType &m, const PreconditionerType &preconditioner);

  void vmult(Vector<double> &dst, const Vector<double> &src) const;

 private:
  const SmartPointer<const MatrixType> matrix;
  const SmartPointer<const PreconditionerType> preconditioner;
};

template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &m, const PreconditionerType &preconditioner)
    : matrix(&m), preconditioner(&preconditioner) {}

template <class MatrixType, class PreconditionerType>
void InverseMatrix<MatrixType, PreconditionerType>::vmult(
    Vector<double> &dst, const Vector<double> &src) const {
  SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  dst = 0;

  cg.solve(*matrix, dst, src, *preconditioner);
}

template <class PreconditionerType>
class SchurComplement : public Subscriptor {
 public:
  SchurComplement(
      const BlockSparseMatrix<double> &system_matrix,
      const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);

  void vmult(Vector<double> &dst, const Vector<double> &src) const;

 private:
  const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
  const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
      A_inverse;

  mutable Vector<double> tmp1, tmp2;
};

template <class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix),
      A_inverse(&A_inverse),
      tmp1(system_matrix.block(0, 0).m()),
      tmp2(system_matrix.block(0, 0).m()) {}

template <class PreconditionerType>
void SchurComplement<PreconditionerType>::vmult(
    Vector<double> &dst, const Vector<double> &src) const {
  system_matrix->block(0, 1).vmult(tmp1, src);
  A_inverse->vmult(tmp2, tmp1);
  system_matrix->block(1, 0).vmult(dst, tmp2);
}

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
    param.declare_entry("AMG for augmented block", "true", Patterns::Bool());
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
    AMG_preconditioner_augmented = param.get_bool("AMG for augmented block");
    tol_AL = param.get_double("Tolerance for Augmented Lagrangian");
    log_result = param.get_bool("Log result");
    max_iterations_AL = param.get_integer("Max steps");
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

  void assemble_system();

  void assemble_stokes();

  void solve();

  void output_results();

  void export_results_to_csv_file();

  std::unique_ptr<Triangulation<spacedim>> space_grid;
  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_grid_tools_cache;
  std::unique_ptr<FESystem<spacedim>> velocity_fe;
  std::unique_ptr<FiniteElement<spacedim>> space_fe;
  std::unique_ptr<DoFHandler<spacedim>> space_dh;
  std::unique_ptr<DoFHandler<spacedim>> velocity_dh;

  std::unique_ptr<Triangulation<dim, spacedim>> embedded_grid;
  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_dh;

  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_configuration_dh;
  Vector<double> embedded_configuration;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_configuration_function;

  std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_value_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      dirichlet_bc_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      body_force_function;

  ParameterAcceptorProxy<ReductionControl> outer_solver_control;

  ParameterAcceptorProxy<ALControl> augmented_lagrangian_control;

  SparsityPattern coupling_sparsity;
  SparsityPattern mass_sparsity;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> mass_matrix_immersed;

  SparseMatrix<double> embedded_stiffness_matrix;
  SparseMatrix<double> coupling_matrix;

  BlockSparsityPattern sparsity_pattern_stokes;
  BlockSparseMatrix<double> stokes_matrix;

  BlockSparsityPattern preconditioner_sparsity_pattern;
  BlockSparseMatrix<double> preconditioner_matrix;

  AffineConstraints<double> constraints;

  BlockVector<double> solution;
  BlockVector<double> stokes_rhs;

  Vector<double> lambda;
  Vector<double> embedded_rhs;
  Vector<double> embedded_value;

  TimerOutput monitor;

  std::string parameters_filename;

  std::shared_ptr<typename InnerPreconditioner<spacedim>::type>
      A_preconditioner;
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

  add_parameter("Embedded configuration finite element degree",
                embedded_configuration_finite_element_degree);

  add_parameter("Coupling quadrature order", coupling_quadrature_order);

  add_parameter("Verbosity level", verbosity_level);

  add_parameter("Solver", solver);

  parse_parameters_call_back.connect([&]() -> void { initialized = true; });
}

template <int dim, int spacedim>
IBStokesProblem<dim, spacedim>::IBStokesProblem(const Parameters &parameters)
    : parameters(parameters),
      embedded_configuration_function("Embedded configuration", spacedim),
      embedded_value_function("Embedded value", spacedim),
      dirichlet_bc_function("Dirichlet boundary condition", spacedim + 1),
      body_force_function("Body force", spacedim),
      outer_solver_control("Outer solver control"),
      augmented_lagrangian_control("Augmented Lagrangian control"),
      monitor(std::cout, TimerOutput::summary,
              TimerOutput::cpu_and_wall_times) {
  embedded_configuration_function.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Function constants", "R=.21, Cx=.5,Cy=.5");

        ParameterAcceptor::prm.set("Function expression",
                                   "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      });

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
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::set_filename(const std::string &filename) {
  Assert(!filename.empty(), ExcMessage("Set an invalid filename"));
  parameters_filename = filename;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_grids_and_dofs() {
  TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

  space_grid = std::make_unique<Triangulation<spacedim>>();

  GridGenerator::hyper_cube(*space_grid, 0., 1, true);

  space_grid->refine_global(parameters.initial_refinement);
  space_grid_tools_cache =
      std::make_unique<GridTools::Cache<spacedim, spacedim>>(*space_grid);

  embedded_grid = std::make_unique<Triangulation<dim, spacedim>>();
  GridGenerator::hyper_cube(*embedded_grid);
  embedded_grid->refine_global(parameters.initial_embedded_refinement);

  embedded_configuration_fe = std::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(
          parameters.embedded_configuration_finite_element_degree) ^
      spacedim);

  embedded_configuration_dh =
      std::make_unique<DoFHandler<dim, spacedim>>(*embedded_grid);

  embedded_configuration_dh->distribute_dofs(*embedded_configuration_fe);
  embedded_configuration.reinit(embedded_configuration_dh->n_dofs());

  VectorTools::interpolate(*embedded_configuration_dh,
                           embedded_configuration_function,
                           embedded_configuration);

  embedded_mapping =
      std::make_unique<MappingFEField<dim, spacedim, Vector<double>>>(
          *embedded_configuration_dh, embedded_configuration);

  setup_embedded_dofs();

  std::vector<Point<spacedim>> support_points(embedded_dh->n_dofs());
  if (parameters.delta_refinement != 0)
    DoFTools::map_dofs_to_support_points(*embedded_mapping, *embedded_dh,
                                         support_points);

  for (unsigned int i = 0; i < parameters.delta_refinement; ++i) {
    const auto point_locations = GridTools::compute_point_locations(
        *space_grid_tools_cache, support_points);
    const auto &cells = std::get<0>(point_locations);
    for (auto &cell : cells) {
      cell->set_refine_flag();
      for (const auto face_no : cell->face_indices())
        if (!cell->at_boundary(face_no))
          cell->neighbor(face_no)->set_refine_flag();
    }
    space_grid->execute_coarsening_and_refinement();
  }

  const double embedded_space_maximal_diameter =
      GridTools::maximal_cell_diameter(*embedded_grid, *embedded_mapping);
  double background_space_minimal_diameter =
      GridTools::minimal_cell_diameter(*space_grid);

  deallog << "Background minimal diameter: "
          << background_space_minimal_diameter
          << ", embedded maximal diameter: " << embedded_space_maximal_diameter
          << ", ratio: "
          << embedded_space_maximal_diameter / background_space_minimal_diameter
          << std::endl;

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
  DoFRenumbering::Cuthill_McKee(*space_dh);
  DoFRenumbering::Cuthill_McKee(
      *velocity_dh);  // we need to renumber in the same way we renumbered DoFs
                      // for velocity

  A_preconditioner.reset();
  std::vector<unsigned int> block_component(spacedim + 1, 0);
  block_component[spacedim] = 1;
  DoFRenumbering::component_wise(*space_dh, block_component);

  {
    constraints.clear();

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(*space_dh, constraints);
    for (const unsigned int id : parameters.dirichlet_ids)
      VectorTools::interpolate_boundary_values(
          *space_dh, id, dirichlet_bc_function, constraints,
          space_fe->component_mask(velocities));
  }
  constraints.close();

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(*space_dh, block_component);
  const types::global_dof_index n_u = dofs_per_block[0];
  const types::global_dof_index n_p = dofs_per_block[1];
  deallog << "Number of degrees of freedom: " << space_dh->n_dofs() << " ("
          << n_u << '+' << n_p << ')' << std::endl;

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

    std::vector<types::global_dof_index> dofs_immersed(
        embedded_dh->get_fe().n_dofs_per_cell());
    for (const auto &immersed_cell : embedded_dh->active_cell_iterators()) {
      immersed_cell->get_dof_indices(dofs_immersed);
      for (const types::global_dof_index idx_row : dofs_immersed)
        for (const types::global_dof_index idx_col : dofs_immersed)
          dsp_stokes.block(0, 0).add(idx_row, idx_col);
    }

    DoFTools::make_sparsity_pattern(*space_dh, coupling_table, dsp_stokes,
                                    constraints, false);

    sparsity_pattern_stokes.copy_from(dsp_stokes);
  }

  {
    BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                   dofs_per_block);

    Table<2, DoFTools::Coupling> preconditioner_coupling(spacedim + 1,
                                                         spacedim + 1);
    for (unsigned int c = 0; c < spacedim + 1; ++c)
      for (unsigned int d = 0; d < spacedim + 1; ++d)
        if (((c == spacedim) && (d == spacedim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(*space_dh, preconditioner_coupling,
                                    preconditioner_dsp, constraints, false);

    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
  }

  stokes_matrix.reinit(sparsity_pattern_stokes);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

  // Initialize matrices
  stokes_matrix.reinit(sparsity_pattern_stokes);

  DynamicSparsityPattern mass_dsp(embedded_dh->n_dofs(), embedded_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*embedded_dh, mass_dsp);
  mass_sparsity.copy_from(mass_dsp);
  mass_matrix_immersed.reinit(mass_sparsity);  // M_immersed

  // Initialize vectors
  solution.reinit(dofs_per_block);
  stokes_rhs.reinit(dofs_per_block);
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

  lambda.reinit(embedded_dh->n_dofs());
  embedded_rhs.reinit(embedded_dh->n_dofs());
  embedded_value.reinit(embedded_dh->n_dofs());

  deallog << "Embedded dofs: " << embedded_dh->n_dofs() << std::endl;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_coupling() {
  TimerOutput::Scope timer_section(monitor, "Setup coupling");

  // const QGauss<dim> quad(parameters.coupling_quadrature_order);
  const QGauss<dim> quad(2 * embedded_fe->degree + 2);

  DynamicSparsityPattern dsp(velocity_dh->n_dofs(), embedded_dh->n_dofs());

  // Here, we use velocity_dh: we want to couple DoF for velocity with the
  // ones of the multiplier.
  NonMatching::create_coupling_sparsity_pattern(
      *velocity_dh, *embedded_dh, quad, dsp, constraints, ComponentMask(),
      ComponentMask(), MappingQ1<spacedim>(), *embedded_mapping);
  coupling_sparsity.copy_from(dsp);
  coupling_matrix.reinit(coupling_sparsity);
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
    fe_values.reinit(cell);
    local_matrix = 0;
    local_rhs = 0;
    local_preconditioner_matrix = 0;

    body_force_function.vector_value_list(fe_values.get_quadrature_points(),
                                          body_force_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      Tensor<1, spacedim> body_force_values_tensor{
          ArrayView{body_force_values[q].begin(), body_force_values[q].size()}};

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
                 + augmented_lagrangian_control.gamma_grad_div * div_phi_u[i] *
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
        local_rhs(i) += phi_u[i] * body_force_values_tensor * fe_values.JxW(q);
      }
    }

    // exploit symmetry
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j)

      {
        local_matrix(i, j) = local_matrix(j, i);
        local_preconditioner_matrix(i, j) = local_preconditioner_matrix(j, i);
      }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        local_matrix, local_rhs, local_dof_indices, stokes_matrix, stokes_rhs);

    constraints.distribute_local_to_global(
        local_preconditioner_matrix, local_dof_indices, preconditioner_matrix);
  }

  if (parameters.solver == "Stokes") {
    deallog << "Computing preconditioner ..." << std::endl;
    A_preconditioner =
        std::make_shared<typename InnerPreconditioner<spacedim>::type>();
    A_preconditioner->initialize(
        stokes_matrix.block(0, 0),
        typename InnerPreconditioner<spacedim>::type::AdditionalData());
  }

  {
    TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

    // const QGauss<dim> quad(parameters.coupling_quadrature_order);
    const QGauss<dim> quad(2 * embedded_fe->degree + 2);
    NonMatching::create_coupling_mass_matrix(
        *velocity_dh, *embedded_dh, quad, coupling_matrix, constraints,
        ComponentMask(), ComponentMask(), MappingQ1<spacedim>(),
        *embedded_mapping);

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
  deallog << "A dimensions (" << stokes_matrix.block(0, 0).m() << ","
          << stokes_matrix.block(0, 0).n() << ")" << std::endl;
  deallog << "B dimensions (" << stokes_matrix.block(1, 0).m() << ","
          << stokes_matrix.block(1, 0).n() << ")" << std::endl;
  deallog << "C dimensions (" << coupling_matrix.n() << ","
          << coupling_matrix.m() << ")" << std::endl;
}

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(monitor, "Solve system");

  // Stokes Only
  if (std::strcmp(parameters.solver.c_str(), "Stokes") == 0) {
    {
      const InverseMatrix<SparseMatrix<double>,
                          typename InnerPreconditioner<spacedim>::type>
          A_inverse(stokes_matrix.block(0, 0), *A_preconditioner);

      Vector<double> tmp(solution.block(0).size());

      {
        Vector<double> schur_rhs(solution.block(1).size());
        A_inverse.vmult(tmp, stokes_rhs.block(0));
        stokes_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= stokes_rhs.block(1);

        SchurComplement<typename InnerPreconditioner<spacedim>::type>
            schur_complement(stokes_matrix, A_inverse);

        SolverControl solver_control(solution.block(1).size(),
                                     1e-6 * schur_rhs.l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        SparseILU<double> preconditioner;
        preconditioner.initialize(preconditioner_matrix.block(1, 1),
                                  SparseILU<double>::AdditionalData());

        InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
            preconditioner_matrix.block(1, 1), preconditioner);

        cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

        constraints.distribute(solution);

        deallog << "  " << solver_control.last_step()
                << " outer CG Schur complement iterations for pressure"
                << std::endl;
      }

      {
        stokes_matrix.block(0, 1).vmult(tmp, solution.block(1));
        tmp *= -1;
        tmp += stokes_rhs.block(0);

        A_inverse.vmult(solution.block(0), tmp);

        constraints.distribute(solution);
      }
    }
  } else if (std::strcmp(parameters.solver.c_str(), "IBStokes") == 0) {
    // Immersed boundary, **without preconditioner**
    // Extract blocks from Stokes
    auto A = linear_operator(stokes_matrix.block(0, 0));
    auto Bt = linear_operator(stokes_matrix.block(0, 1));
    auto B = linear_operator(stokes_matrix.block(1, 0));
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);

    SparseDirectUMFPACK A_inv_umfpack;
    A_inv_umfpack.initialize(stokes_matrix.block(0, 0));
    auto A_inv = linear_operator(stokes_matrix.block(0, 0), A_inv_umfpack);

    // Define inverse operators

    SolverControl solver_control(100 * solution.block(1).size(), 1e-10, false,
                                 false);
    SolverCG<Vector<double>> cg_solver(solver_control);
    auto SBB = B * A_inv * Bt;
    auto SBC = B * A_inv * Ct;
    auto SCB = C * A_inv * Bt;
    auto SCC = C * A_inv * Ct;

    auto SBB_inv = inverse_operator(SBB, cg_solver, PreconditionIdentity());
    auto S_lambda = SCC - SCB * SBB_inv * SBC;
    auto S_lambda_inv =
        inverse_operator(S_lambda, cg_solver, PreconditionIdentity());

    auto A_inv_f = A_inv * stokes_rhs.block(0);
    lambda = S_lambda_inv *
             (C * A_inv_f - embedded_rhs - SCB * SBB_inv * B * A_inv_f);
    deallog << "Computed multiplier" << std::endl;

    auto &p = solution.block(1);
    p = SBB_inv * (B * A_inv_f - SBC * lambda);
    constraints.distribute(solution);
    deallog << "Computed pressure" << std::endl;
    auto &u = solution.block(0);
    u = A_inv * (stokes_rhs.block(0) - Bt * solution.block(1) - Ct * lambda);
    constraints.distribute(solution);
    deallog << "Computed velocity" << std::endl;
  } else if (std::strcmp(parameters.solver.c_str(), "IBStokesAL") == 0) {
    // Immersed boundary, with Augmented Lagrangian preconditioner

    // As before, extract blocks from Stokes

    auto A = linear_operator(stokes_matrix.block(0, 0));
    auto Bt = linear_operator(stokes_matrix.block(0, 1));
    auto B = linear_operator(stokes_matrix.block(1, 0));
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix_immersed);
    auto Mp = linear_operator(preconditioner_matrix.block(1, 1));

    auto Mp_inv = null_operator(Mp);
    DiagonalMatrix<Vector<double>> diag_inverse_pressure_matrix;
    SparseDirectUMFPACK Mp_inv_umfpack;
    if (augmented_lagrangian_control.inverse_diag_square) {
      const unsigned int n_cols_Mp = preconditioner_matrix.block(1, 1).m();
      Vector<double> pressure_diagonal_inv(n_cols_Mp);
      for (types::global_dof_index i = 0; i < n_cols_Mp; ++i)
        pressure_diagonal_inv(i) =
            1. / preconditioner_matrix.block(1, 1).diag_element(i);
      diag_inverse_pressure_matrix.reinit(pressure_diagonal_inv);
      Mp_inv = linear_operator(diag_inverse_pressure_matrix);
    } else {
      Mp_inv_umfpack.initialize(preconditioner_matrix.block(1, 1));
      Mp_inv =
          linear_operator(preconditioner_matrix.block(1, 1), Mp_inv_umfpack);
    }

    const auto Zero = M * 0.0;
    SparseDirectUMFPACK M_immersed_inv_umfpack;
    M_immersed_inv_umfpack.initialize(mass_matrix_immersed);

    auto invW = null_operator(M);
    Vector<double> inverse_squares(mass_matrix_immersed.m());
    for (types::global_dof_index i = 0; i < mass_matrix_immersed.m(); ++i)
      inverse_squares(i) = 1. / (mass_matrix_immersed.diag_element(i) *
                                 mass_matrix_immersed.diag_element(i));

    DiagonalMatrix<Vector<double>> diag_inverse_square(inverse_squares);
    auto invW1 = linear_operator(mass_matrix_immersed, M_immersed_inv_umfpack);
    if (augmented_lagrangian_control.inverse_diag_square)
      invW = linear_operator(diag_inverse_square);
    else
      invW = invW1 * invW1;

    const double gamma = augmented_lagrangian_control.gamma;
    const double gamma_grad_div = augmented_lagrangian_control.gamma_grad_div;
    deallog << "gamma (Grad-div): " << gamma_grad_div << std::endl;
    deallog << "gamma (AL): " << gamma << std::endl;
    auto Aug = null_operator(A);
    if (augmented_lagrangian_control.grad_div_stabilization)
      Aug = A + gamma * Ct * invW * C;
    else
      Aug = A + gamma * Ct * invW * C + gamma_grad_div * Bt * Mp_inv * B;

    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    auto AA = block_operator<3, 3, BlockVector<double>>(
        {{{{Aug, Bt, Ct}},
          {{B, Zero, Zero}},
          {{C, Zero, Zero}}}});  //! Augmented the (1,1) block
    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution.block(0);  // velocity
    solution_block.block(1) = solution.block(1);  // pressure
    solution_block.block(2) = lambda;

    // lagrangian term
    Vector<double> tmp;
    tmp.reinit(stokes_rhs.block(0).size());
    tmp = gamma * Ct * invW * embedded_rhs;
    system_rhs_block.block(0) = stokes_rhs.block(0);
    system_rhs_block.block(0).add(1., tmp);  // ! augmented
    system_rhs_block.block(1) = stokes_rhs.block(1);
    system_rhs_block.block(2) = embedded_rhs;

    SolverControl control_lagrangian(
        augmented_lagrangian_control.max_iterations_AL,
        augmented_lagrangian_control.tol_AL, false,
        augmented_lagrangian_control.log_result);
    SolverCG<Vector<double>> solver_lagrangian(control_lagrangian);

    auto Aug_inv = null_operator(A);
    TrilinosWrappers::PreconditionAMG
        prec_amg_aug;  // will be initialized only if selected
    if (augmented_lagrangian_control.AMG_preconditioner_augmented == true &&
        augmented_lagrangian_control.grad_div_stabilization == true) {
      Vector<double> inverse_squares_multiplier(
          mass_matrix_immersed.m());  // M^{-2}

      for (types::global_dof_index i = 0; i < mass_matrix_immersed.m(); ++i)
        inverse_squares_multiplier(i) =
            1. / (mass_matrix_immersed.diag_element(i) *
                  mass_matrix_immersed.diag_element(i));

      build_AMG_augmented_block(*velocity_dh, *space_dh, coupling_matrix,
                                stokes_matrix.block(0, 0), coupling_sparsity,
                                inverse_squares_multiplier, constraints, gamma,
                                prec_amg_aug);
      // prec_amg_aug.initialize(stokes_matrix.block(0, 0));
      Aug_inv = inverse_operator(Aug, solver_lagrangian, prec_amg_aug);
    } else if (augmented_lagrangian_control.AMG_preconditioner_augmented ==
                   false &&
               augmented_lagrangian_control.grad_div_stabilization == false) {
      // No preconditioner and no grad-div
      Aug_inv =
          inverse_operator(Aug, solver_lagrangian, PreconditionIdentity());
    } else {
      AssertThrow(false, ExcNotImplemented());
    }

    SolverFGMRES<BlockVector<double>> solver_fgmres(outer_solver_control);

    BlockPreconditionerAugmentedLagrangianStokes
        augmented_lagrangian_preconditioner_Stokes{
            Aug_inv, Bt, Ct, invW, Mp_inv, gamma, gamma_grad_div};
    solver_fgmres.solve(AA, solution_block, system_rhs_block,
                        augmented_lagrangian_preconditioner_Stokes);

    solution.block(0) = solution_block.block(0);
    solution.block(1) = solution_block.block(1);
    constraints.distribute(solution);
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
    data_out_stokes.add_data_vector(solution, solution_names,
                                    DataOut<spacedim>::type_dof_data,
                                    data_component_interpretation);
    data_out_stokes.build_patches();

    std::ofstream output("solution-stokes.vtk");
    data_out_stokes.write_vtk(output);
  }

  // Estimate condition number:
  deallog << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
  deallog << "Estimate condition number of CCt using CG" << std::endl;
  SolverControl solver_control(lambda.size(), 1e-12);
  SolverCG<Vector<double>> solver_cg(solver_control);

  solver_cg.connect_condition_number_slot(
      std::bind(output_double_number, std::placeholders::_1,
                "Condition number estimate: "));
  auto Ct = linear_operator(coupling_matrix);
  auto C = transpose_operator(Ct);
  auto CCt = C * Ct;

  Vector<double> u(lambda);
  u = 0.;
  Vector<double> f(lambda);
  f = 1.;
  PreconditionIdentity prec_no;
  try {
    solver_cg.solve(CCt, u, f, prec_no);
  } catch (...) {
    std::cerr << "***CCt solve not successfull (see condition number above)***"
              << std::endl;
  }
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::export_results_to_csv_file() {
  std::ofstream myfile;

  AssertThrow(!parameters_filename.empty(),
              ExcMessage("You must set the name of the parameter file."));
  std::filesystem::path p(parameters_filename);
  myfile.open(p.stem().string() + ".csv",
              std::ios::app);  // get the filename and add proper extension

  myfile << results_data.dofs_background << "," << results_data.dofs_immersed
         << "," << results_data.outer_iterations << "\n";

  myfile.close();
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::run() {
  AssertThrow(parameters.initialized, ExcNotInitialized());
  deallog.depth_console(parameters.verbosity_level);

  setup_grids_and_dofs();
  setup_coupling();
  assemble_stokes();
  solve();
  output_results();
  // export_results_to_csv_file();
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