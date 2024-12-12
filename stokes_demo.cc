#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
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

// boundary values for the velocity
template <int spacedim>
class BoundaryValues : public Function<spacedim> {
 public:
  BoundaryValues() : Function<spacedim>(spacedim + 1) {}

  virtual double value(const Point<spacedim> &p,
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<spacedim> &p,
                            Vector<double> &value) const override;
};

template <int spacedim>
double BoundaryValues<spacedim>::value(const Point<spacedim> &p,
                                       const unsigned int component) const {
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));

  if (component == 0) {
    if (p[0] < 0)
      return -1;
    else if (p[0] > 0)
      return 1;
    else
      return 0;
  }

  return 0;
}

template <int spacedim>
void BoundaryValues<spacedim>::vector_value(const Point<spacedim> &p,
                                            Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues<spacedim>::value(p, c);
}

template <int spacedim>
class RightHandSide : public TensorFunction<1, spacedim> {
 public:
  RightHandSide() : TensorFunction<1, spacedim>() {}

  virtual Tensor<1, spacedim> value(const Point<spacedim> &p) const override;

  virtual void value_list(
      const std::vector<Point<spacedim>> &p,
      std::vector<Tensor<1, spacedim>> &value) const override;
};

template <int spacedim>
Tensor<1, spacedim> RightHandSide<spacedim>::value(
    const Point<spacedim> & /*p*/) const {
  return Tensor<1, spacedim>();
}

template <int spacedim>
void RightHandSide<spacedim>::value_list(
    const std::vector<Point<spacedim>> &vp,
    std::vector<Tensor<1, spacedim>> &values) const {
  for (unsigned int c = 0; c < vp.size(); ++c) {
    values[c] = RightHandSide<spacedim>::value(vp[c]);
  }
}

namespace Stokes {

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
class InverseMatrix : public EnableObserverPointer {
 public:
  InverseMatrix(const MatrixType &m, const PreconditionerType &preconditioner);

  void vmult(Vector<double> &dst, const Vector<double> &src) const;

 private:
  const ObserverPointer<const MatrixType> matrix;
  const ObserverPointer<const PreconditionerType> preconditioner;
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
class SchurComplement : public EnableObserverPointer {
 public:
  SchurComplement(
      const BlockSparseMatrix<double> &system_matrix,
      const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);

  void vmult(Vector<double> &dst, const Vector<double> &src) const;

 private:
  const ObserverPointer<const BlockSparseMatrix<double>> system_matrix;
  const ObserverPointer<
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

template <int dim, int spacedim = dim>
class IBStokesProblem {
 public:
  class Parameters : public ParameterAcceptor {
   public:
    Parameters();

    unsigned int initial_refinement = 4;

    unsigned int delta_refinement = 3;

    unsigned int initial_embedded_refinement = 8;

    std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

    unsigned int embedding_space_finite_element_degree = 1;

    unsigned int embedded_space_finite_element_degree = 1;

    unsigned int embedded_configuration_finite_element_degree = 1;

    unsigned int coupling_quadrature_order = 3;

    bool use_displacement = false;

    unsigned int verbosity_level = 10;

    bool initialized = false;

    std::string solver = "CG";
  };

  ResultsData results_data;
  IBStokesProblem(const Parameters &parameters);

  void run();

  void set_filename(const std::string &filename);

 private:
  const Parameters &parameters;

  void setup_grids_and_dofs();

  void setup_embedding_dofs();

  void setup_embedded_dofs();

  void setup_coupling();

  void assemble_system();

  void assemble_stokes();

  void solve();

  void output_results();

  void export_results_to_csv_file();

  std::unique_ptr<Triangulation<spacedim>> space_grid;
  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_grid_tools_cache;
  std::unique_ptr<FiniteElement<spacedim>> space_fe;
  std::unique_ptr<DoFHandler<spacedim>> space_dh;

  std::unique_ptr<Triangulation<dim, spacedim>> embedded_grid;
  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_dh;

  std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_configuration_dh;
  Vector<double> embedded_configuration;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_configuration_function;

  std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

  //   ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
  //       embedding_rhs_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_value_function;

  ParameterAcceptorProxy<ReductionControl> schur_solver_control;

  SparsityPattern coupling_sparsity;

  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> stiffness_matrix_copy;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> Mass_matrix;
  SparseMatrix<double> mass_matrix_immersed_dg;
  SparseMatrix<double> embedded_stiffness_matrix;
  SparseMatrix<double> coupling_matrix;

  BlockSparsityPattern sparsity_pattern_stokes;
  BlockSparseMatrix<double> stokes_matrix;

  BlockSparsityPattern preconditioner_sparsity_pattern;
  BlockSparseMatrix<double> preconditioner_matrix;

  AffineConstraints<double> constraints;

  BlockVector<double> solution;
  BlockVector<double> stokes_rhs;
  Vector<double> embedding_rhs;
  Vector<double> embedding_rhs_copy;

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
  add_parameter("Initial embedding space refinement", initial_refinement);

  add_parameter("Initial embedded space refinement",
                initial_embedded_refinement);

  add_parameter("Local refinements steps near embedded domain",
                delta_refinement);

  add_parameter("Dirichlet boundary ids", dirichlet_ids);

  add_parameter("Use displacement in embedded interface", use_displacement);

  add_parameter("Embedding space finite element degree",
                embedding_space_finite_element_degree);

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
      //   embedding_rhs_function("Embedding rhs function (body force)",
      //   spacedim),
      embedded_value_function("Embedded value", spacedim),
      schur_solver_control("Schur solver control"),
      monitor(std::cout, TimerOutput::summary,
              TimerOutput::cpu_and_wall_times) {
  embedded_configuration_function.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4,Cy=.4");

        ParameterAcceptor::prm.set("Function expression",
                                   "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      });

  //   embedding_rhs_function.declare_parameters_call_back.connect([]() -> void
  //   {
  //     ParameterAcceptor::prm.set("Function expression", "0; 0");
  //   });

  embedded_value_function.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Function expression", "1; 1");
  });

  schur_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Reduction", "1.e-12");
    ParameterAcceptor::prm.set("Tolerance", "1.e-12");
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

  //   GridGenerator::hyper_cube(*space_grid, 0., 1, true);

  // TODO: remove
  {
    std::vector<unsigned int> subdivisions(spacedim, 1);
    subdivisions[0] = 4;

    const Point<spacedim> bottom_left =
        (spacedim == 2 ? Point<spacedim>(-2, -1) :  // 2d case
             Point<spacedim>(-2, 0, -1));           // 3d case

    const Point<spacedim> top_right =
        (spacedim == 2 ? Point<spacedim>(2, 0) :  // 2d case
             Point<spacedim>(2, 1, 0));           // 3d case

    GridGenerator::subdivided_hyper_rectangle(*space_grid, subdivisions,
                                              bottom_left, top_right);
  }

  for (const auto &cell : space_grid->active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->center()[spacedim - 1] == 0) face->set_all_boundary_ids(1);

  space_grid->refine_global(parameters.initial_refinement);
  space_grid_tools_cache =
      std::make_unique<GridTools::Cache<spacedim, spacedim>>(*space_grid);

  std::ofstream out_ext("grid-ext.gnuplot");
  GridOut grid_out_ext;
  grid_out_ext.write_gnuplot(*space_grid, out_ext);
  out_ext.close();
  std::cout << "External Grid written to grid-ext.gnuplot" << std::endl;

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

  if (parameters.use_displacement == true)
    embedded_mapping =
        std::make_unique<MappingQEulerian<dim, Vector<double>, spacedim>>(
            parameters.embedded_configuration_finite_element_degree,
            *embedded_configuration_dh, embedded_configuration);
  else
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

  if (space_grid->n_cells() < 2e6) {  // do not dump grid when mesh is too fine
    std::ofstream out_refined("grid-refined.gnuplot");
    GridOut grid_out_refined;
    grid_out_refined.write_gnuplot(*space_grid, out_refined);
    out_refined.close();
    std::cout << "Refined Grid written to grid-refined.gnuplot" << std::endl;
  }

  const double embedded_space_maximal_diameter =
      GridTools::maximal_cell_diameter(*embedded_grid, *embedded_mapping);
  double embedding_space_minimal_diameter =
      GridTools::minimal_cell_diameter(*space_grid);

  deallog << "Embedding minimal diameter: " << embedding_space_minimal_diameter
          << ", embedded maximal diameter: " << embedded_space_maximal_diameter
          << ", ratio: "
          << embedded_space_maximal_diameter / embedding_space_minimal_diameter
          << std::endl;

  AssertThrow(
      embedded_space_maximal_diameter < embedding_space_minimal_diameter,
      ExcMessage("The embedding grid is too refined (or the embedded grid "
                 "is too coarse). Adjust the parameters so that the minimal"
                 "grid size of the embedding grid is larger "
                 "than the maximal grid size of the embedded grid."));

  setup_embedding_dofs();
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::setup_embedding_dofs() {
  // Define background FE
  space_dh = std::make_unique<DoFHandler<spacedim>>(*space_grid);
  space_fe = std::make_unique<FESystem<spacedim>>(
      FE_Q<spacedim>(parameters.embedding_space_finite_element_degree + 1) ^
          spacedim,
      FE_Q<spacedim>(parameters.embedding_space_finite_element_degree));

  space_dh->distribute_dofs(*space_fe);

  A_preconditioner.reset();
  DoFRenumbering::Cuthill_McKee(*space_dh);
  std::vector<unsigned int> block_component(spacedim + 1, 0);
  block_component[spacedim] = 1;
  DoFRenumbering::component_wise(*space_dh, block_component);

  {
    constraints.clear();

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(*space_dh, constraints);
    VectorTools::interpolate_boundary_values(
        *space_dh, 1, BoundaryValues<spacedim>(), constraints,
        space_fe->component_mask(velocities));
  }
  constraints.close();

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(*space_dh, block_component);
  const types::global_dof_index n_u = dofs_per_block[0];
  const types::global_dof_index n_p = dofs_per_block[1];
  std::cout << "Number of degrees of freedom: " << space_dh->n_dofs() << " ("
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

  // Initialize vectors
  solution.reinit(dofs_per_block);
  stokes_rhs.reinit(dofs_per_block);

  deallog << "Embedding dofs: " << space_dh->n_dofs() << std::endl;
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

  const QGauss<dim> quad(parameters.coupling_quadrature_order);

  DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

  // Here, we filter the components: we want to couple DoF for velocity with the
  // ones of the multiplier.
  NonMatching::create_coupling_sparsity_pattern(
      *space_grid_tools_cache, *space_dh, *embedded_dh, quad, dsp,
      AffineConstraints<double>(), ComponentMask({true, true, false}),
      ComponentMask(), *embedded_mapping);
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

  const RightHandSide<spacedim> right_hand_side;
  std::vector<Tensor<1, spacedim>> rhs_values(n_q_points,
                                              Tensor<1, spacedim>());

  //   Quantities related to Stokes' weak form
  std::vector<SymmetricTensor<2, spacedim>> symgrad_phi_u(dofs_per_cell);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, spacedim>> phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);

  for (const auto &cell : space_dh->active_cell_iterators()) {
    fe_values.reinit(cell);
    local_matrix = 0;
    local_rhs = 0;
    local_preconditioner_matrix = 0;

    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
        div_phi_u[k] = fe_values[velocities].divergence(k, q);
        phi_u[k] = fe_values[velocities].value(k, q);
        phi_p[k] = fe_values[pressure].value(k, q);
      }

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j <= i; ++j) {
          local_matrix(i, j) +=
              (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])  // (1)
               - div_phi_u[i] * phi_p[j]                  // (2)
               - phi_p[i] * div_phi_u[j])                 // (3)
              * fe_values.JxW(q);                         // * dx

          local_preconditioner_matrix(i, j) += (phi_p[i] * phi_p[j])  // (4)
                                               * fe_values.JxW(q);    // * dx
        }

        local_rhs(i) += phi_u[i] * rhs_values[q] * fe_values.JxW(q);
      }
    }

    // Use symmetry
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
  std::cout << "Computing preconditioner ..." << std::endl;
  A_preconditioner =
      std::make_shared<typename InnerPreconditioner<spacedim>::type>();
  A_preconditioner->initialize(
      stokes_matrix.block(0, 0),
      typename InnerPreconditioner<spacedim>::type::AdditionalData());

  {
    TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

    const QGauss<dim> quad(parameters.coupling_quadrature_order);
    NonMatching::create_coupling_mass_matrix(
        *space_grid_tools_cache, *space_dh, *embedded_dh, quad, coupling_matrix,
        AffineConstraints<double>(), ComponentMask({true, true, false}),
        ComponentMask(), *embedded_mapping);

    VectorTools::interpolate(*embedded_mapping, *embedded_dh,
                             embedded_value_function, embedded_value);
  }
  std::cout << "N_rows A: " << stokes_matrix.block(0, 0).m() << std::endl;
  std::cout << "N_rows B: " << stokes_matrix.block(1, 0).m() << std::endl;
  std::cout << "N_rows C: " << coupling_matrix.n() << std::endl;
  std::cout << "N_cols C: " << coupling_matrix.m() << std::endl;
}

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(monitor, "Solve system");

  // Old way
  if (std::strcmp(parameters.solver.c_str(), "CG") == 0) {
    const InverseMatrix<SparseMatrix<double>,
                        typename InnerPreconditioner<spacedim>::type>
        A_inverse(stokes_matrix.block(0, 0), *A_preconditioner);
    Vector<double> tmp(solution.block(0).size());

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

        std::cout << "  " << solver_control.last_step()
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
  } else {
    AssertThrow(false, ExcNotImplemented());
  }

  // Store iteration counts and DoF
  results_data.dofs_background = space_dh->n_dofs();
  results_data.dofs_immersed = embedded_dh->n_dofs();
  results_data.outer_iterations = schur_solver_control.last_step();
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::output_results() {
  TimerOutput::Scope timer_section(monitor, "Output results");

  //   DataOut<spacedim> embedding_out;

  //   std::ofstream embedding_out_file("embedding.vtu");

  //   embedding_out.attach_dof_handler(*space_dh);
  //   embedding_out.add_data_vector(solution, "solution");
  //   embedding_out.build_patches(parameters.embedding_space_finite_element_degree);
  //   embedding_out.write_vtu(embedding_out_file);

  //   DataOut<dim, spacedim> embedded_out;

  //   // std::ofstream embedded_out_file("embedded.vtu");
  //   std::ofstream embedded_out_file("grid-int.gnuplot");

  //   embedded_out.attach_dof_handler(*embedded_dh);
  //   const auto dg_or_not = parameters.embedded_space_finite_element_degree ==
  //   0
  //                              ? DataOut<dim, spacedim>::type_cell_data
  //                              : DataOut<dim, spacedim>::type_automatic;
  //   embedded_out.add_data_vector(lambda, "lambda", dg_or_not);
  //   embedded_out.add_data_vector(embedded_value, "g", dg_or_not);
  //   embedded_out.build_patches(*embedded_mapping, 1.);
  //   // embedded_out.write_vtu(embedded_out_file);
  //   embedded_out.write_gnuplot(embedded_out_file);

  //   // Vector visualization
  //   std::vector<std::string> solution_names(spacedim);
  //   for (unsigned int i = 0; i < spacedim; ++i) {
  //     solution_names[i] =
  //         "vec_sol_" + std::string(1, 'x' + i);  // "vec_sol_x", "vec_sol_y",
  //         ...
  //   }
  //   std::vector<DataComponentInterpretation::DataComponentInterpretation>
  //       data_component_interpretation(
  //           spacedim,
  //           DataComponentInterpretation::component_is_part_of_vector);

  //   DataOut<spacedim> vecsol_out;
  //   vecsol_out.attach_dof_handler(*space_dh);
  //   vecsol_out.add_data_vector(solution, solution_names,
  //                              DataOut<spacedim>::type_dof_data,
  //                              data_component_interpretation);
  //   vecsol_out.build_patches(parameters.embedding_space_finite_element_degree);
  //   std::ofstream output("vector_value_solution.vtk");
  //   vecsol_out.write_vtk(output);

  std::vector<std::string> solution_names(spacedim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          spacedim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  DataOut<spacedim> data_out;
  data_out.attach_dof_handler(*space_dh);
  data_out.add_data_vector(solution, solution_names,
                           DataOut<spacedim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  std::ofstream output("solution-stokes.vtk");
  data_out.write_vtk(output);

  // Estimate condition number: TODO
  //   std::cout << "- - - - - - - - - - - - - - - - - - - - - - - -" <<
  //   std::endl; std::cout << "Estimate condition number of CCt using CG" <<
  //   std::endl; SolverControl solver_control(lambda.size(), 1e-12);
  //   SolverCG<Vector<double>> solver_cg(solver_control);

  //   solver_cg.connect_condition_number_slot(
  //       std::bind(output_double_number, std::placeholders::_1,
  //                 "Condition number estimate: "));
  //   auto Ct = linear_operator(coupling_matrix);
  //   auto C = transpose_operator(Ct);
  //   auto CCt = C * Ct;

  //   Vector<double> u(lambda);
  //   u = 0.;
  //   Vector<double> f(lambda);
  //   f = 1.;
  //   PreconditionIdentity prec_no;
  //   try {
  //     solver_cg.solve(CCt, u, f, prec_no);
  //   } catch (...) {
  //     std::cerr << "***CCt solve not successfull (see condition number
  //     above)***"
  //               << std::endl;
  //   }
}

template <int dim, int spacedim>
void IBStokesProblem<dim, spacedim>::export_results_to_csv_file() {
  std::ofstream myfile;

  AssertThrow(!parameters_filename.empty(),
              ExcMessage("You must set the name of the parameter file."));
  std::filesystem::path p(parameters_filename);
  myfile.open(p.stem().string() + ".csv",
              std::ios::app);  // get the filename and add proper extension
  // myfile << "DoF (background + immersed)"
  //        << ","
  //        << "Iteration counts"
  //        << "\n";
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
}  // namespace Stokes

int main(int argc, char **argv) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using namespace dealii;
    using namespace Stokes;

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