#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#include "augmented_lagrangian_preconditioner.h"
#include "rational_preconditioner.h"
#include "utilities.h"

namespace Step60 {
using namespace dealii;

template <int dim, int spacedim = dim>
class DistributedLagrangeProblem {
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

  DistributedLagrangeProblem(const Parameters &parameters);

  void run();

 private:
  const Parameters &parameters;

  void setup_grids_and_dofs();

  void setup_embedding_dofs();

  void setup_embedded_dofs();

  void setup_coupling();

  void assemble_system();

  void solve();

  void output_results();

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

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedding_rhs_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_value_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedding_dirichlet_boundary_function;

  ParameterAcceptorProxy<ReductionControl> schur_solver_control;

  SparsityPattern stiffness_sparsity;
  SparsityPattern mass_sparsity;
  SparsityPattern Mass_sparsity;
  SparsityPattern coupling_sparsity;

  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> Mass_matrix;
  SparseMatrix<double> embedded_stiffness_matrix;
  SparseMatrix<double> coupling_matrix;

  AffineConstraints<double> constraints;

  Vector<double> solution;
  Vector<double> embedding_rhs;

  Vector<double> lambda;
  Vector<double> embedded_rhs;
  Vector<double> embedded_value;

  TimerOutput monitor;
};

template <int dim, int spacedim>
DistributedLagrangeProblem<dim, spacedim>::Parameters::Parameters()
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
DistributedLagrangeProblem<dim, spacedim>::DistributedLagrangeProblem(
    const Parameters &parameters)
    : parameters(parameters),
      embedded_configuration_function("Embedded configuration", spacedim),
      embedding_rhs_function("Embedding rhs function"),
      embedded_value_function("Embedded value"),
      embedding_dirichlet_boundary_function(
          "Embedding Dirichlet boundary conditions"),
      schur_solver_control("Schur solver control"),
      monitor(std::cout, TimerOutput::summary,
              TimerOutput::cpu_and_wall_times) {
  embedded_configuration_function.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4,Cy=.4");

        ParameterAcceptor::prm.set("Function expression",
                                   "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      });

  embedding_rhs_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  embedded_value_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });

  embedding_dirichlet_boundary_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  schur_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "1000");
    ParameterAcceptor::prm.set("Reduction", "1.e-12");
    ParameterAcceptor::prm.set("Tolerance", "1.e-12");
  });
}

template <int dim, int spacedim>
void DistributedLagrangeProblem<dim, spacedim>::setup_grids_and_dofs() {
  TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

  space_grid = std::make_unique<Triangulation<spacedim>>();

  GridGenerator::hyper_cube(*space_grid, 0., 1, true);

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
void DistributedLagrangeProblem<dim, spacedim>::setup_embedding_dofs() {
  space_dh = std::make_unique<DoFHandler<spacedim>>(*space_grid);
  space_fe = std::make_unique<FE_Q<spacedim>>(
      parameters.embedding_space_finite_element_degree);
  space_dh->distribute_dofs(*space_fe);

  DoFTools::make_hanging_node_constraints(*space_dh, constraints);
  for (const types::boundary_id id : parameters.dirichlet_ids) {
    VectorTools::interpolate_boundary_values(
        *space_dh, id, embedding_dirichlet_boundary_function, constraints);
  }
  constraints.close();

  DynamicSparsityPattern dsp(space_dh->n_dofs(), space_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*space_dh, dsp, constraints);
  stiffness_sparsity.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity);

  DynamicSparsityPattern mass_dsp(embedded_dh->n_dofs(), embedded_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*embedded_dh, mass_dsp);
  mass_sparsity.copy_from(mass_dsp);
  mass_matrix.reinit(mass_sparsity);                // M_immersed
  embedded_stiffness_matrix.reinit(mass_sparsity);  // A_immersed

  DynamicSparsityPattern Mass_dsp(space_dh->n_dofs(), space_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*space_dh, Mass_dsp, constraints);
  Mass_sparsity.copy_from(Mass_dsp);
  Mass_matrix.reinit(Mass_sparsity);

  solution.reinit(space_dh->n_dofs());
  embedding_rhs.reinit(space_dh->n_dofs());

  deallog << "Embedding dofs: " << space_dh->n_dofs() << std::endl;
}

template <int dim, int spacedim>
void DistributedLagrangeProblem<dim, spacedim>::setup_embedded_dofs() {
  embedded_dh = std::make_unique<DoFHandler<dim, spacedim>>(*embedded_grid);

  if (parameters.embedded_space_finite_element_degree > 0) {
    // use continuous elements if degree>0
    embedded_fe = std::make_unique<FE_Q<dim, spacedim>>(
        parameters.embedded_space_finite_element_degree);
  } else if (parameters.embedded_space_finite_element_degree == 0) {
    // otherwise, DG(0) elements for the multiplier
    embedded_fe = std::make_unique<FE_DGQ<dim, spacedim>>(0);
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
void DistributedLagrangeProblem<dim, spacedim>::setup_coupling() {
  TimerOutput::Scope timer_section(monitor, "Setup coupling");

  const QGauss<dim> quad(parameters.coupling_quadrature_order);

  DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

  NonMatching::create_coupling_sparsity_pattern(
      *space_grid_tools_cache, *space_dh, *embedded_dh, quad, dsp,
      AffineConstraints<double>(), ComponentMask(), ComponentMask(),
      *embedded_mapping);
  coupling_sparsity.copy_from(dsp);
  coupling_matrix.reinit(coupling_sparsity);
}

template <int dim, int spacedim>
void DistributedLagrangeProblem<dim, spacedim>::assemble_system() {
  {
    TimerOutput::Scope timer_section(monitor, "Assemble system");

    MatrixTools::create_laplace_matrix(
        *space_dh, QGauss<spacedim>(2 * space_fe->degree + 1), stiffness_matrix,
        embedding_rhs_function, embedding_rhs,
        static_cast<const Function<spacedim> *>(nullptr), constraints);

    MatrixTools::create_mass_matrix(
        *space_dh, QGauss<spacedim>(2 * space_fe->degree + 1), Mass_matrix,
        static_cast<const Function<spacedim> *>(nullptr), constraints);

    MatrixTools::create_laplace_matrix(*embedded_mapping, *embedded_dh,
                                       QGauss<dim>(2 * space_fe->degree + 1),
                                       embedded_stiffness_matrix);

    MatrixTools::create_mass_matrix(*embedded_mapping, *embedded_dh,
                                    QGauss<dim>(2 * embedded_fe->degree + 1),
                                    mass_matrix);

    VectorTools::create_right_hand_side(
        *embedded_mapping, *embedded_dh,
        QGauss<dim>(2 * embedded_fe->degree + 1), embedded_value_function,
        embedded_rhs);
  }
  {
    TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

    const QGauss<dim> quad(parameters.coupling_quadrature_order);
    NonMatching::create_coupling_mass_matrix(
        *space_grid_tools_cache, *space_dh, *embedded_dh, quad, coupling_matrix,
        AffineConstraints<double>(), ComponentMask(), ComponentMask(),
        *embedded_mapping);

    VectorTools::interpolate(*embedded_mapping, *embedded_dh,
                             embedded_value_function, embedded_value);
  }
}

void output_double_number(double input, const std::string &text) {
  std::cout << text << input << std::endl;
}

template <int dim, int spacedim>
void DistributedLagrangeProblem<dim, spacedim>::solve() {
  TimerOutput::Scope timer_section(monitor, "Solve system");

  // // Old way
  if (std::strcmp(parameters.solver.c_str(), "CG") == 0) {
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S = C * K_inv * Ct;
    SolverCG<Vector<double>> solver_cg(schur_solver_control);
    auto S_inv = inverse_operator(S, solver_cg, PreconditionIdentity());

    lambda = S_inv * (C * K_inv * embedding_rhs - embedded_rhs);

    solution = K_inv * (embedding_rhs - Ct * lambda);

    constraints.distribute(solution);
  }
  // // GMRES with left lower triangular preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "GMRES") == 0) {
    std::cout << "Solving with GMRES" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv = C * K * Ct + M;
    // auto S_inv = C * K * Ct;

    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});

    auto prec = block_operator<2, 2, BlockVector<double>>(
        {{{{K_inv, 0 * Ct}}, {{S_inv * C * K_inv, -1 * S_inv}}}});

    // Initialize block structure
    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    typename SolverGMRES<BlockVector<double>>::AdditionalData data;
    data.force_re_orthogonalization = true;

    SolverGMRES<BlockVector<double>> solver_gmres(schur_solver_control, data);

    solver_gmres.solve(AA, solution_block, system_rhs_block, prec);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  }
  // // GMRES with right upper triangular preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "GMRES_right") == 0) {
    std::cout << "Solving with GMRES (right preconditioning)" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv = C * K * Ct + M;
    // auto S_inv = C * K * Ct;

    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});

    auto prec = block_operator<2, 2, BlockVector<double>>(
        {{{{K_inv, K_inv * Ct * S_inv}}, {{0 * C, -1 * S_inv}}}});

    // Initialize block structure
    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    typename SolverGMRES<BlockVector<double>>::AdditionalData data;
    data.force_re_orthogonalization = true;
    data.right_preconditioning = true;

    SolverGMRES<BlockVector<double>> solver_gmres(schur_solver_control, data);

    solver_gmres.solve(AA, solution_block, system_rhs_block, prec);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  }
  // // FGMRES with right upper triangular preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "FGMRES") == 0) {
    std::cout << "Solving with FGMRES" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv = C * K * Ct + M;
    // auto S_inv = C * K * Ct;

    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});

    auto prec = block_operator<2, 2, BlockVector<double>>(
        {{{{K_inv, K_inv * Ct * S_inv}}, {{0 * C, -1 * S_inv}}}});

    // Initialize block structure
    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    SolverFGMRES<BlockVector<double>> solver_fgmres(schur_solver_control);

    solver_fgmres.solve(AA, solution_block, system_rhs_block, prec);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  }
  // // GMRES with ELMAN right upper triangular preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "ELMAN_triang") == 0) {
    std::cout << "Solving with ELMAN right-preconditioning" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    IterationNumberControl c_ct_solver_control(40, 1e-12, false, false);
    auto C_Ct = C * Ct;
    SolverCG<Vector<double>> solver_cg_c_ct(c_ct_solver_control);
    auto C_Ct_inv =
        inverse_operator(C_Ct, solver_cg_c_ct, PreconditionIdentity());
    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv = (C_Ct_inv * C * K * Ct * C_Ct_inv);
    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});

    auto prec_elman = block_operator<2, 2, BlockVector<double>>(
        {{{{K_inv, K_inv * Ct * S_inv}}, {{0 * C, -1 * S_inv}}}});

    // Initialize block structure
    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    typename SolverGMRES<BlockVector<double>>::AdditionalData data;
    data.force_re_orthogonalization = true;
    data.right_preconditioning = true;

    SolverGMRES<BlockVector<double>> solver_gmres(schur_solver_control, data);

    solver_gmres.solve(AA, solution_block, system_rhs_block, prec_elman);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  }  // // GMRES with scaled-BFBt right upper triangular preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "scaled-BFBt") == 0) {
    std::cout << "Solving with scaled-BFBt right-preconditioning" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(Mass_matrix);
    auto mass = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    // Inverse of the Diagonal mass matrix
    Vector<double> diag_M;
    diag_M.reinit(Mass_matrix.m());
    for (unsigned int i = 0; i < Mass_matrix.m(); ++i)
      diag_M(i) = 1.0 / Mass_matrix.diag_element(i);
    DiagonalMatrix<Vector<double>> diag_mass_inv(diag_M);

    // Inverse of the Diagonal stiffness matrix
    Vector<double> diag_K;
    diag_K.reinit(stiffness_matrix.m());
    for (unsigned int i = 0; i < stiffness_matrix.m(); ++i)
      diag_K(i) = 1.0 / stiffness_matrix.diag_element(i);
    DiagonalMatrix<Vector<double>> diag_K_inv(diag_K);

    // Inverse of the Lumped mass matrix
    Vector<double> lumped_M;
    Vector<double> ones;
    lumped_M.reinit(Mass_matrix.m());
    ones.reinit(Mass_matrix.m());
    ones = 1.;
    Mass_matrix.vmult(lumped_M, ones);
    for (unsigned int i = 0; i < Mass_matrix.m(); ++i)
      lumped_M(i) = 1.0 / lumped_M(i);
    DiagonalMatrix<Vector<double>> lumped_Mass_matrix_inv(lumped_M);

    // Inverse of the Immersed lumped mass matrix
    Vector<double> lumped_m;
    lumped_m.reinit(mass_matrix.m());
    for (unsigned int i = 0; i < mass_matrix.m(); ++i) {
      lumped_m(i) = 0.0;
      for (auto it = mass_matrix.begin(i); it != mass_matrix.end(i); ++it)
        lumped_m(i) += it->value();
      lumped_m(i) = 1.0 / lumped_m(i);
    }
    DiagonalMatrix<Vector<double>> lumped_mass_matrix_inv(lumped_m);

    auto M1_inv = linear_operator(lumped_Mass_matrix_inv);
    auto M2_inv = linear_operator(lumped_Mass_matrix_inv);
    auto M3_inv = linear_operator(lumped_mass_matrix_inv);

    IterationNumberControl c_ct_solver_control(40, 1e-12, false, false);
    auto C_Ct2 = C * M2_inv * Ct;
    SolverCG<Vector<double>> solver_cg_c_ct(c_ct_solver_control);
    auto C_Ct2_inv =
        inverse_operator(C_Ct2, solver_cg_c_ct, PreconditionIdentity());

    auto C_Ct1 = C * M1_inv * Ct;
    auto C_Ct1_inv =
        inverse_operator(C_Ct1, solver_cg_c_ct, PreconditionIdentity());

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv = C_Ct2_inv * C * M2_inv * K * M2_inv * Ct * C_Ct2_inv;

    // auto S_inv = M3_inv * C * M2_inv * K * M2_inv * Ct * M3_inv;

    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});

    auto prec_elman = block_operator<2, 2, BlockVector<double>>(
        {{{{K_inv, K_inv * Ct * S_inv}}, {{0 * C, -1 * S_inv}}}});

    // Initialize block structure
    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    typename SolverGMRES<BlockVector<double>>::AdditionalData data;
    data.force_re_orthogonalization = true;
    data.right_preconditioning = true;

    SolverGMRES<BlockVector<double>> solver_gmres(schur_solver_control, data);

    solver_gmres.solve(AA, solution_block, system_rhs_block, prec_elman);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  }
  // // GMRES with Juvigny diagonal preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "JUVIGNY") == 0) {
    std::cout << "Solving with JUVIGNY diagonal-preconditioning" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv = C * K * Ct;

    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});

    auto prec_diag = block_operator<2, 2, BlockVector<double>>(
        {{{{K_inv, 0 * C}}, {{0 * Ct, S_inv}}}});

    // Initialize block structure
    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    typename SolverGMRES<BlockVector<double>>::AdditionalData data;
    data.force_re_orthogonalization = true;
    data.right_preconditioning = false;

    SolverGMRES<BlockVector<double>> solver_gmres(schur_solver_control, data);

    solver_gmres.solve(AA, solution_block, system_rhs_block, prec_diag);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  }
  // Michal Preconditioner
  else if (std::strcmp(parameters.solver.c_str(), "MICHAL") == 0) {
    std::cout << "Solving with MICHAL preconditioner" << std::endl;
    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto MassS = linear_operator(mass_matrix);

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto S_inv_prec = C * K * Ct + MassS;

    auto S = C * K_inv * Ct;
    SolverCG<Vector<double>> solver_cg(schur_solver_control);

    PrimitiveVectorMemory<Vector<double>> mem;
    SolverGMRES<Vector<double>> solver_gmres(
        schur_solver_control, mem,
        SolverGMRES<Vector<double>>::AdditionalData(20));
    SolverMinRes<Vector<double>> solver_minres(schur_solver_control);

    auto S_inv = inverse_operator(S, solver_minres,
                                  /* PreconditionIdentity() */ S_inv_prec);

    lambda = S_inv * (C * K_inv * embedding_rhs - embedded_rhs);

    solution = K_inv * (embedding_rhs - Ct * lambda);

    constraints.distribute(solution);
  } else if (std::strcmp(parameters.solver.c_str(), "rational") == 0) {
    // intialize operators and block structure
    auto K = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;

    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    auto AA =
        block_operator<2, 2, BlockVector<double>>({{{{K, Ct}}, {{C, Zero}}}});
    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    system_rhs_block.block(0) = embedding_rhs;
    system_rhs_block.block(1) = embedded_rhs;

    // First, compute a bound on the spectral radius of M^{-1)A. That is needed
    // for the rational preconditioner routine
    std::vector<double> min_diags;
    for (unsigned int i = 0; i < mass_matrix.m(); ++i)
      min_diags.push_back(mass_matrix.diag_element(i));

    double rho_bound = (embedded_stiffness_matrix.linfty_norm()) /
                       (*std::min_element(min_diags.begin(), min_diags.end()));

    std::cout << "Upper bound on spectral radius of M^(-1)A: " << rho_bound
              << std::endl;

    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);
    auto K_inv = linear_operator(K, K_inv_umfpack);
    SolverControl solver_control(2000, 1e-14, false, false);

    // Construct the rational preconditioner to be given to fgmres
    RationalPreconditioner rational_prec{K_inv, &embedded_stiffness_matrix,
                                         &mass_matrix, rho_bound};

    // SolverGMRES<BlockVector<double>> solver_min_res(schur_solver_control);
    SolverMinRes<BlockVector<double>> solver_min_res(schur_solver_control);

    solver_min_res.solve(AA, solution_block, system_rhs_block, rational_prec);

    solution = solution_block.block(0);

    constraints.distribute(solution);
  } else if (std::strcmp(parameters.solver.c_str(), "augmented") == 0) {
    // intialize operators and block structure
    auto K = linear_operator(stiffness_matrix);
    auto k = linear_operator(embedded_stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);
    auto M = linear_operator(mass_matrix);
    const auto Zero = M * 0.0;
    SparseDirectUMFPACK M_inv_umfpack;
    M_inv_umfpack.initialize(mass_matrix);

    // Inverse of the Immersed lumped mass matrix
    // Vector<double> lumped_m;
    // lumped_m.reinit(mass_matrix.m());
    // for (unsigned int i = 0; i < mass_matrix.m(); ++i) {
    // lumped_m(i) = 0.0;
    // for (auto it = mass_matrix.begin(i); it != mass_matrix.end(i); ++it)
    // lumped_m(i) += it->value();
    // lumped_m(i) = 1.0 / lumped_m(i);
    //}
    // DiagonalMatrix<Vector<double>> lumped_mass_matrix_inv(lumped_m);
    // auto invW1 = linear_operator(lumped_mass_matrix_inv);

    // const double h_immersed = GridTools::minimal_cell_diameter(*space_grid);
    // const double gamma = 1e2 / h_immersed;
    // auto invW = linear_operator(mass_matrix, M_inv_umfpack);
    // auto Aug = K + gamma * Ct * invW * C;

    // IdentityMatrix W(mass_matrix.m());
    // auto invW = linear_operator(W);

    const double gamma = 10;
    auto invW1 = linear_operator(mass_matrix, M_inv_umfpack);
    auto invW = invW1 * invW1;
    auto Aug = K + gamma * Ct * invW * C;

    // deallog << "stiffness_matrix.linfty_norm(): "
    //<< stiffness_matrix.linfty_norm() << std::endl;
    // deallog << "coupling_matrix.l1_norm(): " << coupling_matrix.l1_norm()
    //<< std::endl;
    // const double gamma =
    // stiffness_matrix.linfty_norm() /
    //(coupling_matrix.l1_norm() * coupling_matrix.l1_norm());

    // auto Aug = K + gamma * Ct * invW * C;
    deallog << "gamma: " << gamma << std::endl;

    BlockVector<double> solution_block;
    BlockVector<double> system_rhs_block;

    auto AA = block_operator<2, 2, BlockVector<double>>(
        {{{{Aug, Ct}}, {{C, Zero}}}});  //! Augmented the (1,1) block
    AA.reinit_domain_vector(solution_block, false);
    AA.reinit_range_vector(system_rhs_block, false);

    solution_block.block(0) = solution;
    solution_block.block(1) = lambda;

    // lagrangian term
    Vector<double> tmp;
    tmp.reinit(embedding_rhs.size());
    tmp = gamma * Ct * invW * embedded_rhs;
    system_rhs_block.block(0) += tmp;  // ! augmented
    system_rhs_block.block(1) = embedded_rhs;

    SolverControl control_lagrangian(6000, 1e-11, false, false);
    SolverCG<Vector<double>> solver_lagrangian(control_lagrangian);

    auto Aug_inv =
        inverse_operator(Aug, solver_lagrangian, PreconditionIdentity());

    SolverFGMRES<BlockVector<double>> solver_fgmres(schur_solver_control);

    BlockPreconditionerAugmentedLagrangian augmented_lagrangian_preconditioner{
        Aug_inv, C, Ct, invW, gamma};
    solver_fgmres.solve(AA, solution_block, system_rhs_block,
                        augmented_lagrangian_preconditioner);

    solution = solution_block.block(0);

    constraints.distribute(solution);

  }

  else {
    AssertThrow(false, ExcNotImplemented());
  }
}

template <int dim, int spacedim>
void DistributedLagrangeProblem<dim, spacedim>::output_results() {
  TimerOutput::Scope timer_section(monitor, "Output results");

  DataOut<spacedim> embedding_out;

  std::ofstream embedding_out_file("embedding.vtu");

  embedding_out.attach_dof_handler(*space_dh);
  embedding_out.add_data_vector(solution, "solution");
  embedding_out.build_patches(parameters.embedding_space_finite_element_degree);
  embedding_out.write_vtu(embedding_out_file);

  DataOut<dim, spacedim> embedded_out;

  // std::ofstream embedded_out_file("embedded.vtu");
  std::ofstream embedded_out_file("grid-int.gnuplot");

  embedded_out.attach_dof_handler(*embedded_dh);
  const auto dg_or_not = parameters.embedded_space_finite_element_degree == 0
                             ? DataOut<dim, spacedim>::type_cell_data
                             : DataOut<dim, spacedim>::type_automatic;
  embedded_out.add_data_vector(lambda, "lambda", dg_or_not);
  embedded_out.add_data_vector(embedded_value, "g", dg_or_not);
  embedded_out.build_patches(*embedded_mapping, 1.);
  // embedded_out.write_vtu(embedded_out_file);
  embedded_out.write_gnuplot(embedded_out_file);

  // Estimate condition number
  std::cout << "- - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
  std::cout << "Estimate condition number of BBt using CG" << std::endl;
  SolverControl solver_control(lambda.size(), 1e-12);
  SolverCG<Vector<double>> solver_cg(solver_control);

  solver_cg.connect_condition_number_slot(
      std::bind(output_double_number, std::placeholders::_1,
                "Condition number estimate: "));
  auto Ct = linear_operator(coupling_matrix);
  auto C = transpose_operator(Ct);
  auto BBt = C * Ct;

  Vector<double> u(lambda);
  u = 0.;
  Vector<double> f(lambda);
  f = 1.;
  PreconditionIdentity prec_no;
  try {
    solver_cg.solve(BBt, u, f, prec_no);
  } catch (...) {
    std::cerr << "***BBt solve not successfull (see condition number above)***"
              << std::endl;
    ;
  }

  // Vector<double> u(lambda);
  // u = 0.;
  // Vector<double> f(lambda);
  // f = 1.;
  // PreconditionIdentity prec_no;
  // auto                 M   = linear_operator(mass_matrix);
  // auto                 Mt  = transpose_operator(M);
  // auto                 MMt = M * Mt;
  // // auto                 K  = linear_operator(stiffness_matrix);
  // solver_cg.solve(MMt, u, f, prec_no);
}

template <int dim, int spacedim>
void DistributedLagrangeProblem<dim, spacedim>::run() {
  AssertThrow(parameters.initialized, ExcNotInitialized());
  deallog.depth_console(parameters.verbosity_level);

  setup_grids_and_dofs();
  setup_coupling();
  assemble_system();
  solve();
  output_results();
}
}  // namespace Step60

int main(int argc, char **argv) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using namespace dealii;
    using namespace Step60;

    const unsigned int dim = 1, spacedim = 2;

    DistributedLagrangeProblem<dim, spacedim>::Parameters parameters;
    DistributedLagrangeProblem<dim, spacedim> problem(parameters);

    std::string parameter_file;
    if (argc > 1)
      parameter_file = argv[1];
    else
      parameter_file = "parameters.prm";

    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
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