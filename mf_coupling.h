#ifndef mf_coupling_h
#define mf_coupling_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#ifdef DEAL_II_WITH_TRILINOS
#include <deal.II/lac/trilinos_vector.h>
#endif

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <mpi.h>

using namespace dealii;

template <int dim, int fe_degree, int n_q_points_1d, int n_components,
          typename Number, typename VectorizedArrayType>
class CouplingEvaluator {
public:
  // AdditionalData to be given to the internal remote point evaluator.
  //[TODO @fdrmrc] Add update flags for the matrix-free object.
  struct AdditionalData {
  public:
    AdditionalData(
        const unsigned int rtree_level = 0, double tolerance = 1e-12,
        const std::function<std::vector<bool>()> &marked_vertices_ = {})
        : rtree_level(rtree_level), tolerance(tolerance),
          marked_vertices(marked_vertices_) {}
    unsigned int rtree_level;
    double tolerance;
    std::function<std::vector<bool>()> marked_vertices;
  };

  CouplingEvaluator(const DoFHandler<dim> *fluid_dh,
                    const DoFHandler<dim> *solid_dh,
                    const AffineConstraints<Number> &fluid_constraints,
                    const AffineConstraints<Number> &solid_constraints,
                    const Mapping<dim> &mapping,
                    const MPI_Comm comm = MPI_COMM_SELF,
                    const AdditionalData &additional_data = AdditionalData());

  template <typename VectorType = TrilinosWrappers::MPI::Vector>
  void vmult(VectorType &dst_vector, const VectorType &source_vector) const;

  template <typename VectorType = TrilinosWrappers::MPI::Vector>
  void Tvmult(VectorType &dst_vector, const VectorType &source_vector) const;

  inline unsigned int m() const { return solid_dh->n_dofs(); }

  inline unsigned int n() const { return fluid_dh->n_dofs(); }

private:
  // Initialize the evaluator at qpoints of the solid.
  void collect_integration_points();

  // Initialize the evaluator at qpoints of the solid.
  template <typename VectorType = TrilinosWrappers::MPI::Vector>
  void interpolate_vector_to_qpoints(const VectorType &vector) const;

  DoFHandler<dim> const *fluid_dh;
  DoFHandler<dim> const *solid_dh;
  AffineConstraints<Number> *fluid_constraints;
  AffineConstraints<Number> *solid_constraints;
  Mapping<dim> *mapping;
  MPI_Comm comm;

  // Objects related to evaluation
  AdditionalData additional_data;
  std::unique_ptr<Utilities::MPI::RemotePointEvaluation<dim>>
      remote_point_evaluator;
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free;

  std::unique_ptr<FEEvaluation<dim, fe_degree, n_q_points_1d, n_components,
                               Number, VectorizedArrayType>>
      fe_eval;

  mutable std::vector<Number> integration_values;
};

template <int dim, int fe_degree, int n_q_points_1d, int n_components,
          typename Number, typename VectorizedArrayType>
CouplingEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number,
                  VectorizedArrayType>::
    CouplingEvaluator(const DoFHandler<dim> *fluid_dh_,
                      const DoFHandler<dim> *solid_dh_,
                      const AffineConstraints<Number> &fluid_constraints_,
                      const AffineConstraints<Number> &solid_constraints_,
                      const Mapping<dim> &mapping_, const MPI_Comm comm_,
                      const typename CouplingEvaluator<
                          dim, fe_degree, n_q_points_1d, n_components, Number,
                          VectorizedArrayType>::AdditionalData &rpe_data) {
  AssertThrow(fluid_dh_ != nullptr, ExcInternalError());
  AssertThrow(solid_dh_ != nullptr, ExcInternalError());
  AssertThrow(fluid_dh_->n_dofs() > 0, ExcInternalError());
  AssertThrow(solid_dh_->n_dofs() > 0, ExcInternalError());
  fluid_dh = fluid_dh_;
  solid_dh = solid_dh_;
  mapping = const_cast<Mapping<dim> *>(&mapping_);
  solid_constraints =
      const_cast<AffineConstraints<Number> *>(&solid_constraints_);
  fluid_constraints =
      const_cast<AffineConstraints<Number> *>(&fluid_constraints_);
  comm = comm_;

  remote_point_evaluator =
      std::make_unique<Utilities::MPI::RemotePointEvaluation<dim>>(
          rpe_data.tolerance, false /*enforce_unique_mapping*/,
          rpe_data.rtree_level, rpe_data.marked_vertices);

  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;
  additional_data.mapping_update_flags =
      (update_values | update_JxW_values | update_quadrature_points);
  matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  matrix_free->reinit(*mapping, *solid_dh, *solid_constraints,
                      QGauss<1>(n_q_points_1d), additional_data);

  // evaluator on the solid domain
  fe_eval =
      std::make_unique<FEEvaluation<dim, fe_degree, n_q_points_1d, n_components,
                                    Number, VectorizedArrayType>>(*matrix_free);

  collect_integration_points();
}

template <int dim, int fe_degree, int n_q_points_1d, int n_components,
          typename Number, typename VectorizedArrayType>
void CouplingEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number,
                       VectorizedArrayType>::collect_integration_points() {

  std::vector<Point<dim>> integration_points;
  integration_points.reserve(solid_dh->get_triangulation().n_active_cells() *
                             n_q_points_1d);

  for (unsigned int cell_batch_idx = 0;
       cell_batch_idx < matrix_free->n_cell_batches(); ++cell_batch_idx) {
    fe_eval->reinit(cell_batch_idx);
    for (const unsigned int q : fe_eval->quadrature_point_indices()) {
      const Point<dim, VectorizedArray<Number>> quad_points_batch =
          fe_eval->quadrature_point(q);
      for (unsigned int i = 0; i < VectorizedArray<Number>::size(); ++i) {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = quad_points_batch[d][i];
        integration_points.push_back(p);
      }
    }
  }

  // With these points, setup a RPE object to evaluate fluid basis functions
  // at the remote points.
  remote_point_evaluator->reinit(integration_points,
                                 fluid_dh->get_triangulation(), *mapping);

  AssertThrow(
      remote_point_evaluator->all_points_found(),
      ExcMessage("Could not interpolate vector to target grid: some points not "
                 "found."));
}

template <int dim, int fe_degree, int n_q_points_1d, int n_components,
          typename Number, typename VectorizedArrayType>
template <typename VectorType>
void CouplingEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number,
                       VectorizedArrayType>::
    interpolate_vector_to_qpoints(const VectorType &vector) const {

  Assert(vector.size() == solid_dh->n_dofs(), ExcDimensionMismatch());
  integration_values.reserve(solid_dh->get_triangulation().n_active_cells() *
                             n_q_points_1d);

  for (unsigned int cell_batch_idx = 0;
       cell_batch_idx < matrix_free->n_cell_batches(); ++cell_batch_idx) {
    fe_eval->reinit(cell_batch_idx);
    fe_eval->read_dof_values(vector);
    fe_eval->evaluate(EvaluationFlags::values);
    for (const unsigned int q : fe_eval->quadrature_point_indices()) {
      VectorizedArray<Number> val = fe_eval->get_value(q) * fe_eval->JxW(q);
      for (unsigned int i = 0; i < VectorizedArray<Number>::size(); ++i)
        integration_values.push_back(val[i]);
    }
  }
}

template <int dim, int fe_degree, int n_q_points_1d, int n_components,
          typename Number, typename VectorizedArrayType>
template <typename VectorType>
void CouplingEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number,
                       VectorizedArrayType>::vmult(VectorType &dst,
                                                   const VectorType &src)
    const {

  AssertDimension(src.size(), fluid_dh->n_dofs());
  AssertDimension(dst.size(), solid_dh->n_dofs());

  // First, evaluate src vector at qpoints
  const auto &values_at_qpoints = VectorTools::point_values<n_components>(
      *remote_point_evaluator, *fluid_dh, src,
      VectorTools::EvaluationFlags::insert);

  matrix_free->initialize_dof_vector(dst);

  unsigned int current_quadrature_point = 0;

  for (unsigned int cell_batch_idx = 0;
       cell_batch_idx < matrix_free->n_cell_batches(); ++cell_batch_idx) {
    fe_eval->reinit(cell_batch_idx);
    for (const unsigned int q : fe_eval->quadrature_point_indices()) {
      Tensor<1, n_components, VectorizedArray<Number>>
          tensorized_values_at_qpoints;

      for (unsigned int i = 0; i < VectorizedArrayType::size(); ++i) {
        typename FEPointEvaluation<n_components, dim, dim, Number>::value_type
            values = values_at_qpoints[current_quadrature_point++];

        if constexpr (n_components == 1) {
          tensorized_values_at_qpoints[0][i] = values;
        } else {
          for (unsigned int c = 0; c < n_components; ++c) {
            tensorized_values_at_qpoints[c][i] = values[c];
          }
        }
      }

      fe_eval->submit_value(tensorized_values_at_qpoints, q);
    }
    fe_eval->integrate(EvaluationFlags::values);
    fe_eval->distribute_local_to_global(dst);
  }
  dst.compress(VectorOperation::add);
}

template <int dim, int fe_degree, int n_q_points_1d, int n_components,
          typename Number, typename VectorizedArrayType>
template <typename VectorType>
void CouplingEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number,
                       VectorizedArrayType>::Tvmult(VectorType &dst,
                                                    const VectorType &src)
    const {

  AssertDimension(src.size(), solid_dh->n_dofs());
  AssertDimension(dst.size(), fluid_dh->n_dofs());

  dst = 0.;
  integration_values.clear();
  // Fills integration_values
  interpolate_vector_to_qpoints(src);

  const auto integration_function = [&](const auto &values,
                                        const auto &cell_data) {
    FEPointEvaluation<n_components, dim, dim, Number> phi_fluid(
        *mapping, fluid_dh->get_fe(), update_values);

    const unsigned int n_dofs_per_cell = fluid_dh->get_fe().dofs_per_cell;

    std::vector<double> local_values;
    local_values.resize(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices;
    local_dof_indices.resize(n_dofs_per_cell);

    for (const auto cell : cell_data.cell_indices()) {
      const auto cell_dofs =
          cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(
              *fluid_dh);

      const auto unit_points = cell_data.get_unit_points(cell);
      const auto src_JxW = cell_data.get_data_view(cell, values);

      phi_fluid.reinit(cell_dofs, unit_points);

      for (const auto q : phi_fluid.quadrature_point_indices())
        phi_fluid.submit_value(src_JxW[q], q);

      phi_fluid.test_and_sum(local_values, EvaluationFlags::values);

      cell_dofs->get_dof_indices(local_dof_indices);
      fluid_constraints->distribute_local_to_global(local_values,
                                                    local_dof_indices, dst);
    }
  };

  (*remote_point_evaluator)
      .template process_and_evaluate<typename FEPointEvaluation<
          n_components, dim, dim, Number>::value_type>(integration_values,
                                                       integration_function);

  dst.compress(VectorOperation::add);
}

#endif
