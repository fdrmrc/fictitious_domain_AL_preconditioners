#ifndef utilities_h
#define utilities_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <mpi.h>

#include <type_traits>

#ifdef DEAL_II_WITH_TRILINOS
#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_RowMatrixTransposer.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/non_matching/coupling.h>

// Multigrid related stuff
#include <deal.II/base/mg_level_object.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_tools.h>

#include <exception>
#include <limits>

using namespace dealii;

double compute_l2_norm_matrix(const SparseMatrix<double> &C,
                              const SparsityPattern &sparsity) {
  SparseMatrix<double> C_copy, CCt;
  SparsityPattern sp;
  sp.copy_from(sparsity);
  CCt.reinit(sp);
  C_copy.reinit(C);
  C_copy.copy_from(C);
  C.Tmmult(CCt, C_copy);

  Vector<double> v0(CCt.m());
  v0 = 1.;
  GrowingVectorMemory<Vector<double>> vector_memory;
  try {
    const double est = Utilities::LinearAlgebra::lanczos_largest_eigenvalue(
        CCt, v0, 8, vector_memory);
    std::cout << "Estimated largest eigenvalue " << std::endl;
    return std::sqrt(est);
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << std::endl;
    std::cout << "Not computed, setting NaN." << std::endl;
    return std::numeric_limits<double>::quiet_NaN();
  }
}

template <int dim, int spacedim>
void build_AMG_augmented_block(
    const DoFHandler<dim, spacedim> &velocity_dh,
    const DoFHandler<dim, spacedim> &space_dh,
    const SparseMatrix<double> &coupling_matrix,
    const SparseMatrix<double> &stiffness_matrix,
    const SparsityPattern &coupling_sparsity,
    const Vector<double> &inverse_squares,
    const AffineConstraints<double> &space_constraints, const double gamma,
    TrilinosWrappers::PreconditionAMG &amg_prec) {
  // Create the transpose.

  // First, wrap the original matrix in a Trilinos matrix
  TrilinosWrappers::SparseMatrix coupling_trilinos;
  SparsityPattern sp;
  sp.copy_from(coupling_sparsity);
  coupling_trilinos.reinit(coupling_matrix, 1e-15, true, &sp);
  auto trilinos_matrix = coupling_trilinos.trilinos_matrix();

  // Now, transpose this matrix through Trilinos
  Epetra_RowMatrixTransposer transposer(&trilinos_matrix);
  Epetra_CrsMatrix *transpose_matrix;
  int err = transposer.CreateTranspose(true, transpose_matrix);
  AssertThrow(err == 0, ExcMessage("Transpose failure!"));
#ifdef DEBUG
  std::cout << "rows original matrix:" << trilinos_matrix.NumGlobalRows()
            << std::endl;
  std::cout << "cols original matrix:" << trilinos_matrix.NumGlobalCols()
            << std::endl;
  std::cout << "rows:" << transpose_matrix->NumGlobalRows() << std::endl;
  std::cout << "cols:" << transpose_matrix->NumGlobalCols() << std::endl;
#endif

  // Now, store the transpose in a deal.II matrix for mat-mat multiplication

  // First, create the sparsity pattern for the transpose
  DynamicSparsityPattern dsp_coupling_sparsity_transposed;
  dsp_coupling_sparsity_transposed.reinit(coupling_sparsity.n_cols(),
                                          coupling_sparsity.n_rows());

  // Loop over the original sparsity pattern
  for (unsigned int row = 0; row < coupling_sparsity.n_rows(); ++row) {
    for (dealii::SparsityPattern::iterator it = coupling_sparsity.begin(row);
         it != coupling_sparsity.end(row); ++it) {
      unsigned int col = it->column();
      // Insert the transposed entry
      dsp_coupling_sparsity_transposed.add(col, row);
    }
  }
  SparsityPattern coupling_sparsity_transposed;
  coupling_sparsity_transposed.copy_from(dsp_coupling_sparsity_transposed);
  SparseMatrix<double> coupling_t;
  coupling_t.reinit(coupling_sparsity_transposed);

  // Now populate the matrix
  const int num_rows = coupling_t.m();
  for (int i = 0; i < num_rows; ++i) {
    int num_entries;
    double *values;
    int *indices;

    transpose_matrix->ExtractMyRowView(i, num_entries, values, indices);

    for (int j = 0; j < num_entries; ++j) {
      coupling_t.set(i, transpose_matrix->GCID(indices[j]), values[j]);
    }
  }
#ifdef DEBUG
  std::cout << "Populated the transpose matrix" << std::endl;
#endif

  // Now, perform matmat multiplication
  const auto &space_fe = velocity_dh.get_fe();
  SparseMatrix<double> augmented_block, BtWinvB;
  DynamicSparsityPattern dsp_aux(velocity_dh.n_dofs(), velocity_dh.n_dofs());
  const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> current_dof_indices(dofs_per_cell);
  dsp_aux.compute_mmult_pattern(coupling_sparsity,
                                coupling_sparsity_transposed);

  // Add sparsity from matrix2
  for (unsigned int row = 0; row < velocity_dh.n_dofs(); ++row) {
    for (auto it = stiffness_matrix.begin(row); it != stiffness_matrix.end(row);
         ++it) {
      dsp_aux.add(row, it->column());
    }
  }

  SparsityPattern sp_aux;
  sp_aux.copy_from(dsp_aux);
  BtWinvB.reinit(sp_aux);

  // Check that is the transpose

#ifdef DEBUG
  for (unsigned int i = 0; i < coupling_matrix.m(); ++i)
    for (unsigned int j = 0; j < coupling_matrix.n(); ++j) {
      std::cout << "Entry " << coupling_matrix.el(i, j) << " and "
                << coupling_t.el(j, i) << std::endl;
      Assert((coupling_matrix.el(i, j) - coupling_t.el(j, i) < 1e-14),
             ExcMessage("Transpose matrix is wrong!"));
    }
#endif

  SparseMatrix<double> coupling_matrix_copy;
  coupling_matrix_copy.reinit(coupling_matrix);
  coupling_matrix_copy.copy_from(coupling_matrix);
  // inverse_squares = 1.;
  coupling_matrix_copy.mmult(BtWinvB, coupling_t, inverse_squares, false);
#ifdef DEBUG
  std::cout << "Performed mat-mat multiplication" << std::endl;
  std::cout << "Rows " << BtWinvB.m() << std::endl;
  std::cout << "Cols " << BtWinvB.n() << std::endl;
  std::cout << "Norm" << BtWinvB.l1_norm() << std::endl;
#endif

  SparseMatrix<double> stiffness_matrix_copy;
  stiffness_matrix_copy.reinit(sp_aux);
  MatrixTools::create_laplace_matrix(
      velocity_dh, QGauss<spacedim>(2 * space_fe.degree + 1),
      stiffness_matrix_copy, static_cast<const Function<spacedim> *>(nullptr),
      space_constraints);

  augmented_block.reinit(stiffness_matrix_copy);
  augmented_block.copy_from(stiffness_matrix_copy);
  augmented_block.add(gamma, BtWinvB);

  //!
  const FEValuesExtractors::Vector velocity_components(0);
  const std::vector<std::vector<bool>> constant_modes =
      DoFTools::extract_constant_modes(
          space_dh, space_dh.get_fe().component_mask(velocity_components));
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.constant_modes = constant_modes;
  amg_data.elliptic = true;
  amg_data.higher_order_elements = true;
  amg_data.smoother_sweeps = 2;
  amg_data.aggregation_threshold = 0.02;

  amg_prec.initialize(augmented_block, amg_data);                 //!
  auto prec_for_cg = linear_operator(augmented_block, amg_prec);  //!
  dealii::deallog << "Initialized AMG preconditioner for augmented block"
                  << std::endl;

  // Print matrices to file to check if one is the transpose of the other
  // #ifdef DEBUG
  //   coupling_matrix.print_formatted(std::cout);
  //   coupling_t.print_formatted(std::cout);
  //   inverse_squares.print(std::cout);
  // #endif
}

std::string deallogname;
std::ofstream deallogfile;
inline void mpi_initlog(const bool console = false,
                        const unsigned int verbosity_level = 10,
                        const std::ios_base::fmtflags flags =
                            std::ios::showpoint | std::ios::left) {
#ifdef DEAL_II_WITH_MPI
  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  if (myid == 0) {
    deallogname = "output";
    deallogfile.open(deallogname.c_str());
    deallog.attach(deallogfile, true, flags);
    deallog.depth_console(console ? verbosity_level : 0);
  }
#else
  (void)console;
  (void)flags;
  // can't use this function if not using MPI
  Assert(false, ExcInternalError());
#endif
}

namespace UtilitiesAL {

namespace internal {
template <int dim0, int dim1, int spacedim>
std::pair<std::vector<Point<spacedim>>, std::vector<unsigned int>>
qpoints_over_locally_owned_cells(
    const GridTools::Cache<dim0, spacedim> &cache,
    const DoFHandler<dim1, spacedim> &immersed_dh, const Quadrature<dim1> &quad,
    const Mapping<dim1, spacedim> &immersed_mapping,
    const bool tria_is_parallel) {
  const auto &immersed_fe = immersed_dh.get_fe();
  std::vector<Point<spacedim>> points_over_local_cells;
  // Keep track of which cells we actually used
  std::vector<unsigned int> used_cells_ids;
  {
    FEValues<dim1, spacedim> fe_v(immersed_mapping, immersed_fe, quad,
                                  update_quadrature_points);
    unsigned int cell_id = 0;
    for (const auto &cell : immersed_dh.active_cell_iterators()) {
      bool use_cell = false;
      if (tria_is_parallel) {
        const auto bbox = cell->bounding_box();
        std::vector<std::pair<
            BoundingBox<spacedim>,
            typename Triangulation<dim0, spacedim>::active_cell_iterator>>
            out_vals;
        cache.get_cell_bounding_boxes_rtree().query(
            boost::geometry::index::intersects(bbox),
            std::back_inserter(out_vals));
        // Each bounding box corresponds to an active cell
        // of the embedding triangulation: we now check if
        // the current cell, of the embedded triangulation,
        // overlaps a locally owned cell of the embedding one
        for (const auto &bbox_it : out_vals)
          if (bbox_it.second->is_locally_owned()) {
            use_cell = true;
            used_cells_ids.emplace_back(cell_id);
            break;
          }
      } else
        // for sequential triangulations, simply use all cells
        use_cell = true;

      if (use_cell) {
        // Reinitialize the cell and the fe_values
        fe_v.reinit(cell);
        const std::vector<Point<spacedim>> &x_points =
            fe_v.get_quadrature_points();

        // Insert the points to the vector
        points_over_local_cells.insert(points_over_local_cells.end(),
                                       x_points.begin(), x_points.end());
      }
      ++cell_id;
    }
  }
  return {std::move(points_over_local_cells), std::move(used_cells_ids)};
}
}  // namespace internal

template <int dim0, int dim1, int spacedim, typename Matrix>
void create_coupling_mass_matrix_transpose(
    const GridTools::Cache<dim0, spacedim> &cache,
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh, const Quadrature<dim1> &quad,
    Matrix &matrix,
    const AffineConstraints<typename Matrix::value_type> &constraints,
    const ComponentMask &space_comps, const ComponentMask &immersed_comps,
    const Mapping<dim1, spacedim> &immersed_mapping,
    const AffineConstraints<typename Matrix::value_type>
        &immersed_constraints) {
  AssertDimension(matrix.m(), immersed_dh.n_dofs());
  AssertDimension(matrix.n(), space_dh.n_dofs());
  Assert(dim1 <= dim0,
         ExcMessage("This function can only work if dim1 <= dim0"));
  Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
         ExcNotImplemented());

  const bool tria_is_parallel =
      (dynamic_cast<const parallel::TriangulationBase<dim0, spacedim> *>(
           &space_dh.get_triangulation()) != nullptr);

  const auto &space_fe = space_dh.get_fe();
  const auto &immersed_fe = immersed_dh.get_fe();

  // Dof indices
  std::vector<types::global_dof_index> dofs(immersed_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> odofs(space_fe.n_dofs_per_cell());

  // Take care of components
  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(space_fe.n_components(), true)
                               : space_comps);

  const ComponentMask immersed_c =
      (immersed_comps.size() == 0
           ? ComponentMask(immersed_fe.n_components(), true)
           : immersed_comps);

  AssertDimension(space_c.size(), space_fe.n_components());
  AssertDimension(immersed_c.size(), immersed_fe.n_components());

  std::vector<unsigned int> space_gtl(space_fe.n_components(),
                                      numbers::invalid_unsigned_int);
  std::vector<unsigned int> immersed_gtl(immersed_fe.n_components(),
                                         numbers::invalid_unsigned_int);

  for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
    if (space_c[i]) space_gtl[i] = j++;

  for (unsigned int i = 0, j = 0; i < immersed_gtl.size(); ++i)
    if (immersed_c[i]) immersed_gtl[i] = j++;

  FullMatrix<typename Matrix::value_type> cell_matrix(
      immersed_dh.get_fe().n_dofs_per_cell(),
      space_dh.get_fe().n_dofs_per_cell());

  FEValues<dim1, spacedim> fe_v(
      immersed_mapping, immersed_dh.get_fe(), quad,
      update_JxW_values | update_quadrature_points | update_values);

  const unsigned int n_q_points = quad.size();
  const unsigned int n_active_c =
      immersed_dh.get_triangulation().n_active_cells();

  const auto used_cells_data = internal::qpoints_over_locally_owned_cells(
      cache, immersed_dh, quad, immersed_mapping, tria_is_parallel);

  const auto &points_over_local_cells = std::get<0>(used_cells_data);
  const auto &used_cells_ids = std::get<1>(used_cells_data);

  // Get a list of outer cells, qpoints and maps.
  const auto cpm =
      GridTools::compute_point_locations(cache, points_over_local_cells);

  const auto &all_cells = std::get<0>(cpm);
  const auto &all_qpoints = std::get<1>(cpm);
  const auto &all_maps = std::get<2>(cpm);

  std::vector<
      std::vector<typename Triangulation<dim0, spacedim>::active_cell_iterator>>
      cell_container(n_active_c);
  std::vector<std::vector<std::vector<Point<dim0>>>> qpoints_container(
      n_active_c);
  std::vector<std::vector<std::vector<unsigned int>>> maps_container(
      n_active_c);

  // Cycle over all cells of underling mesh found
  // call it omesh, elaborating the output
  for (unsigned int o = 0; o < all_cells.size(); ++o) {
    for (unsigned int j = 0; j < all_maps[o].size(); ++j) {
      // Find the index of the "owner" cell and qpoint
      // with regard to the immersed mesh
      // Find in which cell of immersed triangulation the point lies
      unsigned int cell_id;
      if (tria_is_parallel)
        cell_id = used_cells_ids[all_maps[o][j] / n_q_points];
      else
        cell_id = all_maps[o][j] / n_q_points;

      const unsigned int n_pt = all_maps[o][j] % n_q_points;

      // If there are no cells, we just add our data
      if (cell_container[cell_id].empty()) {
        cell_container[cell_id].emplace_back(all_cells[o]);
        qpoints_container[cell_id].emplace_back(
            std::vector<Point<dim0>>{all_qpoints[o][j]});
        maps_container[cell_id].emplace_back(std::vector<unsigned int>{n_pt});
      }
      // If there are already cells, we begin by looking
      // at the last inserted cell, which is more likely:
      else if (cell_container[cell_id].back() == all_cells[o]) {
        qpoints_container[cell_id].back().emplace_back(all_qpoints[o][j]);
        maps_container[cell_id].back().emplace_back(n_pt);
      } else {
        // We don't need to check the last element
        const auto cell_p =
            std::find(cell_container[cell_id].begin(),
                      cell_container[cell_id].end() - 1, all_cells[o]);

        if (cell_p == cell_container[cell_id].end() - 1) {
          cell_container[cell_id].emplace_back(all_cells[o]);
          qpoints_container[cell_id].emplace_back(
              std::vector<Point<dim0>>{all_qpoints[o][j]});
          maps_container[cell_id].emplace_back(std::vector<unsigned int>{n_pt});
        } else {
          const unsigned int pos = cell_p - cell_container[cell_id].begin();
          qpoints_container[cell_id][pos].emplace_back(all_qpoints[o][j]);
          maps_container[cell_id][pos].emplace_back(n_pt);
        }
      }
    }
  }

  typename DoFHandler<dim1, spacedim>::active_cell_iterator
      cell = immersed_dh.begin_active(),
      endc = immersed_dh.end();

  for (unsigned int j = 0; cell != endc; ++cell, ++j) {
    // Reinitialize the cell and the fe_values
    fe_v.reinit(cell);
    cell->get_dof_indices(dofs);

    // Get a list of outer cells, qpoints and maps.
    const auto &cells = cell_container[j];
    const auto &qpoints = qpoints_container[j];
    const auto &maps = maps_container[j];

    for (unsigned int c = 0; c < cells.size(); ++c) {
      // Get the ones in the current outer cell
      typename DoFHandler<dim0, spacedim>::active_cell_iterator ocell(
          *cells[c], &space_dh);
      // Make sure we act only on locally_owned cells
      if (ocell->is_locally_owned()) {
        const std::vector<Point<dim0>> &qps = qpoints[c];
        const std::vector<unsigned int> &ids = maps[c];

        FEValues<dim0, spacedim> o_fe_v(cache.get_mapping(), space_dh.get_fe(),
                                        qps, update_values);
        o_fe_v.reinit(ocell);
        ocell->get_dof_indices(odofs);

        // Reset the matrices.
        cell_matrix = typename Matrix::value_type();

        for (unsigned int i = 0; i < immersed_dh.get_fe().n_dofs_per_cell();
             ++i) {
          const auto comp_i =
              immersed_dh.get_fe().system_to_component_index(i).first;
          if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
            for (unsigned int j = 0; j < space_dh.get_fe().n_dofs_per_cell();
                 ++j) {
              const auto comp_j =
                  space_dh.get_fe().system_to_component_index(j).first;
              if (space_gtl[comp_i] == immersed_gtl[comp_j])
                for (unsigned int oq = 0; oq < o_fe_v.n_quadrature_points;
                     ++oq) {
                  // Get the corresponding q point
                  const unsigned int q = ids[oq];

                  cell_matrix(i, j) +=
                      (fe_v.shape_value(i, q) * o_fe_v.shape_value(j, oq) *
                       fe_v.JxW(q));
                }
            }
        }

        // Now assemble the matrices
        immersed_constraints.distribute_local_to_global(
            cell_matrix, dofs, constraints, odofs, matrix);
      }
    }
  }
}

template <int dim0, int dim1, int spacedim, typename Matrix>
void create_coupling_mass_matrices(
    const GridTools::Cache<dim0, spacedim> &cache,
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh, const Quadrature<dim1> &quad,
    Matrix &matrix, Matrix &matrix1,
    const AffineConstraints<typename Matrix::value_type> &constraints,
    const ComponentMask &space_comps, const ComponentMask &immersed_comps,
    const Mapping<dim1, spacedim> &immersed_mapping,
    const AffineConstraints<typename Matrix::value_type>
        &immersed_constraints) {
  AssertDimension(matrix.m(), immersed_dh.n_dofs());
  AssertDimension(matrix.n(), space_dh.n_dofs());
  AssertDimension(matrix1.n(), immersed_dh.n_dofs());
  AssertDimension(matrix1.m(), space_dh.n_dofs());
  Assert(dim1 <= dim0,
         ExcMessage("This function can only work if dim1 <= dim0"));
  Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
         ExcNotImplemented());

  const bool tria_is_parallel =
      (dynamic_cast<const parallel::TriangulationBase<dim0, spacedim> *>(
           &space_dh.get_triangulation()) != nullptr);

  const auto &space_fe = space_dh.get_fe();
  const auto &immersed_fe = immersed_dh.get_fe();

  // Dof indices
  std::vector<types::global_dof_index> dofs(immersed_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> odofs(space_fe.n_dofs_per_cell());

  // Take care of components
  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(space_fe.n_components(), true)
                               : space_comps);

  const ComponentMask immersed_c =
      (immersed_comps.size() == 0
           ? ComponentMask(immersed_fe.n_components(), true)
           : immersed_comps);

  AssertDimension(space_c.size(), space_fe.n_components());
  AssertDimension(immersed_c.size(), immersed_fe.n_components());

  std::vector<unsigned int> space_gtl(space_fe.n_components(),
                                      numbers::invalid_unsigned_int);
  std::vector<unsigned int> immersed_gtl(immersed_fe.n_components(),
                                         numbers::invalid_unsigned_int);

  for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
    if (space_c[i]) space_gtl[i] = j++;

  for (unsigned int i = 0, j = 0; i < immersed_gtl.size(); ++i)
    if (immersed_c[i]) immersed_gtl[i] = j++;

  FullMatrix<typename Matrix::value_type> cell_matrix(
      immersed_dh.get_fe().n_dofs_per_cell(),
      space_dh.get_fe().n_dofs_per_cell());

  FullMatrix<typename Matrix::value_type> cell_matrix_transpose(
      space_dh.get_fe().n_dofs_per_cell(),
      immersed_dh.get_fe().n_dofs_per_cell());

  FEValues<dim1, spacedim> fe_v(
      immersed_mapping, immersed_dh.get_fe(), quad,
      update_JxW_values | update_quadrature_points | update_values);

  const unsigned int n_q_points = quad.size();
  const unsigned int n_active_c =
      immersed_dh.get_triangulation().n_active_cells();

  const auto used_cells_data = internal::qpoints_over_locally_owned_cells(
      cache, immersed_dh, quad, immersed_mapping, tria_is_parallel);

  const auto &points_over_local_cells = std::get<0>(used_cells_data);
  const auto &used_cells_ids = std::get<1>(used_cells_data);

  // Get a list of outer cells, qpoints and maps.
  const auto cpm =
      GridTools::compute_point_locations(cache, points_over_local_cells);

  const auto &all_cells = std::get<0>(cpm);
  const auto &all_qpoints = std::get<1>(cpm);
  const auto &all_maps = std::get<2>(cpm);

  std::vector<
      std::vector<typename Triangulation<dim0, spacedim>::active_cell_iterator>>
      cell_container(n_active_c);
  std::vector<std::vector<std::vector<Point<dim0>>>> qpoints_container(
      n_active_c);
  std::vector<std::vector<std::vector<unsigned int>>> maps_container(
      n_active_c);

  // Cycle over all cells of underling mesh found
  // call it omesh, elaborating the output
  for (unsigned int o = 0; o < all_cells.size(); ++o) {
    for (unsigned int j = 0; j < all_maps[o].size(); ++j) {
      // Find the index of the "owner" cell and qpoint
      // with regard to the immersed mesh
      // Find in which cell of immersed triangulation the point lies
      unsigned int cell_id;
      if (tria_is_parallel)
        cell_id = used_cells_ids[all_maps[o][j] / n_q_points];
      else
        cell_id = all_maps[o][j] / n_q_points;

      const unsigned int n_pt = all_maps[o][j] % n_q_points;

      // If there are no cells, we just add our data
      if (cell_container[cell_id].empty()) {
        cell_container[cell_id].emplace_back(all_cells[o]);
        qpoints_container[cell_id].emplace_back(
            std::vector<Point<dim0>>{all_qpoints[o][j]});
        maps_container[cell_id].emplace_back(std::vector<unsigned int>{n_pt});
      }
      // If there are already cells, we begin by looking
      // at the last inserted cell, which is more likely:
      else if (cell_container[cell_id].back() == all_cells[o]) {
        qpoints_container[cell_id].back().emplace_back(all_qpoints[o][j]);
        maps_container[cell_id].back().emplace_back(n_pt);
      } else {
        // We don't need to check the last element
        const auto cell_p =
            std::find(cell_container[cell_id].begin(),
                      cell_container[cell_id].end() - 1, all_cells[o]);

        if (cell_p == cell_container[cell_id].end() - 1) {
          cell_container[cell_id].emplace_back(all_cells[o]);
          qpoints_container[cell_id].emplace_back(
              std::vector<Point<dim0>>{all_qpoints[o][j]});
          maps_container[cell_id].emplace_back(std::vector<unsigned int>{n_pt});
        } else {
          const unsigned int pos = cell_p - cell_container[cell_id].begin();
          qpoints_container[cell_id][pos].emplace_back(all_qpoints[o][j]);
          maps_container[cell_id][pos].emplace_back(n_pt);
        }
      }
    }
  }

  typename DoFHandler<dim1, spacedim>::active_cell_iterator
      cell = immersed_dh.begin_active(),
      endc = immersed_dh.end();

  for (unsigned int j = 0; cell != endc; ++cell, ++j) {
    // Reinitialize the cell and the fe_values
    fe_v.reinit(cell);
    cell->get_dof_indices(dofs);

    // Get a list of outer cells, qpoints and maps.
    const auto &cells = cell_container[j];
    const auto &qpoints = qpoints_container[j];
    const auto &maps = maps_container[j];

    for (unsigned int c = 0; c < cells.size(); ++c) {
      // Get the ones in the current outer cell
      typename DoFHandler<dim0, spacedim>::active_cell_iterator ocell(
          *cells[c], &space_dh);
      // Make sure we act only on locally_owned cells
      if (ocell->is_locally_owned()) {
        const std::vector<Point<dim0>> &qps = qpoints[c];
        const std::vector<unsigned int> &ids = maps[c];

        FEValues<dim0, spacedim> o_fe_v(cache.get_mapping(), space_dh.get_fe(),
                                        qps, update_values);
        o_fe_v.reinit(ocell);
        ocell->get_dof_indices(odofs);

        // Reset the matrices.
        cell_matrix = typename Matrix::value_type();

        for (unsigned int i = 0; i < immersed_dh.get_fe().n_dofs_per_cell();
             ++i) {
          const auto comp_i =
              immersed_dh.get_fe().system_to_component_index(i).first;
          if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
            for (unsigned int j = 0; j < space_dh.get_fe().n_dofs_per_cell();
                 ++j) {
              const auto comp_j =
                  space_dh.get_fe().system_to_component_index(j).first;
              if (space_gtl[comp_i] == immersed_gtl[comp_j])
                for (unsigned int oq = 0; oq < o_fe_v.n_quadrature_points;
                     ++oq) {
                  // Get the corresponding q point
                  const unsigned int q = ids[oq];

                  cell_matrix(i, j) +=
                      (fe_v.shape_value(i, q) * o_fe_v.shape_value(j, oq) *
                       fe_v.JxW(q));
                }
            }
        }

        // Now assemble the matrices
        immersed_constraints.distribute_local_to_global(
            cell_matrix, dofs, constraints, odofs, matrix);

        // Take the transpose
        cell_matrix_transpose.copy_transposed(cell_matrix);
        constraints.distribute_local_to_global(
            cell_matrix_transpose, odofs, immersed_constraints, dofs, matrix1);
      }
    }
  }
}

template <int dim0, int dim1, int spacedim, typename number>
void create_coupling_sparsity_pattern_transpose(
    const GridTools::Cache<dim0, spacedim> &cache,
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh, const Quadrature<dim1> &quad,
    SparsityPatternBase &sparsity, const AffineConstraints<number> &constraints,
    const ComponentMask &space_comps, const ComponentMask &immersed_comps,
    const Mapping<dim1, spacedim> &immersed_mapping,
    const AffineConstraints<number> &immersed_constraints) {
  AssertDimension(sparsity.n_rows(), immersed_dh.n_dofs());
  AssertDimension(sparsity.n_cols(), space_dh.n_dofs());
  Assert(dim1 <= dim0,
         ExcMessage("This function can only work if dim1 <= dim0"));
  Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
         ExcNotImplemented());

  const bool tria_is_parallel =
      (dynamic_cast<const parallel::TriangulationBase<dim0, spacedim> *>(
           &space_dh.get_triangulation()) != nullptr);
  const auto &space_fe = space_dh.get_fe();
  const auto &immersed_fe = immersed_dh.get_fe();

  // Dof indices
  std::vector<types::global_dof_index> dofs(immersed_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> odofs(space_fe.n_dofs_per_cell());

  // Take care of components
  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(space_fe.n_components(), true)
                               : space_comps);

  const ComponentMask immersed_c =
      (immersed_comps.size() == 0
           ? ComponentMask(immersed_fe.n_components(), true)
           : immersed_comps);

  AssertDimension(space_c.size(), space_fe.n_components());
  AssertDimension(immersed_c.size(), immersed_fe.n_components());

  // Global to local indices
  std::vector<unsigned int> space_gtl(space_fe.n_components(),
                                      numbers::invalid_unsigned_int);
  std::vector<unsigned int> immersed_gtl(immersed_fe.n_components(),
                                         numbers::invalid_unsigned_int);

  for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
    if (space_c[i]) space_gtl[i] = j++;

  for (unsigned int i = 0, j = 0; i < immersed_gtl.size(); ++i)
    if (immersed_c[i]) immersed_gtl[i] = j++;

  const unsigned int n_q_points = quad.size();
  const unsigned int n_active_c =
      immersed_dh.get_triangulation().n_active_cells();

  const auto qpoints_cells_data = internal::qpoints_over_locally_owned_cells(
      cache, immersed_dh, quad, immersed_mapping, tria_is_parallel);

  const auto &points_over_local_cells = std::get<0>(qpoints_cells_data);
  const auto &used_cells_ids = std::get<1>(qpoints_cells_data);

  // [TODO]: when the add_entries_local_to_global below will implement
  // the version with the dof_mask, this should be uncommented.
  //
  // // Construct a dof_mask, used to distribute entries to the sparsity
  // able< 2, bool > dof_mask(space_fe.n_dofs_per_cell(),
  //                          immersed_fe.n_dofs_per_cell());
  // of_mask.fill(false);
  // or (unsigned int i=0; i<space_fe.n_dofs_per_cell(); ++i)
  //  {
  //    const auto comp_i = space_fe.system_to_component_index(i).first;
  //    if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
  //      for (unsigned int j=0; j<immersed_fe.n_dofs_per_cell(); ++j)
  //        {
  //          const auto comp_j =
  //          immersed_fe.system_to_component_index(j).first; if
  //          (immersed_gtl[comp_j] == space_gtl[comp_i])
  //            dof_mask(i,j) = true;
  //        }
  //  }

  // Get a list of outer cells, qpoints and maps.
  const auto cpm =
      GridTools::compute_point_locations(cache, points_over_local_cells);
  const auto &all_cells = std::get<0>(cpm);
  const auto &maps = std::get<2>(cpm);

  std::vector<
      std::set<typename Triangulation<dim0, spacedim>::active_cell_iterator>>
      cell_sets(n_active_c);

  for (unsigned int i = 0; i < maps.size(); ++i) {
    // Quadrature points should be reasonably clustered:
    // the following index keeps track of the last id
    // where the current cell was inserted
    unsigned int last_id = std::numeric_limits<unsigned int>::max();
    unsigned int cell_id;
    for (const unsigned int idx : maps[i]) {
      // Find in which cell of immersed triangulation the point lies
      if (tria_is_parallel)
        cell_id = used_cells_ids[idx / n_q_points];
      else
        cell_id = idx / n_q_points;

      if (last_id != cell_id) {
        cell_sets[cell_id].insert(all_cells[i]);
        last_id = cell_id;
      }
    }
  }

  // Now we run on each cell of the immersed
  // and build the sparsity
  unsigned int i = 0;
  for (const auto &cell : immersed_dh.active_cell_iterators()) {
    // Reinitialize the cell
    cell->get_dof_indices(dofs);

    // List of outer cells
    const auto &cells = cell_sets[i];

    for (const auto &cell_c : cells) {
      // Get the ones in the current outer cell
      typename DoFHandler<dim0, spacedim>::cell_iterator ocell(*cell_c,
                                                               &space_dh);
      // Make sure we act only on locally_owned cells
      if (ocell->is_locally_owned()) {
        ocell->get_dof_indices(odofs);
        // [TODO]: When the following function will be implemented
        // for the case of non-trivial dof_mask, we should
        // uncomment the missing part.
        immersed_constraints.add_entries_local_to_global(
            dofs, constraints, odofs,
            sparsity);  //, true, dof_mask);
      }
    }
    ++i;
  }
}

template <int dim0, int dim1, int spacedim, typename number>
void create_coupling_sparsity_patterns(
    const GridTools::Cache<dim0, spacedim> &cache,
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh, const Quadrature<dim1> &quad,
    SparsityPatternBase &sparsity, SparsityPatternBase &sparsity1,
    const AffineConstraints<number> &constraints,
    const ComponentMask &space_comps, const ComponentMask &immersed_comps,
    const Mapping<dim1, spacedim> &immersed_mapping,
    const AffineConstraints<number> &immersed_constraints) {
  AssertDimension(sparsity.n_rows(), immersed_dh.n_dofs());
  AssertDimension(sparsity.n_cols(), space_dh.n_dofs());
  AssertDimension(sparsity1.n_cols(), immersed_dh.n_dofs());
  AssertDimension(sparsity1.n_rows(), space_dh.n_dofs());

  Assert(dim1 <= dim0,
         ExcMessage("This function can only work if dim1 <= dim0"));
  Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
         ExcNotImplemented());

  const bool tria_is_parallel =
      (dynamic_cast<const parallel::TriangulationBase<dim0, spacedim> *>(
           &space_dh.get_triangulation()) != nullptr);
  const auto &space_fe = space_dh.get_fe();
  const auto &immersed_fe = immersed_dh.get_fe();

  // Dof indices
  std::vector<types::global_dof_index> dofs(immersed_fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> odofs(space_fe.n_dofs_per_cell());

  // Take care of components
  const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(space_fe.n_components(), true)
                               : space_comps);

  const ComponentMask immersed_c =
      (immersed_comps.size() == 0
           ? ComponentMask(immersed_fe.n_components(), true)
           : immersed_comps);

  AssertDimension(space_c.size(), space_fe.n_components());
  AssertDimension(immersed_c.size(), immersed_fe.n_components());

  // Global to local indices
  std::vector<unsigned int> space_gtl(space_fe.n_components(),
                                      numbers::invalid_unsigned_int);
  std::vector<unsigned int> immersed_gtl(immersed_fe.n_components(),
                                         numbers::invalid_unsigned_int);

  for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
    if (space_c[i]) space_gtl[i] = j++;

  for (unsigned int i = 0, j = 0; i < immersed_gtl.size(); ++i)
    if (immersed_c[i]) immersed_gtl[i] = j++;

  const unsigned int n_q_points = quad.size();
  const unsigned int n_active_c =
      immersed_dh.get_triangulation().n_active_cells();

  const auto qpoints_cells_data = internal::qpoints_over_locally_owned_cells(
      cache, immersed_dh, quad, immersed_mapping, tria_is_parallel);

  const auto &points_over_local_cells = std::get<0>(qpoints_cells_data);
  const auto &used_cells_ids = std::get<1>(qpoints_cells_data);

  // [TODO]: when the add_entries_local_to_global below will implement
  // the version with the dof_mask, this should be uncommented.
  //
  // // Construct a dof_mask, used to distribute entries to the sparsity
  // able< 2, bool > dof_mask(space_fe.n_dofs_per_cell(),
  //                          immersed_fe.n_dofs_per_cell());
  // of_mask.fill(false);
  // or (unsigned int i=0; i<space_fe.n_dofs_per_cell(); ++i)
  //  {
  //    const auto comp_i = space_fe.system_to_component_index(i).first;
  //    if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
  //      for (unsigned int j=0; j<immersed_fe.n_dofs_per_cell(); ++j)
  //        {
  //          const auto comp_j =
  //          immersed_fe.system_to_component_index(j).first; if
  //          (immersed_gtl[comp_j] == space_gtl[comp_i])
  //            dof_mask(i,j) = true;
  //        }
  //  }

  // Get a list of outer cells, qpoints and maps.
  const auto cpm =
      GridTools::compute_point_locations(cache, points_over_local_cells);
  const auto &all_cells = std::get<0>(cpm);
  const auto &maps = std::get<2>(cpm);

  std::vector<
      std::set<typename Triangulation<dim0, spacedim>::active_cell_iterator>>
      cell_sets(n_active_c);

  for (unsigned int i = 0; i < maps.size(); ++i) {
    // Quadrature points should be reasonably clustered:
    // the following index keeps track of the last id
    // where the current cell was inserted
    unsigned int last_id = std::numeric_limits<unsigned int>::max();
    unsigned int cell_id;
    for (const unsigned int idx : maps[i]) {
      // Find in which cell of immersed triangulation the point lies
      if (tria_is_parallel)
        cell_id = used_cells_ids[idx / n_q_points];
      else
        cell_id = idx / n_q_points;

      if (last_id != cell_id) {
        cell_sets[cell_id].insert(all_cells[i]);
        last_id = cell_id;
      }
    }
  }

  // Now we run on each cell of the immersed
  // and build the sparsity
  unsigned int i = 0;
  for (const auto &cell : immersed_dh.active_cell_iterators()) {
    // Reinitialize the cell
    cell->get_dof_indices(dofs);

    // List of outer cells
    const auto &cells = cell_sets[i];

    for (const auto &cell_c : cells) {
      // Get the ones in the current outer cell
      typename DoFHandler<dim0, spacedim>::cell_iterator ocell(*cell_c,
                                                               &space_dh);
      // Make sure we act only on locally_owned cells
      if (ocell->is_locally_owned()) {
        ocell->get_dof_indices(odofs);
        // [TODO]: When the following function will be implemented
        // for the case of non-trivial dof_mask, we should
        // uncomment the missing part.
        immersed_constraints.add_entries_local_to_global(
            dofs, constraints, odofs,
            sparsity);  //, true, dof_mask);

        constraints.add_entries_local_to_global(
            odofs, immersed_constraints, dofs,
            sparsity1);  //, true, dof_mask);
      }
    }
    ++i;
  }
}

template <int dim, int spacedim, typename MatrixType = SparseMatrix<double>,
          typename VectorType = Vector<typename MatrixType::value_type>,
          typename PreconditionerType = TrilinosWrappers::PreconditionAMG>
void create_preconditioner_for_augmented_block(
    const DoFHandler<dim, spacedim> &velocity_dh,
    const DoFHandler<dim, spacedim> &space_dh, const MatrixType &velocity_block,
    const MatrixType &C, const MatrixType &Ct, const VectorType &scaling_vector,
    const AffineConstraints<double> &constraints, const double gamma,
    PreconditionerType &prec) {
  // TODO: add asserts on dimensions of operators
  Assert(dim <= spacedim, ExcImpossibleInDimSpacedim(dim, spacedim));

  if constexpr (std::is_same_v<SparseMatrix<double>, MatrixType>) {
    // Fill sparsity of augmented block
    const auto &sparsity_Ct = Ct.get_sparsity_pattern();
    const auto &sparsity_C = C.get_sparsity_pattern();

    DynamicSparsityPattern augmented_dsp(velocity_dh.n_dofs(),
                                         velocity_dh.n_dofs());
    augmented_dsp.compute_mmult_pattern(sparsity_Ct, sparsity_C);
#ifdef DEBUG
    std::cout << "Computed mmult pattern" << std::endl;
#endif

    // Now you have the sparsity of C^T * C. We need to add entries from the
    // (0,0) block
    std::vector<unsigned int> block_component(spacedim + 1, 0);
    block_component[spacedim] = 1;
    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(space_dh, block_component);
    const types::global_dof_index n_velocity_dofs = dofs_per_block[0];

    const auto &sparsity_pattern_velocity_block =
        velocity_block.get_sparsity_pattern();

    for (unsigned int row = 0; row < n_velocity_dofs; ++row)
      for (SparsityPattern::iterator it =
               sparsity_pattern_velocity_block.begin(row);
           it != sparsity_pattern_velocity_block.end(row); ++it)
        augmented_dsp.add(row, it->column());

#ifdef DEBUG
    std::cout << "Populated sparsity pattern" << std::endl;
#endif

    SparsityPattern augmented_sp;
    augmented_sp.copy_from(augmented_dsp);

    MatrixType augmented_block;
    augmented_block.reinit(augmented_sp);
    Ct.mmult(augmented_block, C, scaling_vector,
             false);  // aug = C^T *scaling_vector*C

#ifdef DEBUG
    std::cout << "Performed mat-mat multiplication" << std::endl;
#endif

    SparseMatrix<double> stiffness_matrix_copy;
    stiffness_matrix_copy.reinit(augmented_sp);
    const auto &space_fe = velocity_dh.get_fe();
    MatrixTools::create_laplace_matrix(
        velocity_dh, QGauss<spacedim>(2 * space_fe.degree + 1),
        stiffness_matrix_copy, static_cast<const Function<spacedim> *>(nullptr),
        constraints);

    // MatrixType velocity_matrix;
    // stiffness_matrix_copy.reinit(augmented_sp);
    // stiffness_matrix_copy.copy_from(velocity_block);
    stiffness_matrix_copy.add(gamma, augmented_block);

#ifdef DEBUG
    std::cout << "Performed augmentation" << std::endl;
#endif

    if constexpr (std::is_same_v<TrilinosWrappers::PreconditionAMG,
                                 std::remove_reference_t<decltype(prec)>>) {
      // Extract constant modes as we have a vector valued problem
      const FEValuesExtractors::Vector velocity_components(0);
      const std::vector<std::vector<bool>> constant_modes =
          DoFTools::extract_constant_modes(
              space_dh, space_dh.get_fe().component_mask(velocity_components));

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.constant_modes = constant_modes;
      amg_data.elliptic = true;
      amg_data.higher_order_elements = true;
      amg_data.smoother_sweeps = 2;
      amg_data.aggregation_threshold = 0.02;

      prec.initialize(stiffness_matrix_copy,
                      amg_data);  //! actually fill the preconditioner
    } else {
      // Only AMG preconditioning supported so far
      AssertThrow(false, ExcNotImplemented());
    }
  } else if constexpr (std::is_same_v<TrilinosWrappers::SparseMatrix,
                                      MatrixType>) {
    Assert((std::is_same_v<TrilinosWrappers::MPI::Vector, VectorType>),
           ExcMessage("You must use Trilinos vectors, as you are using "
                      "Trilinos matrices."));
    MatrixType augmented_block;
    //  The sparsity of augmented_block will be changed by mmult
    Ct.mmult(augmented_block, C,
             scaling_vector);  // aug = C^T *scaling_vector*C

    const Epetra_CrsGraph &epetra_graph =
        augmented_block.trilinos_sparsity_pattern();

    const IndexSet &locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(velocity_dh);
    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(velocity_dh, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(
        dsp, velocity_dh.locally_owned_dofs(), velocity_dh.get_communicator(),
        locally_relevant_dofs);

    const Epetra_CrsMatrix &epetra_matrix = augmented_block.trilinos_matrix();
    const Epetra_Map &row_map = epetra_matrix.RowMap();
    for (int i = 0; i < row_map.NumMyElements(); ++i) {
      const int global_row = row_map.GID(i);
      int num_entries;
      int *column_indices;

      epetra_graph.ExtractMyRowView(i, num_entries, column_indices);
      for (int j = 0; j < num_entries; ++j) {
        dsp.add(global_row, column_indices[j]);
      }
    }

    TrilinosWrappers::SparseMatrix stiffness_matrix_copy;
    stiffness_matrix_copy.reinit(velocity_dh.locally_owned_dofs(),
                                 velocity_dh.locally_owned_dofs(), dsp,
                                 velocity_dh.get_communicator());

    const auto &space_fe = velocity_dh.get_fe();
    MatrixTools::create_laplace_matrix(
        velocity_dh, QGauss<spacedim>(2 * space_fe.degree + 1),
        stiffness_matrix_copy, static_cast<const Function<spacedim> *>(nullptr),
        constraints);

    stiffness_matrix_copy.add(gamma, augmented_block);
    if constexpr (std::is_same_v<TrilinosWrappers::PreconditionAMG,
                                 std::remove_reference_t<decltype(prec)>>) {
      // Extract constant modes as we have a vector valued problem
      const FEValuesExtractors::Vector velocity_components(0);
      const std::vector<std::vector<bool>> constant_modes =
          DoFTools::extract_constant_modes(
              space_dh, space_dh.get_fe().component_mask(velocity_components));

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.constant_modes = constant_modes;
      amg_data.elliptic = true;
      amg_data.higher_order_elements = true;
      amg_data.smoother_sweeps = 2;
      amg_data.aggregation_threshold = 0.02;

      prec.initialize(stiffness_matrix_copy,
                      amg_data);  //! actually fill the preconditioner
    } else {
      // Only AMG preconditioning supported so far
      AssertThrow(false, ExcNotImplemented());
    }
  } else {
    // PETSc not supported so far.
    AssertThrow(false, ExcNotImplemented("Matrix type not supported!"));
  }
}

template <int dim, int spacedim, typename MatrixType = SparseMatrix<double>,
          typename VectorType = Vector<typename MatrixType::value_type>,
          typename PreconditionerType = TrilinosWrappers::PreconditionAMG>
void create_augmented_block(
    const DoFHandler<dim, spacedim> &velocity_dh, const MatrixType &C,
    const MatrixType &Ct, const VectorType &scaling_vector,
    const AffineConstraints<double> &velocity_constraints, const double gamma,
    MatrixType &augmented_matrix) {
#ifdef DEAL_II_WITH_TRILINOS

  Assert(dim <= spacedim, ExcImpossibleInDimSpacedim(dim, spacedim));

  if constexpr (std::is_same_v<TrilinosWrappers::SparseMatrix, MatrixType>) {
    Assert((std::is_same_v<TrilinosWrappers::MPI::Vector, VectorType>),
           ExcMessage("You must use Trilinos vectors, as you are using "
                      "Trilinos matrices."));

    const IndexSet &locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(velocity_dh);

    TrilinosWrappers::SparsityPattern dsp(
        velocity_dh.locally_owned_dofs(), velocity_dh.locally_owned_dofs(),
        locally_relevant_dofs, velocity_dh.get_communicator());

    DoFTools::make_sparsity_pattern(
        velocity_dh, dsp, velocity_constraints, false,
        Utilities::MPI::this_mpi_process(velocity_dh.get_communicator()));
    dsp.compress();

    MatrixType velocity_block;
    velocity_block.reinit(dsp);

    const auto &space_fe = velocity_dh.get_fe();

    {
      const QGauss<spacedim> quadrature_formula(2 * space_fe.degree + 1);
      FEValues<spacedim> fe_values(space_fe, quadrature_formula,
                                   update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values);

      const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();
      const unsigned int n_q_points = quadrature_formula.size();
      const FEValuesExtractors::Vector velocities(0);
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
      std::vector<double> div_phi_u(dofs_per_cell);

      for (const auto &cell : velocity_dh.active_cell_iterators())
        if (cell->is_locally_owned()) {
          fe_values.reinit(cell);

          cell_matrix = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            for (unsigned int k = 0; k < dofs_per_cell; ++k) {
              grad_phi_u[k] = fe_values[velocities].gradient(k, q_point);
              div_phi_u[k] = fe_values[velocities].divergence(k, q_point);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j <= i; ++j)
                cell_matrix(i, j) +=
                    (scalar_product(grad_phi_u[i], grad_phi_u[j]) +
                     gamma * div_phi_u[i] * div_phi_u[j]) *
                    fe_values.JxW(q_point);
            }
          }

          // exploit symmetry
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              cell_matrix(i, j) = cell_matrix(j, i);

          cell->get_dof_indices(local_dof_indices);
          velocity_constraints.distribute_local_to_global(
              cell_matrix, local_dof_indices, velocity_block);
        }

      velocity_block.compress(VectorOperation::add);
    }

    // Extract Trilinos types
    Epetra_CrsMatrix A_trilinos = velocity_block.trilinos_matrix();
    Epetra_CrsMatrix C_trilinos = C.trilinos_matrix();
    Epetra_CrsMatrix Ct_trilinos = Ct.trilinos_matrix();
    auto multi_vector = scaling_vector.trilinos_vector();

    Assert((A_trilinos.NumGlobalRows() !=
            C_trilinos.RangeMap().NumGlobalElements()),
           ExcMessage("Number of columns in C must match dimension of A"));

    // Ensure the MultiVector has only one column.
    Assert((multi_vector.NumVectors() == 1),
           ExcMessage("The MultiVector must have exactly one column."));

    // Create diagonal matrix from first vector of v
    // Explicitly cast the map to Epetra_Map
    const Epetra_Map &map = static_cast<const Epetra_Map &>(multi_vector.Map());

    // Create a diagonal matrix with 1 nonzero entry per row
    Epetra_CrsMatrix diag_matrix(Copy, map, 1);
    for (int i = 0; i < multi_vector.Map().NumMyElements(); ++i) {
      int global_row = multi_vector.Map().GID(i);
      double val = multi_vector[0][i];  // Access first vector
      diag_matrix.InsertGlobalValues(global_row, 1, &val, &global_row);
    }
    diag_matrix.FillComplete();

    // Compute C^T * diag(v)
    Epetra_CrsMatrix *temp1 =
        new Epetra_CrsMatrix(Copy, Ct_trilinos.RowMap(), 0);
    EpetraExt::MatrixMatrix::Multiply(Ct_trilinos, false, diag_matrix, false,
                                      *temp1);

    // Compute (C^T * diag(v)) * C
    Epetra_CrsMatrix *temp2 = new Epetra_CrsMatrix(Copy, temp1->RowMap(), 0);
    EpetraExt::MatrixMatrix::Multiply(*temp1, false, C_trilinos, false, *temp2);
    temp2->Scale(gamma);

    // Add A to the result
    Epetra_CrsMatrix *result = new Epetra_CrsMatrix(Copy, A_trilinos.RowMap(),
                                                    A_trilinos.MaxNumEntries());
    EpetraExt::MatrixMatrix::Add(A_trilinos, false, 1.0, *temp2, false, 1.0,
                                 result);
    result->FillComplete();

    // Finally, initialize the Trilinos matrix
    augmented_matrix.reinit(*result, true);

    // Delete unnecessary objects.
    delete temp1;
    delete temp2;
    delete result;

  } else {
    // PETSc not supported so far.
    AssertThrow(false, ExcNotImplemented("Matrix type not supported!"));
  }
#else
  AssertThrow(
      false,
      ExcMessage(
          "This function requires deal.II to be configured with Trilinos."));

  (void)velocity_dh;
  (void)C;
  (void)Ct;
  (void)scaling_vector;
  (void)velocity_constraints;
  (void)gamma;
  (void)augmented_matrix;
#endif
}

// Similar to the function above, but assembling multigrid matrices on every
// level.
template <int dim, int spacedim, typename MatrixType = SparseMatrix<double>,
          typename VectorType = Vector<typename MatrixType::value_type>>
void assemble_multilevel_matrices(
    const DoFHandler<dim, spacedim> &velocity_dh,
    const MGLevelObject<MatrixType> &mg_matrix_coupling,
    const MGLevelObject<MatrixType> &mg_matrix_coupling_t,
    const VectorType &scaling_vector,
    const MGConstrainedDoFs &mg_constrained_dofs,
    MGLevelObject<MatrixType> &mg_matrix,
    MGLevelObject<MatrixType> &mg_interface_in, const double gamma) {
#ifdef DEAL_II_WITH_TRILINOS

  Assert(dim <= spacedim, ExcImpossibleInDimSpacedim(dim, spacedim));

  if constexpr (std::is_same_v<TrilinosWrappers::SparseMatrix, MatrixType>) {
    Assert((std::is_same_v<TrilinosWrappers::MPI::Vector, VectorType>),
           ExcMessage("You must use Trilinos vectors, as you are using "
                      "Trilinos matrices."));

    const Triangulation<spacedim> &triangulation =
        velocity_dh.get_triangulation();
    const unsigned int n_levels = triangulation.n_global_levels();
    const MPI_Comm mpi_comm = triangulation.get_communicator();

    const auto &space_fe = velocity_dh.get_fe();
    const QGauss<spacedim> quadrature_formula(2 * space_fe.degree + 1);
    FEValues<spacedim> fe_values(space_fe, quadrature_formula,
                                 update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);

    MGLevelObject<MatrixType> mg_matrix_velocity;
    mg_matrix_velocity.resize(0, n_levels - 1);
    mg_matrix_velocity.clear_elements();

    for (unsigned int level = 0; level < n_levels; ++level) {
      const IndexSet dof_set =
          DoFTools::extract_locally_relevant_level_dofs(velocity_dh, level);

      {
        TrilinosWrappers::SparsityPattern dsp(
            velocity_dh.locally_owned_mg_dofs(level),
            velocity_dh.locally_owned_mg_dofs(level), dof_set, mpi_comm);
        MGTools::make_sparsity_pattern(velocity_dh, dsp, level);

        dsp.compress();
        mg_matrix_velocity[level].reinit(dsp);
      }
    }

    // Next, loop over all multigrid levels
    std::vector<AffineConstraints<double>> boundary_constraints(
        triangulation.n_global_levels());
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level) {
      const IndexSet dof_set =
          DoFTools::extract_locally_relevant_level_dofs(velocity_dh, level);

      {
        boundary_constraints[level].reinit(
            velocity_dh.locally_owned_mg_dofs(level),
            DoFTools::extract_locally_relevant_level_dofs(velocity_dh, level));

        for (const types::global_dof_index dof_index :
             mg_constrained_dofs.get_refinement_edge_indices(level))
          boundary_constraints[level].constrain_dof_to_zero(dof_index);
        for (const types::global_dof_index dof_index :
             mg_constrained_dofs.get_boundary_indices(level))
          boundary_constraints[level].constrain_dof_to_zero(dof_index);
        boundary_constraints[level].close();
      }

      {
        TrilinosWrappers::SparsityPattern dsp(
            velocity_dh.locally_owned_mg_dofs(level),
            velocity_dh.locally_owned_mg_dofs(level), dof_set, mpi_comm);

        MGTools::make_interface_sparsity_pattern(
            velocity_dh, mg_constrained_dofs, dsp, level);
        dsp.compress();
        mg_interface_in[level].reinit(dsp);
      }

      const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();
      const unsigned int n_q_points = quadrature_formula.size();
      const FEValuesExtractors::Vector velocities(0);
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
      std::vector<double> div_phi_u(dofs_per_cell);

      // We now loop over cells (not only **active** cells)
      for (const auto &cell : velocity_dh.cell_iterators())
        if (cell->is_locally_owned()) {
          fe_values.reinit(cell);

          cell_matrix = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            for (unsigned int k = 0; k < dofs_per_cell; ++k) {
              grad_phi_u[k] = fe_values[velocities].gradient(k, q_point);
              div_phi_u[k] = fe_values[velocities].divergence(k, q_point);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j <= i; ++j)
                cell_matrix(i, j) +=
                    (scalar_product(grad_phi_u[i], grad_phi_u[j]) +
                     gamma * div_phi_u[i] * div_phi_u[j]) *
                    fe_values.JxW(q_point);
            }
          }

          // exploit symmetry
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              cell_matrix(i, j) = cell_matrix(j, i);

          // Get dof indices for this multigrid level
          cell->get_mg_dof_indices(local_dof_indices);
          // Do classical loc2glb, but using the right multigrid constraints
          boundary_constraints[cell->level()].distribute_local_to_global(
              cell_matrix, local_dof_indices,
              mg_matrix_velocity[cell->level()]);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              if (mg_constrained_dofs.is_interface_matrix_entry(
                      cell->level(), local_dof_indices[i],
                      local_dof_indices[j]))
                mg_interface_in[cell->level()].add(local_dof_indices[i],
                                                   local_dof_indices[j],
                                                   cell_matrix(i, j));
        }

      mg_matrix_velocity[level].compress(VectorOperation::add);
      mg_interface_in[level].compress(VectorOperation::add);

      // Extract Trilinos types
      Epetra_CrsMatrix A_trilinos = mg_matrix_velocity[level].trilinos_matrix();
      Epetra_CrsMatrix C_trilinos = mg_matrix_coupling[level].trilinos_matrix();
      Epetra_CrsMatrix Ct_trilinos =
          mg_matrix_coupling_t[level].trilinos_matrix();
      auto multi_vector = scaling_vector.trilinos_vector();

      Assert((A_trilinos.NumGlobalRows() !=
              C_trilinos.RangeMap().NumGlobalElements()),
             ExcMessage("Number of columns in C must match dimension of A"));

      // Ensure the MultiVector has only one column.
      Assert((multi_vector.NumVectors() == 1),
             ExcMessage("The MultiVector must have exactly one column."));

      // Create diagonal matrix from first vector of v
      // Explicitly cast the map to Epetra_Map
      const Epetra_Map &map =
          static_cast<const Epetra_Map &>(multi_vector.Map());

      // Create a diagonal matrix with 1 nonzero entry per row
      Epetra_CrsMatrix diag_matrix(Copy, map, 1);
      for (int i = 0; i < multi_vector.Map().NumMyElements(); ++i) {
        int global_row = multi_vector.Map().GID(i);
        double val = multi_vector[0][i];  // Access first vector
        diag_matrix.InsertGlobalValues(global_row, 1, &val, &global_row);
      }
      diag_matrix.FillComplete();

      // Compute C^T * diag(v)
      Epetra_CrsMatrix *temp1 =
          new Epetra_CrsMatrix(Copy, Ct_trilinos.RowMap(), 0);
      EpetraExt::MatrixMatrix::Multiply(Ct_trilinos, false, diag_matrix, false,
                                        *temp1);

      // Compute (C^T * diag(v)) * C
      Epetra_CrsMatrix *temp2 = new Epetra_CrsMatrix(Copy, temp1->RowMap(), 0);
      EpetraExt::MatrixMatrix::Multiply(*temp1, false, C_trilinos, false,
                                        *temp2);
      temp2->Scale(gamma);

      // Add A to the result
      Epetra_CrsMatrix *result = new Epetra_CrsMatrix(
          Copy, A_trilinos.RowMap(), A_trilinos.MaxNumEntries());
      EpetraExt::MatrixMatrix::Add(A_trilinos, false, 1.0, *temp2, false, 1.0,
                                   result);
      result->FillComplete();

      // Finally, initialize the Trilinos matrix **on each level**
      mg_matrix[level].reinit(*result, true);

      // Delete unnecessary objects.
      delete temp1;
      delete temp2;
      delete result;
    }

  } else {
    // PETSc not supported so far.
    AssertThrow(false, ExcNotImplemented("Matrix type not supported!"));
  }
#else
  AssertThrow(
      false,
      ExcMessage(
          "This function requires deal.II to be configured with Trilinos."));

  (void)velocity_dh;
  (void)C;
  (void)Ct;
  (void)scaling_vector;
  (void)velocity_constraints;
  (void)gamma;
  (void)augmented_matrix;
#endif
}

}  // namespace UtilitiesAL
#endif