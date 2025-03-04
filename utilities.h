#ifndef utilities_h
#define utilities_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>

#include <type_traits>

#ifdef DEAL_II_WITH_TRILINOS
#include <Epetra_CrsMatrix.h>
#include <Epetra_RowMatrixTransposer.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif

#include <deal.II/non_matching/coupling.h>

#include <exception>
#include <limits>

using namespace dealii;

double compute_l2_norm_matrix(const SparseMatrix<double>& C,
                              const SparsityPattern& sparsity) {
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
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << std::endl;
    std::cout << "Not computed, setting NaN." << std::endl;
    return std::numeric_limits<double>::quiet_NaN();
  }
}

template <typename MatrixType>
void export_to_matlab_csv(const MatrixType& matrix,
                          const std::string& filename) {
  if (matrix.m() > 1e3 || matrix.n() > 1e3) {
    return;
  }
  std::ofstream out(filename);
  out.precision(16);  // Use high precision for the values

  // Write full matrix as CSV
  for (unsigned int row = 0; row < matrix.m(); ++row) {
    for (unsigned int col = 0; col < matrix.n(); ++col) {
      out << matrix.el(row, col);
      if (col < matrix.n() - 1) {
        out << ",";
      }
    }
    out << "\n";
  }
  out.close();
}

void export_sparse_to_matlab_csv(const dealii::SparseMatrix<double>& matrix,
                                 const std::string& filename) {
  if (matrix.m() > 1e3 || matrix.n() > 1e3) {
    return;
  }
  std::ofstream out(filename);

  if (!out.is_open()) {
    throw std::runtime_error("Failed to open the file for writing.");
  }

  // Iterate over all rows
  for (unsigned int row = 0; row < matrix.m(); ++row) {
    // Iterate over all nonzero entries in the current row
    for (dealii::SparseMatrix<double>::const_iterator entry = matrix.begin(row);
         entry != matrix.end(row); ++entry) {
      // Get the column index and value
      unsigned int col = entry->column();
      double value = entry->value();
      out << row + 1 << " " << col + 1 << " " << value
          << "\n";  // MATLAB uses 1-based indexing
    }
  }

  out.close();
}

template <int dim, int spacedim>
void build_AMG_augmented_block(
    const DoFHandler<dim, spacedim>& velocity_dh,
    const DoFHandler<dim, spacedim>& space_dh,
    const SparseMatrix<double>& coupling_matrix,
    const SparseMatrix<double>& stiffness_matrix,
    const SparsityPattern& coupling_sparsity,
    const Vector<double>& inverse_squares,
    const AffineConstraints<double>& space_constraints, const double gamma,
    TrilinosWrappers::PreconditionAMG& amg_prec) {
  // Create the transpose.

  // First, wrap the original matrix in a Trilinos matrix
  TrilinosWrappers::SparseMatrix coupling_trilinos;
  SparsityPattern sp;
  sp.copy_from(coupling_sparsity);
  coupling_trilinos.reinit(coupling_matrix, 1e-15, true, &sp);
  auto trilinos_matrix = coupling_trilinos.trilinos_matrix();

  // Now, transpose this matrix through Trilinos
  Epetra_RowMatrixTransposer transposer(&trilinos_matrix);
  Epetra_CrsMatrix* transpose_matrix;
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
    double* values;
    int* indices;

    transpose_matrix->ExtractMyRowView(i, num_entries, values, indices);

    for (int j = 0; j < num_entries; ++j) {
      coupling_t.set(i, transpose_matrix->GCID(indices[j]), values[j]);
    }
  }

  std::cout << "Populated the transpose matrix" << std::endl;

  // Now, perform matmat multiplication
  const auto& space_fe = velocity_dh.get_fe();
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
      stiffness_matrix_copy, static_cast<const Function<spacedim>*>(nullptr),
      space_constraints);
  export_sparse_to_matlab_csv(stiffness_matrix_copy, "velocity_stokes.txt");
  stiffness_matrix_copy *= 0.;

  {
    const QGauss<spacedim> quadrature_formula(space_fe.degree + 2);

    FEValues<spacedim> fe_values(space_fe, quadrature_formula,
                                 update_values | update_quadrature_points |
                                     update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(spacedim);

    // Precompute stuff for Stokes' weak form
    std::vector<SymmetricTensor<2, spacedim>> symgrad_phi_u(dofs_per_cell);
    std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, spacedim>> phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);

    for (const auto& cell : velocity_dh.active_cell_iterators()) {
      fe_values.reinit(cell);
      local_matrix = 0;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          grad_phi_u[k] = fe_values[velocities].gradient(k, q);
          div_phi_u[k] = fe_values[velocities].divergence(k, q);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j <= i; ++j) {
            local_matrix(i, j) +=
                (1. * scalar_product(grad_phi_u[i],
                                     grad_phi_u[j])  // symgrad-symgrad
                 + gamma * div_phi_u[i] *
                       div_phi_u[j]) *  // grad-div stabilization
                fe_values.JxW(q);
          }
        }
      }

      // exploit symmetry
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j) {
          local_matrix(i, j) = local_matrix(j, i);
        }

      cell->get_dof_indices(local_dof_indices);
      space_constraints.distribute_local_to_global(
          local_matrix, local_dof_indices, stiffness_matrix_copy);
    }
  }

  std::cout << "assembled" << std::endl;
  augmented_block.reinit(stiffness_matrix_copy);
  augmented_block.copy_from(stiffness_matrix_copy);
  augmented_block.add(gamma, BtWinvB);
  std::cout << "Created augmented block" << std::endl;

  //!
  const FEValuesExtractors::Vector velocity_components(0);
  const std::vector<std::vector<bool>> constant_modes =
      DoFTools::extract_constant_modes(
          space_dh, space_dh.get_fe().component_mask(velocity_components));
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.constant_modes = constant_modes;
  // amg_data.elliptic = true;
  amg_data.higher_order_elements = true;
  amg_data.smoother_sweeps = 2;
  amg_data.output_details = 10;  // Maximum verbosity
  amg_data.aggregation_threshold = 0.02;

  std::cout << "Before initializing AMG" << std::endl;
  amg_prec.initialize(augmented_block, amg_data);  //!
  // auto prec_for_cg = linear_operator(augmented_block, amg_prec);  //!
  dealii::deallog << "Initialized AMG preconditioner for augmented block"
                  << std::endl;

  // Print matrices to file to check if one is the transpose of the other
  // #ifdef DEBUG
  //   coupling_matrix.print_formatted(std::cout);
  //   coupling_t.print_formatted(std::cout);
  //   inverse_squares.print(std::cout);
  // #endif

  export_to_matlab_csv(augmented_block, "augmented_matrix_stokes.csv");
  export_sparse_to_matlab_csv(augmented_block, "augmented_matrix_stokes.txt");
}

std::vector<double> linspace(double start, double end, std::size_t num) {
  AssertThrow(start < end,
              ExcMessage("Invalid range given. Check start and stop values."));
  std::vector<double> result(num);
  double step = (end - start) / (num - 1);

  // Generate values using std::iota and a lambda for transformation
  std::iota(result.begin(), result.end(), 0);
  for (std::size_t i = 0; i < num; ++i) {
    result[i] = start + step * i;
  }

  return result;
}

template <int dim, int spacedim = dim>
void build_AMG_augmented_block_scalar(
    const DoFHandler<dim, spacedim>& space_dh,
    const SparseMatrix<double>& coupling_matrix,
    const SparseMatrix<double>& stiffness_matrix,
    const Vector<double>& inverse_diag_mass_squared,
    const SparsityPattern& coupling_sparsity,
    const AffineConstraints<double>& space_constraints, const double gamma,
    const double beta_1, TrilinosWrappers::PreconditionAMG& amg_prec) {
#ifdef DEAL_II_WITH_TRILINOS

  // Create the transpose.

  // First, wrap the original matrix in a Trilinos matrix
  TrilinosWrappers::SparseMatrix coupling_trilinos;
  SparsityPattern sp;
  sp.copy_from(coupling_sparsity);
  coupling_trilinos.reinit(coupling_matrix, 1e-15, true, &sp);
  auto trilinos_matrix = coupling_trilinos.trilinos_matrix();

  // Now, transpose this matrix through Trilinos
  Epetra_RowMatrixTransposer transposer(&trilinos_matrix);
  Epetra_CrsMatrix* transpose_matrix;
  int err = transposer.CreateTranspose(true, transpose_matrix);
  AssertThrow(err == 0, ExcMessage("Transpose failure!"));
  // #ifdef DEBUG
  //   std::cout << "rows original matrix:" << trilinos_matrix.NumGlobalRows()
  //             << std::endl;
  //   std::cout << "cols original matrix:" << trilinos_matrix.NumGlobalCols()
  //             << std::endl;
  //   std::cout << "rows:" << transpose_matrix->NumGlobalRows() << std::endl;
  //   std::cout << "cols:" << transpose_matrix->NumGlobalCols() << std::endl;
  // #endif

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
    double* values;
    int* indices;

    transpose_matrix->ExtractMyRowView(i, num_entries, values, indices);

    for (int j = 0; j < num_entries; ++j) {
      coupling_t.set(i, transpose_matrix->GCID(indices[j]), values[j]);
    }
  }
#ifdef DEBUG
  std::cout << "Populated the transpose matrix" << std::endl;
#endif

  // Now, perform matmat multiplication
  const auto& space_fe = space_dh.get_fe();
  SparseMatrix<double> augmented_block, BtWinvB;
  DynamicSparsityPattern dsp_aux(space_dh.n_dofs(), space_dh.n_dofs());
  const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> current_dof_indices(dofs_per_cell);
  dsp_aux.compute_mmult_pattern(coupling_sparsity,
                                coupling_sparsity_transposed);

  // Add sparsity from matrix2
  for (unsigned int row = 0; row < space_dh.n_dofs(); ++row) {
    for (auto it = stiffness_matrix.begin(row); it != stiffness_matrix.end(row);
         ++it) {
      dsp_aux.add(row, it->column());
    }
  }

  SparsityPattern sp_aux;
  sp_aux.copy_from(dsp_aux);
  BtWinvB.reinit(sp_aux);

  // Check that is the transpose

  // #ifdef DEBUG
  //   for (unsigned int i = 0; i < coupling_matrix.m(); ++i)
  //     for (unsigned int j = 0; j < coupling_matrix.n(); ++j) {
  //       std::cout << "Entry " << coupling_matrix.el(i, j) << " and "
  //                 << coupling_t.el(j, i) << std::endl;
  //       Assert((coupling_matrix.el(i, j) - coupling_t.el(j, i) < 1e-14),
  //              ExcMessage("Transpose matrix is wrong!"));
  //     }
  // #endif

  SparseMatrix<double> coupling_matrix_copy;
  coupling_matrix_copy.reinit(coupling_matrix);
  coupling_matrix_copy.copy_from(coupling_matrix);
  coupling_matrix_copy.mmult(BtWinvB, coupling_t, inverse_diag_mass_squared,
                             false);
#ifdef DEBUG
  std::cout << "Performed mat-mat multiplication" << std::endl;
  std::cout << "Rows " << BtWinvB.m() << std::endl;
  std::cout << "Cols " << BtWinvB.n() << std::endl;
  std::cout << "Norm" << BtWinvB.l1_norm() << std::endl;
#endif

  SparseMatrix<double> stiffness_matrix_copy;
  stiffness_matrix_copy.reinit(sp_aux);

  Functions::ConstantFunction<dim> function{
      beta_1};  // the A_1 matrix is beta_1 (\grad u, \grad v)
  Vector<double> embedding_rhs_copy;
  embedding_rhs_copy.reinit(space_dh.n_dofs());
  MatrixTools::create_laplace_matrix(
      space_dh, QGauss<spacedim>(space_fe.degree + 1), stiffness_matrix_copy,
      &function, space_constraints);

  augmented_block.reinit(stiffness_matrix_copy);
  augmented_block.copy_from(stiffness_matrix_copy);
  augmented_block.add(gamma, BtWinvB);

  amg_prec.initialize(augmented_block);                           //!
  auto prec_for_cg = linear_operator(augmented_block, amg_prec);  //!
  std::cout << "Initialized AMG for A_1" << std::endl;

// Print matrices to file to check if one is the transpose of the other
// #ifdef DEBUG
//   coupling_matrix.print_formatted(std::cout);
//   coupling_t.print_formatted(std::cout);
//   inverse_squares.print(std::cout);
// #endif
#endif
}

#endif
