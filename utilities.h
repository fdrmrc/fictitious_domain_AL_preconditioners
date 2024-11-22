#ifndef utilities_h
#define utilities_h

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/utilities.h>
#include <deal.II/lac/vector.h>

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
  const double est = Utilities::LinearAlgebra::lanczos_largest_eigenvalue(
      CCt, v0, 8, vector_memory);
  std::cout << "Estimated largest eigenvalue " << std::endl;
  return std::sqrt(est);
}

#endif