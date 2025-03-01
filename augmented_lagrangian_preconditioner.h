#ifndef augmented_lagrangian_prec_h
#define augmented_lagrangian_prec_h

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

class BlockPreconditionerAugmentedLagrangian {
 public:
  BlockPreconditionerAugmentedLagrangian(
      const LinearOperator<Vector<double>> Aug_inv_,
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> Ct_,
      const LinearOperator<Vector<double>> invW_, const double gamma_ = 1e2) {
    Aug_inv = Aug_inv_;
    C = C_;
    Ct = Ct_;
    invW = invW_;
    gamma = gamma_;
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const {
    v.block(0) = 0.;
    v.block(1) = 0.;

    v.block(1) = -gamma * invW * u.block(1);
    v.block(0) = Aug_inv * (u.block(0) - Ct * v.block(1));
  }

  LinearOperator<Vector<double>> K;
  LinearOperator<Vector<double>> Aug_inv;
  LinearOperator<Vector<double>> C;
  LinearOperator<Vector<double>> invW;
  LinearOperator<Vector<double>> Ct;
  double gamma;
};

class BlockPreconditionerAugmentedLagrangianStokes {
 public:
  BlockPreconditionerAugmentedLagrangianStokes(
      const LinearOperator<Vector<double>> Aug_inv_,
      const LinearOperator<Vector<double>> Bt_,
      const LinearOperator<Vector<double>> Ct_,
      const LinearOperator<Vector<double>> invW_,
      const LinearOperator<Vector<double>> Mp_inv_, const double gamma_,
      const double gamma_grad_div_) {
    Aug_inv = Aug_inv_;
    Bt = Bt_;
    Ct = Ct_;
    invW = invW_;
    Mp_inv = Mp_inv_;
    gamma = gamma_;
    gamma_grad_div = gamma_grad_div_;
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const {
    v.block(0) = 0.;
    v.block(1) = 0.;
    v.block(2) = 0.;

    v.block(2) = -gamma * invW * u.block(2);
    v.block(1) = -gamma_grad_div * Mp_inv * u.block(1);
    v.block(0) = Aug_inv * (u.block(0) - Bt * v.block(1) - Ct * v.block(2));
  }

  LinearOperator<Vector<double>> Aug_inv;
  LinearOperator<Vector<double>> Bt;
  LinearOperator<Vector<double>> invW;
  LinearOperator<Vector<double>> Mp_inv;
  LinearOperator<Vector<double>> Ct;
  double gamma;
  double gamma_grad_div;
};

namespace EllipticInterfacePreconditioners {

// Original AL preconditioner
class BlockTriangularALPreconditioner {
 public:
  BlockTriangularALPreconditioner(
      const LinearOperator<BlockVector<double>> Aug_inv_,
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> M_,
      const LinearOperator<Vector<double>> invW_, const double gamma_) {
    Aug_inv = Aug_inv_;
    C = C_;
    Ct = transpose_operator(C);
    M = M_;
    invW = invW_;
    gamma = gamma_;
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const {
    v.block(0) = 0.;
    v.block(1) = 0.;
    v.block(2) = 0.;

    v.block(2) = -gamma * invW * u.block(2);

    // Now, use the action of the Aug_inv on the first two components of the
    // solution
    BlockVector<double> uu2, result;

    uu2.reinit(2);
    uu2.block(0).reinit(u.block(0));
    uu2.block(0) = u.block(0) - Ct * v.block(2);

    uu2.block(1).reinit(u.block(1));
    uu2.block(1) = u.block(1) + 1. * M * v.block(2);

    result.reinit(2);
    result.block(0).reinit(u.block(0));
    result.block(1).reinit(u.block(1));

    result = Aug_inv * uu2;
    // Copy solutions back to the output of this routine
    v.block(0) = result.block(0);
    v.block(1) = result.block(1);
  }

  LinearOperator<BlockVector<double>> Aug_inv;
  LinearOperator<Vector<double>> C;
  LinearOperator<Vector<double>> Ct;
  LinearOperator<Vector<double>> M;
  LinearOperator<Vector<double>> invW;
  double gamma;
};

// Implementation of the modified AL preconditioner for elliptic interface
// problem. Notice how we need the inverse of the diagonal blocks.
class BlockTriangularALPreconditionerModified {
 public:
  BlockTriangularALPreconditionerModified(
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> M_,
      const LinearOperator<Vector<double>> invW_, const double gamma_,
      const LinearOperator<Vector<double>> A11_inv_,
      const LinearOperator<Vector<double>> A22_inv_) {
    A11_inv = A11_inv_;
    A22_inv = A22_inv_;
    C = C_;
    Ct = transpose_operator(C);
    M = M_;
    invW = invW_;
    gamma = gamma_;
  }

  template <typename BlockVectorType>
  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    // Assert that the block vectors have the right number of blocks
    Assert(src.n_blocks() == 3, ExcDimensionMismatch(src.n_blocks(), 3));
    Assert(dst.n_blocks() == 3, ExcDimensionMismatch(dst.n_blocks(), 3));

    // Extract the blocks from the source vector
    const auto &u = src.block(0);
    const auto &u2 = src.block(1);
    const auto &lambda = src.block(2);

    // Create temporary vectors for intermediate results
    typename BlockVectorType::BlockType temp(lambda.size());

    // 1. Compute the third block first (1st inverse application: invW)
    dst.block(2) = invW * lambda;
    dst.block(2) *= -gamma;

    // Prepare for the second block calculation
    temp = M * (-1.0 / gamma * dst.block(2));  // M * invW * lambda

    // 2. Compute the second block (2nd inverse application: A22_inv)
    // A22_inv * (u2 - gamma * M * invW * lambda)
    temp *= -gamma;
    temp += u2;
    dst.block(1) = A22_inv * temp;

    // Prepare for the first block calculation
    auto B_T = -gamma * Ct * invW * M;
    temp = B_T * dst.block(1);  // B_T * A22_inv * (...)
    temp *= -1.0;
    temp += u;

    // Add the term with Ct
    auto C_T_term = Ct * (-1.0 / gamma * dst.block(2));  // Ct * invW * lambda
    temp += gamma * C_T_term;

    // 3. Compute the first block (3rd inverse application: A11_inv)
    dst.block(0) = A11_inv * temp;
  }

  LinearOperator<Vector<double>> A11_inv;
  LinearOperator<Vector<double>> A22_inv;
  LinearOperator<Vector<double>> C;
  LinearOperator<Vector<double>> Ct;
  LinearOperator<Vector<double>> M;
  LinearOperator<Vector<double>> invW;
  double gamma;
};
}  // namespace EllipticInterfacePreconditioners

#endif