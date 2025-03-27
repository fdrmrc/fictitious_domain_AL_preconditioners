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
// problem. Notice how we need the inverse of the diagonal blocks. Here we
// assume that invW is a generic inverse, not necessarily depending on M. See
// the next class for the case in which W = M^2.
template <typename VectorType,
          typename BlockVectorType = TrilinosWrappers::MPI::BlockVector>
class BlockTriangularALPreconditionerModifiedGeneric {
 public:
  using PayloadType =
      std::conditional_t<std::is_same_v<VectorType, Vector<double>>,
                         internal::LinearOperatorImplementation::EmptyPayload,
                         dealii::TrilinosWrappers::internal::
                             LinearOperatorImplementation::TrilinosPayload>;

  BlockTriangularALPreconditionerModifiedGeneric(
      const LinearOperator<VectorType, VectorType, PayloadType> C_,
      const LinearOperator<VectorType, VectorType, PayloadType> M_,
      const LinearOperator<VectorType, VectorType, PayloadType> invW_,
      const double gamma_,
      const LinearOperator<VectorType, VectorType, PayloadType> A11_inv_,
      const LinearOperator<VectorType, VectorType, PayloadType> A22_inv_) {
    A11_inv = A11_inv_;
    A22_inv = A22_inv_;
    C = C_;
    Ct = transpose_operator(C);
    M = M_;
    invW = invW_;
    gamma = gamma_;
  }

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    // Assert that the block vectors have the right number of blocks
    Assert(src.n_blocks() == 3, ExcDimensionMismatch(src.n_blocks(), 3));
    Assert(dst.n_blocks() == 3, ExcDimensionMismatch(dst.n_blocks(), 3));

    // Extract the blocks from the source vector
    const auto &u = src.block(0);
    const auto &u2 = src.block(1);
    const auto &lambda = src.block(2);

    dst.block(2) = -gamma * invW * lambda;
    dst.block(1) = A22_inv * (u2 + M * dst.block(2));
    dst.block(0) = A11_inv * (u + gamma * Ct * invW * M * dst.block(1) -
                              Ct * dst.block(2));
  }

  LinearOperator<VectorType, VectorType, PayloadType> A11_inv;
  LinearOperator<VectorType, VectorType, PayloadType> A22_inv;
  LinearOperator<VectorType, VectorType, PayloadType> C;
  LinearOperator<VectorType, VectorType, PayloadType> Ct;
  LinearOperator<VectorType, VectorType, PayloadType> M;
  LinearOperator<VectorType, VectorType, PayloadType> invW;
  double gamma;
};

// Specialization of the class above in case W = M^2. Notice how invW*M
// simplifies to invM while computing the second block inside vmult().
template <typename VectorType,
          typename BlockVectorType = TrilinosWrappers::MPI::BlockVector>
class BlockTriangularALPreconditionerModified {
 public:
  using PayloadType =
      std::conditional_t<std::is_same_v<VectorType, Vector<double>>,
                         internal::LinearOperatorImplementation::EmptyPayload,
                         dealii::TrilinosWrappers::internal::
                             LinearOperatorImplementation::TrilinosPayload>;

  BlockTriangularALPreconditionerModified(
      const LinearOperator<VectorType, VectorType, PayloadType> C_,
      const LinearOperator<VectorType, VectorType, PayloadType> M_,
      const LinearOperator<VectorType, VectorType, PayloadType> invM_,
      const LinearOperator<VectorType, VectorType, PayloadType> invW_,
      const double gamma_,
      const LinearOperator<VectorType, VectorType, PayloadType> A11_inv_,
      const LinearOperator<VectorType, VectorType, PayloadType> A22_inv_) {
    A11_inv = A11_inv_;
    A22_inv = A22_inv_;
    C = C_;
    Ct = transpose_operator(C);
    M = M_;
    invM = invM_;
    invW = invW_;
    gamma = gamma_;
  }

  void vmult(BlockVectorType &dst, const BlockVectorType &src) const {
    // Assert that the block vectors have the right number of blocks
    Assert(src.n_blocks() == 3, ExcDimensionMismatch(src.n_blocks(), 3));
    Assert(dst.n_blocks() == 3, ExcDimensionMismatch(dst.n_blocks(), 3));

    // Extract the blocks from the source vector
    const auto &u = src.block(0);
    const auto &u2 = src.block(1);
    const auto &lambda = src.block(2);

    dst.block(2) = -gamma * invW * lambda;
    dst.block(1) = A22_inv * (u2 + M * dst.block(2));
    dst.block(0) =
        A11_inv * (u + gamma * Ct * invM * dst.block(1) - Ct * dst.block(2));
  }

  LinearOperator<VectorType, VectorType, PayloadType> A11_inv;
  LinearOperator<VectorType, VectorType, PayloadType> A22_inv;
  LinearOperator<VectorType, VectorType, PayloadType> C;
  LinearOperator<VectorType, VectorType, PayloadType> Ct;
  LinearOperator<VectorType, VectorType, PayloadType> M;
  LinearOperator<VectorType, VectorType, PayloadType> invM;
  LinearOperator<VectorType, VectorType, PayloadType> invW;
  double gamma;
};
}  // namespace EllipticInterfacePreconditioners

#endif