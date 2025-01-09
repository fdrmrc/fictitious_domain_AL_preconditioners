#ifndef augmented_lagrangian_prec_h
#define augmented_lagrangian_prec_h

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <type_traits>

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

template <typename VectorType, typename BlockVectorType>
class BlockPreconditionerAugmentedLagrangianStokes {
 public:
  using PayloadType =
      std::conditional_t<std::is_same_v<VectorType, Vector<double>>,
                         internal::LinearOperatorImplementation::EmptyPayload,
                         dealii::TrilinosWrappers::internal::
                             LinearOperatorImplementation::TrilinosPayload>;

  BlockPreconditionerAugmentedLagrangianStokes(
      const LinearOperator<VectorType, VectorType, PayloadType> Aug_inv_,
      const LinearOperator<VectorType, VectorType, PayloadType> Bt_,
      const LinearOperator<VectorType, VectorType, PayloadType> Ct_,
      const LinearOperator<VectorType, VectorType, PayloadType> invW_,
      const LinearOperator<VectorType, VectorType, PayloadType> Mp_inv_,
      const double gamma_, const double gamma_grad_div_) {
    Aug_inv = Aug_inv_;
    Bt = Bt_;
    Ct = Ct_;
    invW = invW_;
    Mp_inv = Mp_inv_;
    gamma = gamma_;
    gamma_grad_div = gamma_grad_div_;
  }

  void vmult(BlockVectorType &v, const BlockVectorType &u) const {
    v.block(0) = 0.;
    v.block(1) = 0.;
    v.block(2) = 0.;

    v.block(2) = -gamma * invW * u.block(2);
    v.block(1) = -gamma_grad_div * Mp_inv * u.block(1);
    v.block(0) = Aug_inv * (u.block(0) - Bt * v.block(1) - Ct * v.block(2));
  }

  LinearOperator<VectorType, VectorType, PayloadType> Aug_inv;
  LinearOperator<VectorType, VectorType, PayloadType> Bt;
  LinearOperator<VectorType, VectorType, PayloadType> invW;
  LinearOperator<VectorType, VectorType, PayloadType> Mp_inv;
  LinearOperator<VectorType, VectorType, PayloadType> Ct;
  double gamma;
  double gamma_grad_div;
};

#endif