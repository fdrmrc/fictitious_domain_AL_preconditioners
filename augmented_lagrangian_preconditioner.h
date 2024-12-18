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

class BlockPreconditionerAugmentedLagrangian
{
public:
  BlockPreconditionerAugmentedLagrangian(
      const LinearOperator<Vector<double>> Aug_inv_,
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> Ct_,
      const LinearOperator<Vector<double>> invW_, const double gamma_ = 1e2)
  {
    Aug_inv = Aug_inv_;
    C = C_;
    Ct = Ct_;
    invW = invW_;
    gamma = gamma_;
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const
  {
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

class BlockPreconditionerAugmentedLagrangianStokes
{
public:
  BlockPreconditionerAugmentedLagrangianStokes(
      const LinearOperator<Vector<double>> Aug_inv_,
      const LinearOperator<Vector<double>> B_,
      const LinearOperator<Vector<double>> Bt_,
      const LinearOperator<Vector<double>> C_,
      const LinearOperator<Vector<double>> Ct_,
      const LinearOperator<Vector<double>> invW_,
      const LinearOperator<Vector<double>> Mp_inv_, const double gamma_)
  {
    Aug_inv = Aug_inv_;
    B = B_;
    Bt = Bt_;

    SB = -1 * B * Aug_inv * Bt;

    C = C_;
    Ct = Ct_;
    invW = invW_;
    Mp_inv = Mp_inv_;
    gamma = gamma_;
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const
  {
    SolverControl control_SB(1000, 1e-6, false, false);
    SolverCG<Vector<double>> solver_SB(control_SB);
    SB_inv = inverse_operator(SB, solver_SB, PreconditionIdentity());

    // auto SC = -1 * C * Aug_inv * Ct;
    // auto SC_inv = inverse_operator(SC, solver_SB, PreconditionIdentity());

    v.block(0) = 0.;
    v.block(1) = 0.;
    v.block(2) = 0.;

    std::cout << "v2" << std::endl;
    v.block(2) = -gamma * invW * u.block(2);
    // v.block(2) = SC_inv * u.block(2);
    std::cout << "v1" << std::endl;
    v.block(1) = -gamma * Mp_inv * u.block(1);
    // v.block(1) = SB_inv * u.block(1);
    std::cout << "v0" << std::endl;
    v.block(0) = Aug_inv * (u.block(0) - Bt * v.block(1) - Ct * v.block(2));
  }

  LinearOperator<Vector<double>> K;
  LinearOperator<Vector<double>> Aug_inv;
  LinearOperator<Vector<double>> B;
  LinearOperator<Vector<double>> Bt;
  LinearOperator<Vector<double>> SB;
  mutable LinearOperator<Vector<double>> SB_inv;
  LinearOperator<Vector<double>> C;
  LinearOperator<Vector<double>> invW;
  LinearOperator<Vector<double>> Mp_inv;
  LinearOperator<Vector<double>> Ct;
  double gamma;
};

#endif