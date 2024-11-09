#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

class RationalPreconditioner {
 public:
  RationalPreconditioner(const LinearOperator<Vector<double>> K_inv_,
                         const SparseMatrix<double> *const embedded_stiffness,
                         const SparseMatrix<double> *const embedded_mass,
                         const double rho_bound_) {
    K_inv = K_inv_;
    rho_bound = rho_bound_;
    // copy A and M
    A_immersed_matrix.reinit(*embedded_stiffness);
    A_immersed_matrix.copy_from(*embedded_stiffness);
    M_immersed_matrix.reinit(*embedded_mass);
    M_immersed_matrix.copy_from(*embedded_mass);
    A_immersed = linear_operator(*embedded_stiffness);
    M = linear_operator(*embedded_mass);
  }

  void vmult(BlockVector<double> &v, const BlockVector<double> &u) const {
    // First, solve upper block
    v.block(0) = K_inv * (u.block(0));

    TrilinosWrappers::PreconditionAMG prec_amg;
    SolverControl solver_control(2000, 1e-14, false, false);
    SolverCG<Vector<double>> solver_cg(solver_control);

    const unsigned int n_poles = poles.size();
    std::vector<Vector<double>> solutions;

    // Define AMG prec for each one of the shifted linear systems
    for (unsigned int i = 1; i < n_poles + 1; ++i) {
      SparseMatrix<double> matrix;
      matrix.reinit(A_immersed_matrix.get_sparsity_pattern());
      matrix.copy_from(A_immersed_matrix);
      matrix.add(-rho_bound * poles[i - 1], M_immersed_matrix);
      prec_amg.initialize(matrix);

      auto Ap = (A_immersed - rho_bound * poles[i - 1] * M);
      auto invAp = inverse_operator(Ap, solver_cg, prec_amg);

      solutions.push_back(rho_bound * res[i] * invAp * u.block(1));
    }

    auto invM = inverse_operator(M, solver_cg, PreconditionIdentity());
    Vector<double> first_term;
    first_term = res[0] * invM * u.block(1);
    solutions.push_back(first_term);

    Vector<double> sum_vecs(solutions[0].size());
    sum_vecs = 0.;
    for (auto &solution : solutions) sum_vecs += solution;
    v.block(1) = sum_vecs;
  }

 private:
  LinearOperator<Vector<double>> K_inv;
  LinearOperator<Vector<double>> M;
  TrilinosWrappers::PreconditionAMG prec_amg;

  static constexpr std::array<double, 21> res{
      {1.1133752551375149e+01,  -4.5192561264009555e+02,
       -5.4280235488093114e+00, -6.6119823627983498e-01,
       -1.5483255874020074e-01, -4.8435293477731435e-02,
       -1.7569986796633446e-02, -6.9011933591631392e-03,
       -2.8275585395562131e-03, -1.1823861060446343e-03,
       -4.9806992558149195e-04, -2.0975776516702764e-04,
       -8.7959042415258930e-05, -3.6650480089224726e-05,
       -1.5149104182285630e-05, -6.1866179967421625e-06,
       -2.4691626461139533e-06, -9.3898594542244485e-07,
       -3.2099152020952601e-07, -8.4169497470931466e-08,
       -7.7616172944516437e-09}};

  static constexpr std::array<double, 20> poles{
      {-4.9917060842594275e+01, -5.2698715191349796e+00,
       -1.7156755741861143e+00, -7.5569620064292298e-01,
       -3.7811376547012854e-01, -2.0130525955937850e-01,
       -1.1058502730933521e-01, -6.1664070123493613e-02,
       -3.4578652087400880e-02, -1.9394206381182760e-02,
       -1.0845568864180035e-02, -6.0343457447149737e-03,
       -3.3328397814762593e-03, -1.8198589302273998e-03,
       -9.7434812604726647e-04, -5.0332017175529794e-04,
       -2.4317839761161207e-04, -1.0297057301403903e-04,
       -3.2227929557637293e-05, -3.3293811779427837e-06}};
  double rho_bound;

  LinearOperator<Vector<double>> A_immersed;
  mutable SparseMatrix<double> A_immersed_matrix;
  mutable SparseMatrix<double> M_immersed_matrix;
};