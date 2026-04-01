// Assemble the system in normal way
/* ---------------------------------------------------------------------

 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Najwa Alshehri, KAUST, 2023
 */

#include <deal.II/base/config.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_bubbles.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#include "assemble_coupling_mass_matrix_with_exact_intersections.h"
#include "compute_intersections.h"
#include "create_coupling_sparsity_pattern_with_exact_intersections.h"

using namespace dealii;

static constexpr double coefficient_omega = 1.0;  // beta_1
static constexpr double coefficient_omega2 = 1e5; // beta_2

// exact solutions and its gradients
// based on omega = B(0,2) in R^d and omega2 = B(0,1) in R^dim
// and the given beta_1 and beta_2
template <int dim> class ExactSolution : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double ExactSolution<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const {
  double return_value;

  auto r = p.norm();
  if (r <= 1) // inside domain omega2
  {
    return_value = (3. * coefficient_omega2 / coefficient_omega + 1. - r * r) /
                   (2. * dim * coefficient_omega2);
  } else {
    return_value = (4. - r * r) / (2. * dim * coefficient_omega);
  }

  return return_value;
}

template <int dim>
Tensor<1, dim> // Tensor<1,dim> is a vector (a rank-1 tensor) in R^dim
ExactSolution<dim>::gradient(const Point<dim> &p,
                             const unsigned int /*component*/) const {
  auto r = p.norm();
  Tensor<1, dim> gradient;
  if (r <= 1) {
    for (int i = 0; i < dim; ++i) {
      gradient[i] = -p(i) / (dim * coefficient_omega2);
    }
  } else {
    for (int i = 0; i < dim; ++i) {
      gradient[i] = -p(i) / (dim * coefficient_omega);
    }
  }
  return gradient;
}

// This function must be of 2 components 0 is the solution and 1 is dummy
template <int dim> class ExactSolution2 : public Function<dim> {
public:
  ExactSolution2()
      : Function<dim>(2) // 2 components
  {}
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double ExactSolution2<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const {
  double return_value = 0.0;
  auto r = p.norm();
  if (component == 0) {
    return_value = (3. * coefficient_omega2 / coefficient_omega + 1. - r * r) /
                   (2. * dim * coefficient_omega2); // 1-10
  } else {
    return_value = -0;
  }
  return return_value;
}

template <int dim>
Tensor<1, dim>
ExactSolution2<dim>::gradient(const Point<dim> &p,
                              const unsigned int component) const {
  Tensor<1, dim> gradient;
  if (component == 0) {
    for (int i = 0; i < dim; ++i) {
      gradient[i] = -p(i) / (dim * coefficient_omega2);
    }
  }
  return gradient;
}

// define boundary values to be the exact solution itself
template <int dim> class BoundaryValues : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const {
  auto r = p.norm();
  return (4. - r * r) / (2. * dim * coefficient_omega);
}

// again, must be of 2 components 0 is the solution and 1 is dummy
template <int dim> class BoundaryValues2 : public Function<dim> {
public:
  BoundaryValues2()
      : Function<dim>(2) // 2 components
  {}
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues2<dim>::value(const Point<dim> &p,
                                   const unsigned int component) const {
  auto r = p.norm();
  double return_value = 0.0;
  if (component == 0) {
    return_value = (3. * coefficient_omega2 / coefficient_omega + 1. - r * r) /
                   (2. * dim * coefficient_omega2);
  }
  return return_value;
}

template <int dim> class InterfaceDLM {
public:
  InterfaceDLM();
  virtual ~InterfaceDLM() = default;
  void run();

private:
  void find_intersection();
  void make_grid_omega();
  void make_grid_omega2();
  void setup_system_omega();
  void setup_system_omega2();
  void setup_coupling();
  void assemble_system_omega();
  void assemble_system_omega2();
  void assemble_coupling_system();
  void solve();
  void exact_solution(
      const unsigned int cycle); // despite its name, it is for computing errors
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation_omega; // store mesh for Omega
  GridTools::Cache<dim, dim> omega_grid_tools_cache;
  Triangulation<dim> triangulation_omega2; // store mesh for Omega_2
  GridTools::Cache<dim, dim> omega2_grid_tools_cache;

  FE_Q<dim> fe;
  FESystem<dim> fe2; // FESystem is a combination of two FEs, one for Omega_2
                     // and one for coupling term
  DoFHandler<dim> omega_dh; // dh is short for DoFhandler
  DoFHandler<dim> omega2_dh;

  AffineConstraints<double>
      constraints; // for adaptive grid refinement for Omega
  AffineConstraints<double>
      constraints2; // for adaptive grid refinement for Omega2

  SparseMatrix<double> A_omega; // matrix A_1 in paper
  SparsityPattern sparsity_pattern_omega;

  SparseMatrix<double> B_omega2; // matrix B = [A_2, -C2T ; C1, -C2] in paper
  SparsityPattern sparsity_pattern_omega2;

  SparseMatrix<double> coupling_matrix; // matrix C1T in paper
  SparsityPattern coupling_sparsity;

  Vector<double> u_omega;  // to store the solution u_omega in the end
  Vector<double> u_omega2; // to store the solution u_omega2 in the end

  Vector<double>
      rhs_omega; // to store rhs, later will be copied into solution.rhs
  Vector<double> rhs_omega2; // same as above
  double u_l2_error;         // for errors calculation
  double u_l2_error2;
  double u_H1_error;
  double u_H1_error2;
  double u2_l2_error;
  double u2_l2_error2;
  double u2_H1_error;
  double u2_H1_error2;

  // coefficient_omega cannot be equal to coefficient_omega2
  // for elker condition coefficient_omega2 must be strictly greater than
  // coefficient_omega results show that in some cases this condition can be
  // weaken
  // double coefficient_omega = 1.0; // beta_1, in theory can be any
  // scalar-valued function, but here we let it be constant
  double rhs1 = 1.0; // the function f1 in paper, here we let it be constant
  // double coefficient_omega2 = 10.0; // beta_2
  double rhs2 = 3.0; // the function f2 in paper, here we let it be constant
  const FEValuesExtractors::Scalar primal;
  const FEValuesExtractors::Scalar multiplier;
  std::vector<std::tuple<typename Triangulation<dim, dim>::cell_iterator,
                         typename Triangulation<dim, dim>::cell_iterator,
                         Quadrature<dim>>>
      intersections_info; // to collect the intersection information

  ConvergenceTable convergence_table;
  ConvergenceTable convergence_table2;

  TimerOutput computing_timer;
};

template <int dim>
InterfaceDLM<dim>::InterfaceDLM()
    : omega_grid_tools_cache(triangulation_omega),
      omega2_grid_tools_cache(triangulation_omega2),
      fe(1) // let u has finite element Q1
      ,
      fe2(FE_Q<dim>(1), 1, FE_Q<dim>(1),
          1) // let [u_2, lambda] has finite element Q1B - P0
      ,
      omega_dh(triangulation_omega), omega2_dh(triangulation_omega2),
      primal(0) // index for "primal" variable, which is u_2
      ,
      multiplier(1) // index for "multiplier" variable, which is lambda
      ,
      computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times) {}

template <int dim> void InterfaceDLM<dim>::make_grid_omega() {
  GridGenerator::hyper_cube(
      triangulation_omega, -1.,
      1.); // creates omega to be [-1.4,1.4]^dim (1.4 is just sqrt(2))
}

template <int dim> void InterfaceDLM<dim>::make_grid_omega2() {
  GridGenerator::hyper_ball(
      triangulation_omega2, {0., 0.},
      0.22); // without 2nd and 3rd arguments, hyper_ball
             // defaults to using origin as center and radius 1
}

template <int dim> void InterfaceDLM<dim>::setup_system_omega() {
  omega_dh.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(omega_dh, constraints);

  VectorTools::interpolate_boundary_values(
      omega_dh, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();
  DynamicSparsityPattern dsp(omega_dh.n_dofs(), omega_dh.n_dofs());
  DoFTools::make_sparsity_pattern(omega_dh, dsp, constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern_omega.copy_from(dsp);
  A_omega.reinit(sparsity_pattern_omega);
  u_omega.reinit(omega_dh.n_dofs());
  rhs_omega.reinit(omega_dh.n_dofs());

  deallog << "Omega dofs: " << omega_dh.n_dofs() << std::endl;
}

template <int dim> void InterfaceDLM<dim>::setup_system_omega2() {
  omega2_dh.distribute_dofs(
      fe2); // since fe2 is a FESystem of [u_2, lambda], omega2_dh will
            // have (#dof of u_2 + #dof of lambda) dofs per cell
  DoFRenumbering::component_wise(omega2_dh);

  constraints2.clear();
  DoFTools::make_hanging_node_constraints(omega2_dh, constraints2);

  constraints2.close();
  DynamicSparsityPattern dsp(
      omega2_dh.n_dofs(),
      omega2_dh
          .n_dofs()); // this set up the size of (sparsity pattern of) matrix B
  DoFTools::make_sparsity_pattern(omega2_dh, dsp, constraints2,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern_omega2.copy_from(dsp);
  B_omega2.reinit(sparsity_pattern_omega2);
  u_omega2.reinit(omega2_dh.n_dofs());
  rhs_omega2.reinit(omega2_dh.n_dofs());
  deallog << "Omega2 dofs: " << omega2_dh.n_dofs() << std::endl;
}

template <int dim> void InterfaceDLM<dim>::find_intersection() {
  TimerOutput::Scope scope(computing_timer, "find_intersection");

  intersections_info = NonMatching::compute_intersection(
      omega_grid_tools_cache, omega2_grid_tools_cache,
      3,      // degree of quad
      1e-14); // tol
  double sum = 0.;
  for (const auto &v : intersections_info) {
    const auto qweights = std::get<2>(v);
    sum += std::accumulate(qweights.get_weights().begin(),
                           qweights.get_weights().end(), 0.);
  }

  Assert(
      std::abs(sum - GridTools::volume(triangulation_omega2)) < 1e-9,
      ExcMessage("Sum of weights is not equal to the measure of the immersed "
                 "domain Omega2. Check your intersection routine."));
  std::cout << "accuracy of intersection "
            << std::abs(sum - GridTools::volume(triangulation_omega2))
            << std::endl;
}

template <int dim> void InterfaceDLM<dim>::setup_coupling() {
  // TimerOutput::Scope scope(computing_timer, "setup_coupling");
  DynamicSparsityPattern dsp(
      omega_dh.n_dofs(),
      omega2_dh.n_dofs()); // this is the matrix [0 C1T] in paper, so it is of
  // size #dof of u x (#dof of u_2 + #dof of lambda)

  QGauss<dim> quad(3);
  NonMatching::create_coupling_sparsity_pattern(
      omega_dh, omega2_dh, quad, dsp, constraints,
      ComponentMask(), // for coupling u_omega
      ComponentMask(std::vector<bool>{false, true}), MappingQ1<dim>(),
      MappingQ1<dim>(), // false means take solution in
                        // Omega_2, true means takes lambda
      constraints2);
  // NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
  //     intersections_info, omega_dh, omega2_dh, dsp, constraints,
  //     ComponentMask(), // for coupling u_omega
  //     ComponentMask(
  //         std::vector<bool>{false, true}), // false means take solution in
  //                                          // Omega_2, true means takes
  //                                          lambda
  //     constraints2);

  coupling_sparsity.copy_from(dsp);
  coupling_matrix.reinit(coupling_sparsity);
  deallog << "setup coupling" << std::endl;
}

template <int dim> void InterfaceDLM<dim>::assemble_system_omega() {
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(
      dofs_per_cell,
      dofs_per_cell); // matrix of contribution to AA from each cell
  Vector<double> cell_rhs(
      dofs_per_cell); // matrix of contribution to RHS from each cell

  std::vector<types::global_dof_index> local_dof_indices(
      dofs_per_cell); // to store global indices of local dofs

  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  for (const auto &cell_omega : omega_dh.active_cell_iterators()) {
    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit(cell_omega);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices()) {
          cell_matrix(i, j) +=
              (coefficient_omega *                // a(x_q)
               fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
               fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
               fe_values.JxW(q_index));           // dx
        }
        cell_rhs(i) += ((fe_values.shape_value(i, q_index) * // phi_i(x_q)
                         rhs1)) *                            // f(x_q)
                       fe_values.JxW(q_index);               // dx
      }
      cell_omega->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, A_omega, rhs_omega);
    }
  }
}

template <int dim> void InterfaceDLM<dim>::assemble_system_omega2() {
  QGauss<2> quadrature_formula2(fe2.degree + 1);
  FEValues<2> fe_values2(fe2, quadrature_formula2,
                         update_values | update_quadrature_points |
                             update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe2.n_dofs_per_cell();

  FullMatrix<double> cell_matrix2(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs2(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : omega2_dh.active_cell_iterators()) {
    cell_matrix2 = 0;
    cell_rhs2 = 0;
    fe_values2.reinit(cell);

    for (const unsigned int q_index : fe_values2.quadrature_point_indices()) {
      for (const unsigned int i : fe_values2.dof_indices()) {
        for (const unsigned int j : fe_values2.dof_indices()) {
          cell_matrix2(i, j) +=
              (((coefficient_omega2 - coefficient_omega) * // b2(x_q)-b(x_q)
                fe_values2[primal].gradient(i,
                                            q_index) * // grad phi_i(x_q)_omega2
                fe_values2[primal].gradient(j,
                                            q_index)) // grad phi_j(x_q)_omega2

               - (fe_values2[primal].value(i,
                                           q_index) * // phi_i(x_q)_omega2
                  fe_values2[multiplier].value(j,
                                               q_index)) //  phi_j(x_q)_omega

               - (fe_values2[multiplier].value(i,
                                               q_index) * // phi_i(x_q)_omega
                  fe_values2[primal].value(j, q_index))   // phi_j(x_q)_omega2

               ) *
              fe_values2.JxW(q_index); // dx
        }
        cell_rhs2(i) +=
            (fe_values2[primal].value(i, q_index) * // phi_i(x_q)_omega2
             (rhs2 - rhs1)                          // f2(x_q)-f1(x_q)
             ) *
            fe_values2.JxW(q_index); // dx
      }
    }
    cell->get_dof_indices(local_dof_indices);
    constraints2.distribute_local_to_global(
        cell_matrix2, cell_rhs2, local_dof_indices, B_omega2, rhs_omega2);
  }
}

template <int dim> void InterfaceDLM<dim>::assemble_coupling_system() {
  TimerOutput::Scope scope(computing_timer, "Assemble_coupling_system");

  // assemble coupling_matrix
  // already done by the code from the header file
  // NonMatching::assemble_coupling_mass_matrix_with_exact_intersections(
  //     omega_dh, omega2_dh, intersections_info, coupling_matrix, constraints,
  //     ComponentMask(), ComponentMask(std::vector<bool>{false, true}),
  //     MappingQ1<dim>(), MappingQ1<dim>(), constraints2);

  QGauss<dim> quad(3);
  NonMatching::create_coupling_mass_matrix(
      omega_dh, omega2_dh, quad, coupling_matrix, constraints,
      ComponentMask(), // for coupling u_omega
      ComponentMask(
          std::vector<bool>{false, true}), // false means take solution in
                                           // Omega_2, true means takes lambda
      MappingQ1<dim>(), MappingQ1<dim>(), constraints2);
}

template <int dim> void InterfaceDLM<dim>::solve() {
  TimerOutput::Scope scope(computing_timer, "Solve");

  /// Start by creating the inverse stiffness matrix
  SparseDirectUMFPACK A_omega_inv_umfpack;
  A_omega_inv_umfpack.initialize(A_omega);
  SparseDirectUMFPACK B_omega2_inv_umfpack;
  B_omega2_inv_umfpack.initialize(B_omega2);

  // encapsulate the matrices into operators
  auto A1 = linear_operator(A_omega);
  auto B = linear_operator(B_omega2);
  auto C1t = linear_operator(coupling_matrix);
  auto C1 = transpose_operator(C1t);

  using BVec = BlockVector<double>; // shorthands
  using LinOp = decltype(A1);

  auto AA = block_operator<2, 2, BVec>(
      {{{{A1, C1t}}, {{C1, B}}}}); // assemble AA from A1, B, C1, C1t

  // for calculating preconditioners
  auto A1_inv = linear_operator(A1, A_omega_inv_umfpack);
  auto B_inv = linear_operator(B, B_omega2_inv_umfpack);
  auto X = -1.0 * B_inv * C1 * A1_inv;
  auto X2 = -1.0 * A1_inv * C1t * B_inv;

  auto low_tri_prec = block_operator<2, 2, BVec>(
      {{{{A1_inv, 0.0 * C1t}}, {{X, B_inv}}}}); // preconditioner 1

  auto up_tri_prec = block_operator<2, 2, BVec>(
      {{{{A1_inv, X2}}, {{0.0 * C1, B_inv}}}}); // preconditioner 2

  std::array<LinOp, 2> diag_ops = {{A1_inv, B_inv}};
  auto diagprecAA =
      block_diagonal_operator<2, BVec>(diag_ops); // preconditioner 3

  // SolverControl solver_control(200000, 1e-9);
  ReductionControl solver_control(1000, 1e-10, 1e-10, true, true);
  typename SolverGMRES<BVec>::AdditionalData data_fgmres;
  data_fgmres.max_basis_size = 50;
  data_fgmres.right_preconditioning = true;
  SolverGMRES<BVec> solver(solver_control, data_fgmres);

  BVec system_rhs;
  BVec solution;
  AA.reinit_domain_vector(system_rhs, false); // ???
  AA.reinit_range_vector(solution, false);

  system_rhs.block(0) = rhs_omega;
  system_rhs.block(1) = rhs_omega2;

  // solve the system
  solver.solve(AA, solution, system_rhs,
               up_tri_prec); // specify the preconditioner to use here

  std::cout << "   FGMRES converged in " << solver_control.last_step()
            << " iterations with residual " << solver_control.last_value()
            << std::endl;

  u_omega = solution.block(0);  // store the solution in u_omega
  u_omega2 = solution.block(1); // store the solution in u_omega2

  // "distribute" constraints, i.e., replace the wrong solution
  // at the constrained nodes by the correct solution computed from
  // unconstrained nodes
  constraints.distribute(u_omega);
  constraints2.distribute(u_omega2);
}

template <int dim>
void InterfaceDLM<dim>::exact_solution(const unsigned int cycle) {
  deallog << " Fe field function is ok" << std::endl;
  const ComponentSelectFunction<dim> primal_mask(0, 2);
  const ComponentSelectFunction<dim> multiplier_mask(1, 2);

  u_l2_error = 0.0;
  u_l2_error2 = 0.0;
  u_H1_error = 0.0;
  u_H1_error2 = 0.0;
  u2_l2_error = 0.0;
  u2_l2_error2 = 0.0;
  u2_H1_error = 0.0;
  u2_H1_error2 = 0.0;
  Vector<double> zero_vector1(omega_dh.n_dofs());
  Vector<double> zero_vector2(omega_dh.n_dofs());
  Vector<double> zero_vector3(omega2_dh.n_dofs());
  Vector<double> zero_vector4(omega2_dh.n_dofs());
  Vector<double> cellwise_errors(triangulation_omega.n_active_cells());
  Vector<double> cellwise_errors2(triangulation_omega2.n_active_cells());

  QTrapezoid<1> q_trapez;
  QIterated<dim> quadrature(q_trapez, 3);

  //----------  ||e_u||_L2 ----------------
  VectorTools::integrate_difference(omega_dh, u_omega,
                                    // fe_function,
                                    ExactSolution<dim>(), cellwise_errors,
                                    quadrature,
                                    // QGauss<dim>(3),
                                    VectorTools::L2_norm);
  deallog << " integrate difference" << std::endl;

  u_l2_error = VectorTools::compute_global_error(
      triangulation_omega, cellwise_errors, VectorTools::L2_norm);
  //----------  ||u||_L2 ----------------
  VectorTools::integrate_difference(
      omega_dh, zero_vector1, ExactSolution<dim>(), cellwise_errors, quadrature,
      // QGauss<dim>(3),
      VectorTools::L2_norm);

  u_l2_error2 = VectorTools::compute_global_error(
      triangulation_omega, cellwise_errors, VectorTools::L2_norm);

  std::cout << "||e_u||_L2/||u||_L2  = " << u_l2_error / u_l2_error2
            << std::endl;
  //----------  ||e_u||_H1 ----------------
  VectorTools::integrate_difference(omega_dh, u_omega,
                                    // fe_function,
                                    ExactSolution<dim>(), cellwise_errors,
                                    quadrature,
                                    // QGauss<dim>(3),
                                    VectorTools::H1_seminorm);

  u_H1_error = VectorTools::compute_global_error(
      triangulation_omega, cellwise_errors, VectorTools::H1_seminorm);
  //----------  ||u||_H1 ----------------
  VectorTools::integrate_difference(
      omega_dh, zero_vector2, ExactSolution<dim>(), cellwise_errors, quadrature,
      // QGauss<dim>(3),
      VectorTools::H1_seminorm);
  u_H1_error2 = VectorTools::compute_global_error(
      triangulation_omega, cellwise_errors, VectorTools::H1_seminorm);

  std::cout << "||e_u||_H1/||u||_H1  = " << u_H1_error / u_H1_error2
            << std::endl;
  //----------  ||e_u2||_L2 ----------------
  VectorTools::integrate_difference(omega2_dh, u_omega2, ExactSolution2<dim>(),
                                    cellwise_errors2, quadrature,
                                    VectorTools::L2_norm, &primal_mask);
  u2_l2_error = VectorTools::compute_global_error(
      triangulation_omega2, cellwise_errors2, VectorTools::L2_norm);
  //----------  ||u2||_L2 ----------------
  VectorTools::integrate_difference(
      omega2_dh, zero_vector3, ExactSolution2<dim>(), cellwise_errors2,
      quadrature, VectorTools::L2_norm, &primal_mask);
  u2_l2_error2 = VectorTools::compute_global_error(
      triangulation_omega2, cellwise_errors2, VectorTools::L2_norm);

  std::cout << "||e_u2||_L2/||u2||_L2  = " << u2_l2_error / u2_l2_error2
            << std::endl;
  //----------  ||e_u2||_H1 ----------------
  VectorTools::integrate_difference(omega2_dh, u_omega2, ExactSolution2<dim>(),
                                    cellwise_errors2, quadrature,
                                    VectorTools::H1_norm, &primal_mask);

  u2_H1_error = VectorTools::compute_global_error(
      triangulation_omega2, cellwise_errors2, VectorTools::H1_norm);
  //----------  ||u2||_H1 ----------------
  VectorTools::integrate_difference(
      omega2_dh, zero_vector4, ExactSolution2<dim>(), cellwise_errors2,
      quadrature, VectorTools::H1_norm, &primal_mask);
  u2_H1_error2 = VectorTools::compute_global_error(
      triangulation_omega2, cellwise_errors2, VectorTools::H1_norm);

  std::cout << "||e_u2||_H1/||u2||_H1  = " << u2_H1_error / u2_H1_error2
            << std::endl;

  const unsigned int n_dofs = omega_dh.n_dofs();
  const unsigned int n_active_cells = triangulation_omega.n_active_cells();
  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("dofs1", n_dofs);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("exact_error_L2", u_l2_error / u_l2_error2);
  convergence_table.add_value("exact_error_H1", u_H1_error / u_H1_error2);
  const unsigned int n_dofs2 = omega2_dh.n_dofs();
  const unsigned int n_active_cells2 = triangulation_omega2.n_active_cells();
  convergence_table.add_value("dofs2", n_dofs2);
  convergence_table.add_value("cells2", n_active_cells2);
  convergence_table.add_value("exact_error2_L2", u2_l2_error / u2_l2_error2);
  convergence_table.add_value("exact_error2_H1", u2_H1_error / u2_H1_error2);
}

template <int dim>
void InterfaceDLM<dim>::output_results(const unsigned int cycle) const {
  {
    // MatrixOut     rhs_out;
    // std::ofstream matrix_output("rhs_omega-" + std::to_string(cycle) +
    // ".vtu"); matrix_out.build_patches(A_omega, "A_omega");
    // matrix_out.write_vtu(matrix_output);

    // MatrixOut     matrix_out;
    // std::ofstream matrix_output("A_omega-" + std::to_string(cycle) + ".vtu");
    // matrix_out.build_patches(A_omega, "A_omega");
    // matrix_out.write_vtu(matrix_output);

    GridOut grid_out;
    std::ofstream output("grid-" + std::to_string(cycle) + ".gnuplot");
    GridOutFlags::Gnuplot gnuplot_flags(false, 5);
    grid_out.set_flags(gnuplot_flags);
    MappingQ<dim> mapping(3);
    grid_out.write_gnuplot(triangulation_omega, output, &mapping);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(omega_dh);
    data_out.add_data_vector(u_omega, "u_omega");
    data_out.build_patches();

    std::ofstream output1("u_omega-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output1);

    DataOut<dim> data2_out;
    data2_out.attach_dof_handler(omega2_dh);
    data2_out.add_data_vector(u_omega2, "u_omega2");
    data2_out.build_patches();

    std::ofstream output2("u_omega2-" + std::to_string(cycle) + ".vtu");
    data2_out.write_vtu(output2);
  }
}

template <int dim> void InterfaceDLM<dim>::run() {
  deallog.depth_console(10);
  deallog.push("RUN");
  // mesh size = sqrt(area of domain / number of cells)

  std::ofstream outfile("error_u.txt");
  std::ofstream outfile2("error_u2.txt");

  for (unsigned int cycle = 0; cycle < 12; ++cycle) {
    deallog << "Cycle " << cycle << std::endl;
    if (cycle == 0) {
      make_grid_omega();
      triangulation_omega.refine_global(4);
      make_grid_omega2();
      triangulation_omega2.refine_global(0); // 4
      std::cout << "h=" << triangulation_omega.begin_active()->diameter()
                << std::endl;
      std::cout << "h2=" << triangulation_omega2.begin_active()->diameter()
                << std::endl;
      std::cout << "h/h2="
                << triangulation_omega.begin_active()->diameter() /
                       triangulation_omega2.begin_active()->diameter()
                << std::endl;
    } else {
      triangulation_omega.refine_global(1);
      triangulation_omega2.refine_global(1);
    }
    find_intersection();
    setup_system_omega();
    setup_system_omega2();

    setup_coupling();

    assemble_system_omega();
    assemble_system_omega2();
    assemble_coupling_system();

    solve();

    exact_solution(cycle);
    outfile << omega_dh.n_dofs() << " " << u_l2_error / u_l2_error2 << " "
            << u_H1_error / u_H1_error2
            << triangulation_omega.begin_active()->diameter() << std::endl;

    outfile2 << omega2_dh.n_dofs() - triangulation_omega2.n_active_cells()
             << " " << u2_l2_error / u2_l2_error2 // L2norm
             << " " << u2_H1_error / u2_H1_error2 // H1 norm
             << " " << triangulation_omega2.n_active_cells() << std::endl;

    output_results(cycle);
    convergence_table.write_text(std::cout);

    computing_timer.print_summary();
    computing_timer.reset();
  }
  deallog.pop();

  outfile.close();
  outfile2.close();
}

int main() {
  try {
    InterfaceDLM<2> laplace_problem_2d;
    laplace_problem_2d.run();
  }

  catch (std::exception &exc) {
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