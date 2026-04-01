// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Luca Heltai
//
// This file is part of the FSI-suite platform, based on the deal.II library.
//
// The FSI-suite platform is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3.0 of the License,
// or (at your option) any later version. The full text of the license can be
// found in the file LICENSE at the top level of the FSI-suite platform
// distribution.
//
// ---------------------------------------------------------------------
#ifndef compute_intersections_h
#define compute_intersections_h

#include <deal.II/base/config.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <set>
#include <tuple>
#include <vector>

// #include "moonolith_tools.h"

namespace dealii {
namespace NonMatching {
/**
 * @brief Intersect `cell0` and `cell1` and construct a
 *        `Quadrature<spacedim>` of degree `degree`` over the intersection,
 *        i.e. in the real space. Mappings for both cells are in `mapping0`
 *        and `mapping1`, respectively.
 *
 * @tparam dim0
 * @tparam dim1
 * @tparam spacedim
 * @param cell0 A `cell_iteratator` to the first cell
 * @param cell1 A `cell_iteratator` to the first cell
 * @param degree The degree of the `Quadrature` you want to build there
 * @param mapping0 The `Mapping` object describing the first cell
 * @param mapping1 The `Mapping` object describing the second cell
 * @return Quadrature<spacedim>
 */
template <int dim0, int dim1, int spacedim>
dealii::Quadrature<spacedim> compute_intersection(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int degree, const Mapping<dim0, spacedim> &mapping0,
    const Mapping<dim1, spacedim> &mapping1) {

  AssertThrow(false, ExcNotImplemented());
  // if constexpr ((dim0 == 1 && dim1 == 3) || (dim0 == 3 && dim1 == 1) ||
  //               (dim0 == 1 && dim1 == 1)) {
  //   (void)cell0;
  //   (void)cell1;
  //   (void)degree;
  //   (void)mapping0;
  //   (void)mapping1;
  //   AssertThrow(false, ExcNotImplemented());
  //   return dealii::Quadrature<spacedim>();
  // } else {
  //   return moonolith::compute_intersection(cell0, cell1, degree, mapping0,
  //                                          mapping1);
}

/**
 * @brief Given two triangulations cached inside `GridTools::Cache` objects,
 * compute all intersections between the two and return a vector where each
 * entry is a tuple containing iterators to the respective cells and a
 * `Quadrature<spacedim>` formula to integrate over the intersection.
 *
 * @tparam dim0 Intrinsic dimension of the immersed grid
 * @tparam dim1 Intrinsic dimension of the ambient grid
 * @tparam spacedim
 * @param space_cache
 * @param immersed_cache
 * @param degree Degree of the desired quadrature formula
 * @return std::vector<std::tuple< typename dealii::Triangulation<dim0,
 * spacedim>::active_cell_iterator, typename dealii::Triangulation<dim1,
 * spacedim>::active_cell_iterator, dealii::Quadrature<spacedim>>>
 */
template <int dim0, int dim1, int spacedim>
std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                       typename Triangulation<dim1, spacedim>::cell_iterator,
                       Quadrature<spacedim>>>
compute_intersection(const GridTools::Cache<dim0, spacedim> &space_cache,
                     const GridTools::Cache<dim1, spacedim> &immersed_cache,
                     const unsigned int degree, const double tol) {
  Assert(degree >= 1, ExcMessage("degree cannot be less than 1"));

  std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                         typename Triangulation<dim1, spacedim>::cell_iterator,
                         Quadrature<spacedim>>>
      cells_with_quads;

  const auto &space_tree =
      space_cache.get_locally_owned_cell_bounding_boxes_rtree();

  // The immersed tree *must* contain all cells, also the non-locally owned
  // ones.
  const auto &immersed_tree = immersed_cache.get_cell_bounding_boxes_rtree();

  // references to triangulations' info (cp cstrs marked as delete)
  const auto &mapping0 = space_cache.get_mapping();
  const auto &mapping1 = immersed_cache.get_mapping();
  namespace bgi = boost::geometry::index;
  // Whenever the BB space_cell intersects the BB of an embedded cell,
  // store the space_cell in the set of intersected_cells
  for (const auto &[immersed_box, immersed_cell] : immersed_tree) {
    for (const auto &[space_box, space_cell] :
         space_tree | bgi::adaptors::queried(bgi::intersects(immersed_box))) {
      const auto test_intersection = compute_intersection<dim0, dim1, spacedim>(
          space_cell, immersed_cell, degree, mapping0, mapping1);

      // if (test_intersection.get_points().size() !=
      const auto &weights = test_intersection.get_weights();
      const double area = std::accumulate(weights.begin(), weights.end(), 0.0);
      if (area > tol) // non-trivial intersection
      {
        cells_with_quads.push_back(
            std::make_tuple(space_cell, immersed_cell, test_intersection));
      }
    }
  }

  return cells_with_quads;
}

} // namespace NonMatching
} // namespace dealii
#endif