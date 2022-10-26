#pragma once

#include "Datastructures.hpp"

/**
 * @brief Static discretization methods to modify the fields
 *
 */
class Discretization {
  public:
    Discretization() = default;

    /**
     * @brief Constructor to set the discretization parameters
     *
     * @param[in] cell size in x direction
     * @param[in] cell size in y direction
     * @param[in] upwinding coefficient
     */
    Discretization(dtype dx, dtype dy, dtype gamma);

    /**
     * @brief Diffusion discretization in 2D using central differences
     *
     * @param[in] data to be discretized
     * @param[in] x index
     * @param[in] y index
     *
     */
    static dtype diffusion(const Matrix<dtype> &A, int i, int j);

    /**
     * @brief Convection in x direction using donor-cell scheme
     *
     * @param[in] x-velocity field
     * @param[in] y-velocity field
     * @param[in] x index
     * @param[in] y index
     * @param[out] result
     *
     */
    static dtype convection_u(const Matrix<dtype> &U, const Matrix<dtype> &V, int i, int j);

    /**
     * @brief Convection in y direction using donor-cell scheme
     *
     * @param[in] x-velocity field
     * @param[in] y-velocity field
     * @param[in] x index
     * @param[in] y index
     * @param[out] result
     *
     */
    static dtype convection_v(const Matrix<dtype> &U, const Matrix<dtype> &V, int i, int j);

    /**
     * @brief Convection of Temperature in x direction using donor-cell scheme
     *
     * @param[in] Temperature field
     * @param[in] x-velocity field
     * @param[in] x index
     * @param[in] y index
     * @param[out] result
     *
     */
    static dtype convection_Tu(const Matrix<dtype> &T, const Matrix<dtype> &U, int i, int j);


    /**
     * @brief Convection of Temperature in y direction using donor-cell scheme
     *
     * @param[in] Temperature field
     * @param[in] y-velocity field
     * @param[in] x index
     * @param[in] y index
     * @param[out] result
     *
     */
    static dtype convection_Tv(const Matrix<dtype> &T, const Matrix<dtype> &V, int i, int j);

    /**
     * @brief Laplacian term discretization using central difference
     *
     * @param[in] data to be discretized
     * @param[in] x index
     * @param[in] y index
     * @param[out] result
     *
     */
    static dtype laplacian(const Matrix<dtype> &P, int i, int j);

    /**
     * @brief Terms of laplacian needed for SOR, i.e. excluding unknown value at
     * (i,j)
     *
     * @param[in] data to be discretized
     * @param[in] x index
     * @param[in] y index
     * @param[out] result
     *
     */
    static dtype sor_helper(const Matrix<dtype> &P, int i, int j);

    /**
     * @brief Compute interpolated value in the middle between two grid points via linear interpolation.
     *
     * @param[in] A data to be interpolated
     * @param[in] i index of first value used for interpolation
     * @param[in] j index of first value used for interpolation
     * @param[in] i_offset defines index of the second value used for interpolation as i+i_offset
     * @param[in] j_offset defines index of the second value used for interpolation as j+j_offset
     * @param[out] result
     *
     */
    static dtype interpolate(const Matrix<dtype> &A, int i, int j, int i_offset, int j_offset);

    dtype get_gamma();

  private:
    static dtype _dx;
    static dtype _dy;
    static dtype _gamma;
};
