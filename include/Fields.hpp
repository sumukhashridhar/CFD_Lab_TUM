#pragma once

#include "Datastructures.hpp"
#include "Discretization.hpp"
#include "Grid.hpp"

#include <vector>


/**
 * @brief Class of container and modifier for the physical fields
 *
 */
class Fields {
  public:
    Fields() = default;

    /**
     * @brief Constructor for the fields
     *
     * @param[in] kinematic viscosity
     * @param[in] initial timestep size
     * @param[in] adaptive timestep coefficient
     * @param[in] number of cells in x direction
     * @param[in] number of cells in y direction
     * @param[in] initial x-velocity
     * @param[in] initial y-velocity
     * @param[in] initial pressure
     * @param[in] initial temperature
     * @param[in] thermal diffusivity - alpha
     * @param[in] thermal expansion co-efficient - beta
     *
     */
    Fields(dtype _nu, dtype _dt, dtype _tau, int imax, int jmax, dtype UI, dtype VI, dtype PI, dtype TI, const Grid &grid, dtype _alpha, dtype _beta, bool _isHeatTransfer, dtype gx, dtype gy);

    /**
     * @brief Calculates the convective and diffusive fluxes in x and y
     * direction based on explicit discretization of the momentum equations
     *
     * @param[in] grid in which the fluxes are calculated
     *
     */
    void calculate_fluxes(Grid &grid);

    /**
     * @brief Right hand side calculations using the fluxes for the pressure
     * Poisson equation
     *
     * @param[in] grid in which the calculations are done
     *
     */
    void calculate_rs(Grid &grid);

    /**
     * @brief Velocity calculation using pressure values
     *
     * @param[in] grid in which the calculations are done
     *
     */
    void calculate_velocities(Grid &grid);

    /**
     * @brief Temperature calculation 
     *
     * @param[in] grid in which the calculations are done
     *
     */
    void calculate_temperatures(Grid &grid);

    /**
     * @brief Adaptive step size calculation using x-velocity condition,
     * y-velocity condition and CFL condition
     *
     * @param[in] grid in which the calculations are done
     *
     */
    dtype calculate_dt(Grid &grid);

    /// x-velocity index based access and modify
    dtype &u(int i, int j);

    /// y-velocity index based access and modify
    dtype &v(int i, int j);

    /// pressure index based access and modify
    dtype &p(int i, int j);

    /// temperature index based access and modify
    dtype &t(int i, int j);

    /// RHS index based access and modify
    dtype &rs(int i, int j);

    /// x-momentum flux index based access and modify
    dtype &f(int i, int j);

    /// y-momentum flux index based access and modify
    dtype &g(int i, int j);

    /// get timestep size
    dtype* dt();
    /// pressure matrix access and modify
    Matrix<dtype> &p_matrix();

    /// Temperature matrix access and modify
    Matrix<dtype> &t_matrix();

    Matrix<dtype> &u_matrix();

    Matrix<dtype> &v_matrix();

    Matrix<dtype> &f_matrix();

    Matrix<dtype> &g_matrix();

    Matrix<dtype> &rs_matrix();

    /// function to check if heat transfer occurs
    bool isHeatTransfer();

    dtype get_alpha();
    dtype get_nu();
    dtype get_beta();
    dtype get_tau();

    dtype get_gx();
    dtype get_gy();

  private:
    /// x-velocity matrix
    Matrix<dtype> _U;
    /// y-velocity matrix
    Matrix<dtype> _V;
    /// pressure matrix
    Matrix<dtype> _P;
    /// x-momentum flux matrix
    Matrix<dtype> _F;
    /// y-momentum flux matrix
    Matrix<dtype> _G;
    /// right hand side matrix
    Matrix<dtype> _RS;
    // Temperature Matrix
    Matrix<dtype> _T;

    /// kinematic viscosity
    dtype _nu;
    /// gravitional accelearation in x direction
    dtype _gx{0.0};
    /// gravitional accelearation in y direction
    dtype _gy{0.0};
    /// timestep size
    dtype _dt;
    /// adaptive timestep coefficient
    dtype _tau;
    /// thermal diffusivity alpha
    dtype _alpha;
    /// thermal expansion co-efficient
    dtype _beta;
    /// Check for heat transfer
    bool _isHeatTransfer;

};
