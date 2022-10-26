#include "Fields.hpp"

#include <algorithm>
#include <iostream>

Fields::Fields(dtype nu, dtype dt, dtype tau, int imax, int jmax, dtype UI, dtype VI, dtype PI, dtype TI, const Grid &grid, dtype alpha, dtype beta, bool isHeatTransfer, dtype gx, dtype gy)
    : _nu(nu), _dt(dt), _tau(tau),_alpha(alpha), _beta(beta), _isHeatTransfer(isHeatTransfer), _gx(gx), _gy(gy) {

    _U = Matrix<dtype>(imax + 2, jmax + 2, 0.0);
    _V = Matrix<dtype>(imax + 2, jmax + 2, 0.0);
    _P = Matrix<dtype>(imax + 2, jmax + 2, 0.0);
    _F = Matrix<dtype>(imax + 2, jmax + 2, 0.0);
    _G = Matrix<dtype>(imax + 2, jmax + 2, 0.0);
    _RS = Matrix<dtype>(imax + 2, jmax + 2, 0.0);

    if (_isHeatTransfer){
        _T = Matrix<dtype>(imax + 2, jmax + 2, TI);
    }

    for (auto currentCell : grid.fluid_cells()){
        int i = currentCell->i();
        int j = currentCell->j();
        _U(i,j) = UI;
        _V(i,j) = VI;
        _P(i,j) = PI;
    }
}

/********************************************************************************
 * This function calculates fluxes F and G as mentioned in equation (9) and (10)
 *******************************************************************************/

void Fields::calculate_fluxes(Grid &grid) {
     
    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        _F(i,j) = _U(i,j) + _dt * (_nu*Discretization::diffusion(_U,i,j) - Discretization::convection_u(_U,_V,i,j) + _gx);
        _G(i,j) = _V(i,j) + _dt * (_nu*Discretization::diffusion(_V,i,j) - Discretization::convection_v(_U,_V,i,j) + _gy);
    }

    /********************************************************************************
     * This section modifies fluxes F and G for heat transfer as mentioned in equation (32) and (33)
    *******************************************************************************/

    if(_isHeatTransfer){
        for (auto currentCell : grid.fluid_cells()) {
            int i = currentCell->i();
            int j = currentCell->j();
            _F(i,j) = _F(i,j) - _beta * _dt / 2.0 * (_T(i,j) + _T(i + 1, j)) * _gx - _dt * _gx;
            _G(i,j) = _G(i,j) - _beta * _dt / 2.0 * (_T(i,j) + _T(i, j + 1)) * _gy - _dt * _gy;
        }
    }
    
    for (auto currentCell: grid.fixed_wall_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        // B_NE fixed wall corner cell with fluid cells on the North and East directions 

        if(currentCell->is_border(border_position::TOP) && currentCell->is_border(border_position::RIGHT)){
            _F(i, j) = _U(i, j);
            _G(i, j) = _V(i, j);
        }

        // B_SE fixed wall corner cell with fluid cells on the South and East directions 

       else if(currentCell->is_border(border_position::BOTTOM) && currentCell->is_border(border_position::RIGHT)){
            _F(i, j) = _U(i, j);
            _G(i,j - 1) = _V(i,j - 1);

        }

        // B_NW fixed wall corner cell with fluid cells on the North and West directions 

        else if(currentCell->is_border(border_position::TOP) && currentCell->is_border(border_position::LEFT)){
            _F(i - 1, j) = _U(i - 1, j);
            _G(i, j) = _V(i, j);
        }

        // B_SW fixed wall corner cell with fluid cells on the South and West directions 

        else if(currentCell->is_border(border_position::BOTTOM) && currentCell->is_border(border_position::LEFT)){
            _F(i - 1, j) = _U(i - 1, j);
            _G(i, j - 1) = _V(i, j - 1);
        }
        else if(currentCell -> is_border(border_position::TOP))
            _G(i,j) = _V(i,j);

        else if(currentCell -> is_border(border_position::BOTTOM))
            _G(i,j - 1) = _V(i,j - 1);

        else if(currentCell -> is_border(border_position::LEFT))
            _F(i - 1, j) = _U(i - 1, j);

        else if(currentCell -> is_border(border_position::RIGHT))
            _F(i, j) = _U(i, j);

    }

    for (auto currentCell: grid.moving_wall_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        _G(i,j - 1) = _V(i,j - 1);
    }

    for (auto currentCell: grid.inflow_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        _F(i,j) = _U(i,j);
    }

    for (auto currentCell: grid.outflow_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        _F(i - 1,j) = _U(i - 1,j);
    }
}

/********************************************************************************
 * This function calculates the RHS of equation (11) i.e. Pressure SOR
 *******************************************************************************/

void Fields::calculate_rs(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        _RS(i, j) = 1 / _dt * ((_F(i, j) - _F(i - 1, j)) / grid.dx() + 
                               (_G(i, j) - _G(i, j - 1)) / grid.dy()); 
    }
}

/*****************************************************************************************
 * This function updates velocity after Pressure SOR as mentioned in equation (7) and (8)
 ****************************************************************************************/
void Fields::calculate_velocities(Grid &grid) {

    for (auto currentCell : grid.fluid_cells()){
        int i = currentCell->i();
        int j = currentCell->j();
        if ((currentCell->neighbour(border_position::RIGHT)->type() == cell_type::FLUID) || (currentCell->neighbour(border_position::RIGHT)->type() == cell_type::OUTFLOW)) {
            _U(i, j) = _F(i, j) - (_dt/grid.dx()) * (_P(i + 1, j) - _P(i, j));           
        }
        if ((currentCell->neighbour(border_position::TOP)->type() == cell_type::FLUID) || (currentCell->neighbour(border_position::TOP)->type() == cell_type::OUTFLOW)) {
            _V(i, j) = _G(i, j) - (_dt/grid.dy()) * (_P(i, j + 1) - _P(i, j));
        }
    }
}

/*****************************************************************************************
 * This function calculate timestep for adaptive time stepping *
 ****************************************************************************************/

dtype Fields::calculate_dt(Grid &grid) {
    // Stability constraint for explicit time stepping according to equation (22)
    dtype t1 = 1 / (2 * _nu * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));
    dtype u_max = 0, v_max = 0, temp;
    for (int i = 0; i < grid.imaxb(); ++i){
        for(int j=0;j<grid.jmaxb();++j)
        {
            temp = std::abs(_U(i,j));
            if(temp > u_max){
                u_max = temp;
            }
            temp = std::abs(_V(i,j));
            if(temp > v_max){
                v_max = temp;
            }
        }
    }

    // Courant Number limitation t2,t3 according to equation (22)
    dtype t2 = grid.dx() / u_max;
    dtype t3 = grid.dy() / v_max;   
    // Stability constraint for explicit time stepping according to equation (37)
    dtype t4 = 1 / (2 * _alpha * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));
    _dt = _tau * std::min({t1, t2, t3, t4});
    return _dt;
}

/*****************************************************************************************
 * This function calculate temperatures according to equation (36) *
 ****************************************************************************************/

void Fields::calculate_temperatures(Grid &grid)
{

    dtype imaxb = grid.imaxb();
    dtype jmaxb = grid.jmaxb();
    Matrix<dtype> T_temp(imaxb, jmaxb, 0);
    for(auto i = 0; i < imaxb; i++)
    {
        for(auto j = 0; j < jmaxb; j++)
        {
            T_temp(i,j) = _T(i,j);
        }
    }

    for(auto currentCell: grid.fluid_cells())
    {
        int i = currentCell->i();
        int j = currentCell->j();
        _T(i,j) = _dt * (_alpha * Discretization::diffusion(T_temp,i,j) - Discretization::convection_Tu(T_temp,_U,i,j) - Discretization::convection_Tv(T_temp,_V,i,j)) + T_temp(i,j);
    }
    
}

dtype &Fields::p(int i, int j) { return _P(i, j); }
dtype &Fields::u(int i, int j) { return _U(i, j); }
dtype &Fields::v(int i, int j) { return _V(i, j); }
dtype &Fields::f(int i, int j) { return _F(i, j); }
dtype &Fields::g(int i, int j) { return _G(i, j); }
dtype &Fields::rs(int i, int j) { return _RS(i, j); }
dtype &Fields::t(int i, int j) { return _T(i,j);}

Matrix<dtype> &Fields::p_matrix() { return _P; }

Matrix<dtype> &Fields::t_matrix() { return _T; }

Matrix<dtype> &Fields::u_matrix() { return _U; }

Matrix<dtype> &Fields::v_matrix() { return _V; }

Matrix<dtype> &Fields::f_matrix() { return _F; }

Matrix<dtype> &Fields::g_matrix() { return _G; }

Matrix<dtype> &Fields::rs_matrix() { return _RS; }

bool Fields::isHeatTransfer() { return _isHeatTransfer;}

dtype* Fields::dt() { return &_dt; }

dtype Fields::get_alpha() { return _alpha;}

dtype Fields::get_nu() { return _nu;}

dtype Fields::get_beta() { return _beta;}

dtype Fields::get_tau() { return _tau;}

dtype Fields::get_gx() { return _gx;}

dtype Fields::get_gy() { return _gy;}
