#include "Fields.hpp"

#include <algorithm>
#include <iostream>
#include "Communication.hpp"

Fields::Fields(double nu, double dt, double tau, int imax, int jmax, double UI, double VI, double PI, double TI, const Grid &grid, double alpha, double beta, bool isHeatTransfer, double gx, double gy)
    : _nu(nu), _dt(dt), _tau(tau),_alpha(alpha), _beta(beta), _isHeatTransfer(isHeatTransfer), _gx(gx), _gy(gy) {

    _U = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _V = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _P = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _F = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _G = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _RS = Matrix<double>(imax + 2, jmax + 2, 0.0);

    if (_isHeatTransfer){
        _T = Matrix<double>(imax + 2, jmax + 2, TI);
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
     
    Communication::communicate(_F);
    Communication::communicate(_G);
    
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

    Communication::communicate(_U);
    Communication::communicate(_V);
    for (auto currentCell : grid.fluid_cells()){
        int i = currentCell->i();
        int j = currentCell->j();
        if ((currentCell->neighbour(border_position::RIGHT)->type() == cell_type::FLUID) || (currentCell->neighbour(border_position::RIGHT)->type() == cell_type::OUTFLOW) || (currentCell->neighbour(border_position::RIGHT)->type() == cell_type::HALO)) {
            _U(i, j) = _F(i, j) - (_dt/grid.dx()) * (_P(i + 1, j) - _P(i, j));           
        }
        if ((currentCell->neighbour(border_position::TOP)->type() == cell_type::FLUID) || (currentCell->neighbour(border_position::TOP)->type() == cell_type::OUTFLOW) || (currentCell->neighbour(border_position::TOP)->type() == cell_type::HALO)) {
            _V(i, j) = _G(i, j) - (_dt/grid.dy()) * (_P(i, j + 1) - _P(i, j));
        }
    }


}

/*****************************************************************************************
 * This function calculate timestep for adaptive time stepping *
 ****************************************************************************************/

double Fields::calculate_dt(Grid &grid) {
    // Stability constraint for explicit time stepping according to equation (22)
    double t1 = 1 / (2 * _nu * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));
    double u_max = 0, v_max = 0, temp, umax, vmax;
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
    double t4 = 1 / (2 * _alpha * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));

    // std::cout << "Rank " << Communication::get_rank() << " was here\n";
    // if (Communication::get_rank() == 0) {

        // std::cout << "Rank " << Communication::get_rank() << " was here\n";

        umax = Communication::reduce_max(u_max);
        vmax = Communication::reduce_max(v_max);
        // std::cout << "Rank was here after reduce\n";

        MPI_Bcast(&umax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&vmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // std::cout << "Rank was here after Bcast\n";


    //Communication::broadcast(umax);
    //Communication::broadcast(vmax);

    // }
    // std::cout << "Rank " << Communication::get_rank() << " was here before barrier after bcast\n";
    // MPI_Barrier(MPI_COMM_WORLD);

    // std::cout << "Rank " << Communication::get_rank() << " was here after barrier\n";

    
    double t2 = grid.dx() / umax;
    double t3 = grid.dy() / vmax; 

    // move up t4, calc min(dx&dy) for each process max(umax&vmax)
    // for each process and find out t2&t3
    // but this will be done for eah process which doesnt make sense

    // Courant Number limitation t2,t3 according to equation (22)
  
    // Stability constraint for explicit time stepping according to equation (37)

    _dt = _tau * std::min({t1, t2, t3, t4});

    return _dt;
}

/*****************************************************************************************
 * This function calculate temperatures according to equation (36) *
 ****************************************************************************************/

void Fields::calculate_temperatures(Grid &grid)
{
    Communication::communicate(_T);

    double imaxb = grid.imaxb();
    double jmaxb = grid.jmaxb();
    Matrix<double> T_temp(imaxb, jmaxb, 0);
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

double &Fields::p(int i, int j) { return _P(i, j); }
double &Fields::u(int i, int j) { return _U(i, j); }
double &Fields::v(int i, int j) { return _V(i, j); }
double &Fields::f(int i, int j) { return _F(i, j); }
double &Fields::g(int i, int j) { return _G(i, j); }
double &Fields::rs(int i, int j) { return _RS(i, j); }
double &Fields::t(int i, int j) { return _T(i,j);}

Matrix<double> &Fields::p_matrix() { return _P; }

Matrix<double> &Fields::u_matrix() { return _U; }

Matrix<double> &Fields::v_matrix() { return _V; }


bool Fields::isHeatTransfer() { return _isHeatTransfer;}

double Fields::dt() const { return _dt; }
