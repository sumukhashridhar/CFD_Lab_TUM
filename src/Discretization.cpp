#include "Discretization.hpp"

#include <cmath>

dtype Discretization::_dx = 0.0;
dtype Discretization::_dy = 0.0;
dtype Discretization::_gamma = 0.0;

Discretization::Discretization(dtype dx, dtype dy, dtype gamma) {
    _dx = dx;
    _dy = dy;
    _gamma = gamma;
}

/***********************************************************************
The following function returns the convection due to u in equation (9)
The discretization for the 2 terms are found in first 2 equations of (4)
***********************************************************************/

dtype Discretization::convection_u(const Matrix<dtype> &U, const Matrix<dtype> &V, int i, int j) {
    dtype du2_dx = 1/ _dx * (pow(interpolate(U,i,j,1,0), 2) - pow(interpolate(U,i,j,-1,0), 2)) +
                        _gamma/_dx * ((std::abs(interpolate(U,i,j,1,0)) * (U(i, j) - U(i + 1, j)) / 2) -
                        std::abs(interpolate(U,i,j,-1,0))*(U(i - 1, j)-U(i, j)) / 2) ;
    dtype duv_dy = 1/ _dy * (((interpolate(V,i,j,1,0)) * (interpolate(U,i,j,0,1)))  - ( (interpolate(V,i,j-1,1,0)) * (interpolate(U,i,j,0,-1)))  ) +
                            _gamma / _dy * ( (std::abs(interpolate(V,i,j,1,0))*(U(i, j) - U(i, j + 1)) / 2) - 
                            (std::abs(interpolate(V,i,j-1,1,0)) * (U(i, j - 1) - U(i, j)) / 2 ));    

    dtype result = du2_dx + duv_dy;
    return result;
}

/***********************************************************************
The following function returns the convection due to v in equation (10)
The discretization for the 2 terms are found in first 2 equations of (5)
***********************************************************************/

dtype Discretization::convection_v(const Matrix<dtype> &U, const Matrix<dtype> &V, int i, int j) {
    dtype dv2_dy = 1/ _dy * (pow(interpolate(V,i,j,0,1), 2) - pow(interpolate(V,i,j-1,0,1), 2)) +
                        _gamma/_dy * ((std::abs(interpolate(V,i,j,0,1)) * (V(i, j) - V(i, j+1)) / 2) -
                        std::abs(interpolate(V, i, j - 1, 0, 1)) * (V(i, j-1)- V(i, j)) / 2) ;
    dtype duv_dx = 1/ _dx * (((interpolate(U, i, j, 0, 1)) * (interpolate(V, i, j, 1, 0)))  - ( (interpolate(U,i-1,j,0,1)) * (interpolate(V,i-1,j,1,0)))) +
                            _gamma / _dx * ( (std::abs(interpolate(U,i,j,0,1))*(V(i, j) - V(i + 1, j)) / 2) - 
                            (std::abs(interpolate(U,i-1,j,0,1)) * (V(i-1, j) - V(i, j)) / 2 ));    

    dtype result = dv2_dy + duv_dx;
    return result;
}

/*****************************************************************************************
The following function returns the diffusion due to u or v in equation (9)/(10)
The discretization for the 2 terms are found in third and fourth  equation of (4)/(5)
******************************************************************************************/
dtype Discretization::diffusion(const Matrix<dtype> &A, int i, int j) {
    dtype result = (A(i + 1, j) - 2.0 * A(i, j) + A(i - 1, j)) / (_dx * _dx) +
                    (A(i, j + 1) - 2.0 * A(i, j) + A(i, j - 1)) / (_dy * _dy);

    return result;
}

/*****************************************************************************************
The following function returns the convection of Temperature in x direction as per 
the first part of equation (36)
******************************************************************************************/

dtype Discretization::convection_Tu(const Matrix<dtype> &T, const Matrix<dtype> &U, int i, int j)
{
    dtype result;
    result = 1/_dx * ( U(i, j) * interpolate(T,i,j,1,0) - U(i - 1,j) * interpolate(T,i-1,j,1,0)) + 
                        _gamma/_dx * ( std::abs(U(i,j)) * (T(i, j) - T(i + 1, j)) / 2 - std::abs(U(i - 1,j)) * (T(i - 1, j) - T(i, j)) / 2 );
    return result;
}

/*****************************************************************************************
The following function returns the convection of Temperature in y direction as per 
the second part of equation (36)
******************************************************************************************/

dtype Discretization::convection_Tv(const Matrix<dtype> &T, const Matrix<dtype> &V, int i, int j)
{
    dtype result;
    result = 1/_dy * ( V(i, j) * interpolate(T,i,j,0,1) - V(i,j - 1) * interpolate(T,i,j-1,0,1) ) + 
                        _gamma/_dy * ( std::abs(V(i,j)) * (T(i, j) - T(i, j + 1)) / 2 - std::abs(V(i,j - 1)) * (T(i, j - 1) - T(i, j))/2 );
    return result;
}

dtype Discretization::laplacian(const Matrix<dtype> &P, int i, int j) {
    dtype result = (P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) / (_dx * _dx) +
                    (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) / (_dy * _dy);
    return result;
}

dtype Discretization::sor_helper(const Matrix<dtype> &P, int i, int j) {
    dtype result = (P(i + 1, j) + P(i - 1, j)) / (_dx * _dx) + (P(i, j + 1) + P(i, j - 1)) / (_dy * _dy);
    return result;
}

dtype Discretization::interpolate(const Matrix<dtype> &A, int i, int j, int i_offset, int j_offset) {

    dtype result =( A(i, j) + A(i + i_offset, j + j_offset)) / 2;
    return result;


}

dtype Discretization::get_gamma() {return _gamma;}