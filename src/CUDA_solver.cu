#include "CUDA_solver.cuh"
#include <cmath>
#include <iostream>
#include <iomanip> 
#include <math.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#define at(var, i, j) var[(j) * (*gpu_size_x) + (i)]

#define check_bound_north(j) ((j + 1) < *gpu_size_y)
#define check_bound_south(j) ((j - 1) >= 0)
#define check_bound_east(i) ((i + 1) < *gpu_size_x)
#define check_bound_west(i) ((i - 1) >=0)

__device__ dtype interpolate(dtype *A, int i, int j, int i_offset, int j_offset, int *gpu_size_x) {
    dtype result = (at(A, i, j) + at(A, i + i_offset, j + j_offset)) / 2;
    return result;
}

__device__ dtype diffusion(dtype *A, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {
    dtype result = (at(A, i + 1, j) - 2.0 * at(A, i, j) + at(A, i - 1, j)) / (gpu_dx * gpu_dx) +
                    (at(A, i, j + 1) - 2.0 * at(A, i, j) + at(A, i, j - 1)) / (gpu_dy * gpu_dy);

    return result;
}

__device__ dtype convection_u(dtype *gpu_U, dtype *gpu_V, dtype gpu_gamma, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {

    dtype t1 = interpolate(gpu_U, i, j, 1, 0, gpu_size_x);
    dtype t2 = interpolate(gpu_U, i, j, -1, 0, gpu_size_x);
    dtype du2_dx = 1 / gpu_dx * ((t1 * t1) - (t2 * t2)) + gpu_gamma / gpu_dx *\
            ((fabsf(interpolate(gpu_U, i, j, 1, 0, gpu_size_x)) * (at(gpu_U, i, j) - at(gpu_U, i + 1, j)) / 2) -\
             fabsf(interpolate(gpu_U, i, j, -1, 0, gpu_size_x)) * (at(gpu_U, i - 1, j) - at(gpu_U, i, j)) / 2);
    dtype duv_dy = 1 / gpu_dy * (((interpolate(gpu_V, i, j, 1, 0, gpu_size_x)) * (interpolate(gpu_U, i, j, 0, 1, gpu_size_x))) -\
                         ((interpolate(gpu_V, i, j - 1, 1, 0, gpu_size_x)) * (interpolate(gpu_U, i, j, 0, -1, gpu_size_x)))) +\
                    gpu_gamma / gpu_dy *\
                        ((fabsf(interpolate(gpu_V, i, j, 1, 0, gpu_size_x)) * (at(gpu_U, i, j) - at(gpu_U, i, j + 1)) / 2) -\
                         (fabsf(interpolate(gpu_V, i, j - 1, 1, 0, gpu_size_x)) * (at(gpu_U, i, j - 1) - at(gpu_U, i, j)) / 2));

    dtype result = du2_dx + duv_dy;
    return result;
}

__device__ dtype convection_v(dtype *gpu_U, dtype *gpu_V, dtype gpu_gamma, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {

    dtype t1 = interpolate(gpu_V, i, j, 0, 1, gpu_size_x);
    dtype t2 = interpolate(gpu_V, i, j - 1, 0, 1, gpu_size_x);
    dtype dv2_dy = 1 / gpu_dy * ((t1 * t1) - (t2 * t2)) + gpu_gamma / gpu_dy *\
            ((fabsf(interpolate(gpu_V, i, j, 0, 1, gpu_size_x)) * (at(gpu_V, i, j) - at(gpu_V, i, j + 1)) / 2) -\
             fabsf(interpolate(gpu_V, i, j - 1, 0, 1, gpu_size_x)) * (at(gpu_V, i, j - 1) - at(gpu_V, i, j)) / 2);
    dtype duv_dx = 1 / gpu_dx *\
                        (((interpolate(gpu_U, i, j, 0, 1, gpu_size_x)) * (interpolate(gpu_V, i, j, 1, 0, gpu_size_x))) -\
                         ((interpolate(gpu_U, i - 1, j, 0, 1, gpu_size_x)) * (interpolate(gpu_V, i - 1, j, 1, 0, gpu_size_x)))) +\
                    gpu_gamma / gpu_dx *\
                        ((fabsf(interpolate(gpu_U, i, j, 0, 1, gpu_size_x)) * (at(gpu_V, i, j) - at(gpu_V, i + 1, j)) / 2) -\
                         (fabsf(interpolate(gpu_U, i - 1, j, 0, 1, gpu_size_x)) * (at(gpu_V, i - 1, j) - at(gpu_V, i, j)) / 2));

    dtype result = dv2_dy + duv_dx;
    return result;
}

__device__ dtype convection_Tu(dtype *gpu_T, dtype *gpu_U, int i, int j, dtype gpu_dx, dtype gpu_dy, dtype gpu_gamma, int *gpu_size_x) {
    dtype result;
    result = 1 / gpu_dx * (at(gpu_U, i, j) * interpolate(gpu_T, i, j, 1, 0, gpu_size_x) - at(gpu_U, i - 1, j) * interpolate(gpu_T, i - 1, j, 1, 0, gpu_size_x)) +
             gpu_gamma / gpu_dx * (fabsf(at(gpu_U, i, j)) * (at(gpu_T, i, j) - at(gpu_T, i + 1, j)) / 2 - fabsf(at(gpu_U, i - 1, j)) * (at(gpu_T, i - 1, j) - at(gpu_T, i, j)) / 2);
    return result;
}

__device__ dtype convection_Tv(dtype *gpu_T, dtype *gpu_V, int i, int j, dtype gpu_dx, dtype gpu_dy, dtype gpu_gamma, int *gpu_size_x) {
    dtype result;
    result = 1 / gpu_dy * (at(gpu_V, i, j) * interpolate(gpu_T, i, j, 0, 1, gpu_size_x) - at(gpu_V, i, j - 1) * interpolate(gpu_T, i, j - 1, 0, 1, gpu_size_x)) +
             gpu_gamma / gpu_dy * (fabsf(at(gpu_V, i, j)) * (at(gpu_T, i, j) - at(gpu_T, i, j + 1)) / 2 - fabsf(at(gpu_V, i, j - 1)) * (at(gpu_T, i, j - 1) - at(gpu_T, i, j)) / 2);
    return result;
}

__device__ dtype sor_helper(dtype *gpu_P, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {
    dtype result = (at(gpu_P,i + 1, j) + at(gpu_P,i - 1, j)) / (gpu_dx * gpu_dx) + (at(gpu_P,i, j + 1) + at(gpu_P,i, j - 1)) / (gpu_dy * gpu_dy);
    return result;
}

__device__ dtype laplacian(dtype *gpu_P, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {
    dtype result = (at(gpu_P,i + 1, j) - 2.0 * at(gpu_P,i, j) + at(gpu_P,i - 1, j)) / (gpu_dx * gpu_dx) +
                   (at(gpu_P,i, j + 1) - 2.0 * at(gpu_P,i, j) + at(gpu_P,i, j - 1)) / (gpu_dy * gpu_dy);
    return result;
}

dim3 CUDA_solver::get_num_blocks(int size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

dim3 CUDA_solver::get_num_blocks_2d(int gpu_size_x, int gpu_size_y) {
    return (dim3((gpu_size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (gpu_size_y + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y));
}

__global__ void FixedWallBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, dtype *gpu_T, int *gpu_geometry_data,
int *gpu_fluid_id, int *gpu_moving_wall_id, int *gpu_fixed_wall_id, int *gpu_inflow_id, int *gpu_outflow_id, int *gpu_adiabatic_id, int *gpu_hot_id,
int *gpu_cold_id, dtype *gpu_wall_temp_a, dtype *gpu_wall_temp_h, dtype *gpu_wall_temp_c, bool *gpu_isHeatTransfer, int *gpu_size_x, int
*gpu_size_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && (at(gpu_geometry_data, i, j) == 3 || at(gpu_geometry_data, i, j) == 5 || at(gpu_geometry_data, i, j) == 6 || at(gpu_geometry_data, i, j) == 7)) {
        // obstacles B_NE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and East directions 

        if(check_bound_north(j) && check_bound_east(i) && at(gpu_geometry_data,i,j+1)==*gpu_fluid_id && at(gpu_geometry_data,i+1,j)==*gpu_fluid_id) {
            at(gpu_U, i, j) = 0.0;
            at(gpu_U, i - 1, j) = -at(gpu_U, i - 1, j + 1);
            at(gpu_V, i, j) = 0.0;
            at(gpu_V, i, j - 1) = -at(gpu_V, i + 1, j - 1);
            at(gpu_P, i, j) = (at(gpu_P, i, j + 1) + at(gpu_P, i + 1, j))/2;

            if(*gpu_isHeatTransfer==1) {
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T, i, j) = (at(gpu_T, i + 1, j) + at(gpu_T, i, j + 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j + 1) + at(gpu_T,i + 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j + 1) + at(gpu_T,i + 1, j) )/2;
            }
        }

        // obstacles B_SE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and East directions 

        else if(check_bound_south(j) && check_bound_east(i) && at(gpu_geometry_data,i,j-1)==*gpu_fluid_id&& at(gpu_geometry_data,i+1,j)==*gpu_fluid_id) {

            at(gpu_U, i, j) = 0.0;
            at(gpu_U, i - 1, j) = -at(gpu_U, i - 1, j - 1);
            at(gpu_V, i, j - 1) = 0.0;
            at(gpu_V, i, j) = -at(gpu_V, i + 1, j);
            at(gpu_P,i, j) = (at(gpu_P,i + 1, j) + at(gpu_P,i, j - 1))/2;

            if(*gpu_isHeatTransfer==1) {
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = (at(gpu_T, i + 1, j) + at(gpu_T, i, j - 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T, i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j - 1) + at(gpu_T,i + 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j - 1) + at(gpu_T,i + 1, j) )/2;
            }

        }

        // obstacle B_NW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and West directions 
            
        else if(check_bound_north(j) && check_bound_west(i) && at(gpu_geometry_data,i,j+1)==*gpu_fluid_id && at(gpu_geometry_data,i-1,j)==*gpu_fluid_id) {

            at(gpu_U,i - 1, j) = 0.0;
            at(gpu_U,i, j) = -at(gpu_U,i, j + 1);
            at(gpu_V,i, j) = 0.0;
            at(gpu_V,i, j - 1) = -at(gpu_V,i - 1, j - 1);
            at(gpu_P,i,j) = (at(gpu_P,i - 1, j) + at(gpu_P,i, j + 1))/2;

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = (at(gpu_T,i - 1, j) + at(gpu_T,i, j + 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j + 1) + at(gpu_T,i - 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j + 1) + at(gpu_T,i - 1, j) )/2;
            }

        }

        // obstacle B_SW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and West directions 

        else if(check_bound_south(j) && check_bound_west(i) && at(gpu_geometry_data,i,j-1)==*gpu_fluid_id&& at(gpu_geometry_data,i-1,j)==*gpu_fluid_id){
            at(gpu_U,i - 1, j) = 0.0;
            at(gpu_U,i, j) = at(gpu_U,i, j - 1);
            at(gpu_V,i, j - 1) = 0.0;
            at(gpu_V,i, j) = -at(gpu_V,i - 1, j);
            at(gpu_P,i, j) = (at(gpu_P,i - 1, j) + at(gpu_P,i, j - 1))/2;
           
            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                   at(gpu_T,i, j) = (at(gpu_T,i - 1, j) + at(gpu_T,i, j - 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j - 1) + at(gpu_T,i - 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j - 1) + at(gpu_T,i - 1, j) )/2;
            }
            
        }

        // Bottom Wall B_N (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North direction

        else if(check_bound_north(j) && at(gpu_geometry_data,i,j+1)==*gpu_fluid_id){
            at(gpu_U,i, j) = -at(gpu_U,i, j + 1);
            at(gpu_V,i, j) = 0.0;
            at(gpu_P,i, j) = at(gpu_P,i, j + 1);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i, j + 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - at(gpu_T,i, j + 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - at(gpu_T,i, j + 1);
            }
        }

        // Top Wall B_S (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South direction

        else if(check_bound_south(j) && at(gpu_geometry_data,i,j-1)==*gpu_fluid_id){

            at(gpu_U,i, j) = -at(gpu_U,i, j - 1);
            at(gpu_V,i, j) = 0.0;
            at(gpu_P,i, j) = at(gpu_P,i, j - 1);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i, j - 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2 * (*gpu_wall_temp_h) - at(gpu_T,i, j - 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2 * (*gpu_wall_temp_c) - at(gpu_T,i, j - 1);
            }
        }

        // Left Wall B_E (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the East direction

        else if(check_bound_east(i) && at(gpu_geometry_data,i+1,j)==*gpu_fluid_id){
            at(gpu_U,i, j) = 0.0;
            at(gpu_V,i, j) = -at(gpu_V,i + 1, j);
            at(gpu_P,i, j) = at(gpu_P,i + 1, j);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i + 1, j);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id) 
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - at(gpu_T,i + 1, j);        
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - at(gpu_T,i + 1, j);
            }
        }

        
        /***********************************************************************************************
        * Right Wall B_W (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on the West direction *
        ***********************************************************************************************/

        else if(check_bound_west(i) && at(gpu_geometry_data,i-1,j)==0){
            //Since u grid is staggered, the u velocity of cells to left of ghost layer should be set to 0.
            at(gpu_U,i - 1, j) = 0.0; 
            at(gpu_V,i, j) = -at(gpu_V,i - 1, j);
            at(gpu_P,i, j) = at(gpu_P,i - 1, j);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i - 1, j);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - at(gpu_T,i - 1, j);
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - at(gpu_T,i - 1, j);
            }
        }
    }
}

__global__ void MovingWallBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, int *gpu_moving_wall_id, dtype *gpu_wall_velocity, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_moving_wall_id && check_bound_south(j)) {       
        at(gpu_U,i, j) = 2*(*gpu_wall_velocity)- at(gpu_U,i, j-1);
        //Since v grid is staggered, the v velocity of cells to below of ghost layer should be set to 0.
        at(gpu_V,i,j - 1) = 0.0;
        at(gpu_P,i,j) = at(gpu_P,i, j-1);
    }
}

__global__ void InFlowBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, int *gpu_inflow_id, dtype *gpu_UIN, dtype *gpu_VIN, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (check_bound_east(i) && i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_inflow_id) {
        at(gpu_U,i,j) = (*gpu_UIN);
        at(gpu_V,i,j) = 2*(*gpu_VIN) - at(gpu_V,i + 1, j);
        at(gpu_P,i,j) = at(gpu_P,i + 1, j);
    }
}

__global__ void OutFlowBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, int *gpu_outflow_id, dtype *gpu_POUT, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (check_bound_west(i) && i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_outflow_id) {
            at(gpu_U,i,j) = at(gpu_U,i - 1,j);
            at(gpu_V,i,j) = at(gpu_V,i - 1,j);
            at(gpu_P,i,j) = 2*(*gpu_POUT) - at(gpu_P,i - 1, j);
    }
}

__global__ void calc_T_kernel(dtype *gpu_T, dtype *gpu_T_temp, dtype *gpu_U, dtype *gpu_V, int *gpu_fluid_id, dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_dt,
                              dtype *gpu_alpha, dtype *gpu_gamma, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    //NEED TO DO THIS ONLY FOR FLUID CELLS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_fluid_id)
        at(gpu_T, i, j) = (*gpu_dt) * (*gpu_alpha * diffusion(gpu_T_temp, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - convection_Tu(gpu_T_temp, gpu_U, i, j, *gpu_dx, *gpu_dy, *gpu_gamma, gpu_size_x) - convection_Tv(gpu_T_temp, gpu_V, i, j, *gpu_dx, *gpu_dy, *gpu_gamma, gpu_size_x)) + at(gpu_T_temp, i, j);
}

__global__ void calc_fluxes_kernel(dtype *gpu_F, dtype *gpu_G, dtype *gpu_U, dtype *gpu_V, dtype *gpu_T, int *gpu_fluid_id, int *gpu_geometry_data, dtype *gpu_gx, dtype *gpu_gy, dtype *gpu_dx, dtype *gpu_dy, int *gpu_size_x, int *gpu_size_y, dtype *gpu_gamma, dtype *gpu_beta,
                             dtype *gpu_nu, dtype *gpu_dt, bool *gpu_isHeatTransfer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data,i,j) == *gpu_fluid_id){

        at(gpu_F, i, j) = at(gpu_U, i, j) + (*gpu_dt) * ((*gpu_nu) * diffusion(gpu_U, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - convection_u(gpu_U, gpu_V, *gpu_gamma, i, j, *gpu_dx, *gpu_dy, gpu_size_x) + (*gpu_gx));
        at(gpu_G, i, j) = at(gpu_V, i, j) + (*gpu_dt) * ((*gpu_nu) * diffusion(gpu_V, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - convection_v(gpu_U, gpu_V, *gpu_gamma, i, j, *gpu_dx, *gpu_dy, gpu_size_x) + (*gpu_gy));

        if (*gpu_isHeatTransfer) {

                at(gpu_F,i,j) = at(gpu_F,i,j) - (*gpu_beta) * (*gpu_dt) / 2.0 * (at(gpu_T,i,j) + at(gpu_T,i + 1, j)) * (*gpu_gx) - (*gpu_dt) * (*gpu_gx);
                at(gpu_G,i,j) = at(gpu_G,i,j) - (*gpu_beta) * (*gpu_dt) / 2.0 * (at(gpu_T,i,j) + at(gpu_T,i, j + 1)) * (*gpu_gy) - (*gpu_dt) * (*gpu_gy);
        }
    }
}

__global__ void fluxes_bc_kernel(dtype *gpu_F, dtype *gpu_G, dtype *gpu_U, dtype *gpu_V, int *gpu_fluid_id, int *gpu_fixed_wall_id, int *gpu_adiabatic_id, int *gpu_hot_id, int *gpu_cold_id, int *gpu_moving_wall_id, int *gpu_inflow_id, int *gpu_outflow_id, int *gpu_geometry_data, int *gpu_size_x, int *gpu_size_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < *gpu_size_x && j < *gpu_size_y){

        if (at(gpu_geometry_data, i, j) == *gpu_fixed_wall_id || at(gpu_geometry_data, i, j) == *gpu_adiabatic_id || at(gpu_geometry_data, i, j) == *gpu_hot_id || at(gpu_geometry_data, i, j) == *gpu_cold_id) {
            // B_NE fixed wall corner cell with fluid cells on the North and East directions

            if (check_bound_north(j) && check_bound_east(i) && at(gpu_geometry_data, i, j + 1) == *gpu_fluid_id && at(gpu_geometry_data, i + 1, j) == *gpu_fluid_id) {
                at(gpu_F, i, j) = at(gpu_U, i, j);
                at(gpu_G, i, j) = at(gpu_V, i, j);
            }

            // B_SE fixed wall corner cell with fluid cells on the South and East directions

            else if (check_bound_south(j) && check_bound_east(i) && at(gpu_geometry_data, i, j - 1) == *gpu_fluid_id && at(gpu_geometry_data, i + 1, j) == *gpu_fluid_id) {
                at(gpu_F, i, j) = at(gpu_U, i, j);
                at(gpu_G, i, j - 1) = at(gpu_V, i, j - 1);
            }

            // B_NW fixed wall corner cell with fluid cells on the North and West directions

            else if (check_bound_north(j) && check_bound_west(i) && at(gpu_geometry_data, i, j + 1) == *gpu_fluid_id && at(gpu_geometry_data, i - 1, j) == *gpu_fluid_id) {
                at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);
                at(gpu_G, i, j) = at(gpu_V, i, j);
            }

            // B_SW fixed wall corner cell with fluid cells on the South and West directions

            else if (check_bound_south(j) && check_bound_west(i) && at(gpu_geometry_data, i, j - 1) == *gpu_fluid_id && at(gpu_geometry_data, i - 1, j) == *gpu_fluid_id) {
                at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);
                at(gpu_G, i, j - 1) = at(gpu_V, i, j - 1);
            } 
            else if (check_bound_north(j) && at(gpu_geometry_data, i, j + 1) == *gpu_fluid_id)
                at(gpu_G, i, j) = at(gpu_V, i, j);

            else if (check_bound_south(j) && at(gpu_geometry_data, i, j - 1) == *gpu_fluid_id)
                at(gpu_G,i,j - 1) = at(gpu_V,i,j - 1);

            else if (check_bound_west(i) && at(gpu_geometry_data, i - 1, j) == *gpu_fluid_id)
                at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);

            else if (check_bound_east(i) && at(gpu_geometry_data, i + 1, j) == *gpu_fluid_id)
                at(gpu_F, i, j) = at(gpu_U, i, j);

        }

        else if (at(gpu_geometry_data, i, j) == *gpu_moving_wall_id) {
            at(gpu_G, i, j - 1) = at(gpu_V, i, j - 1);
        } 
        
        else if (at(gpu_geometry_data, i, j) == *gpu_inflow_id) {
            at(gpu_F, i, j) = at(gpu_U, i, j);
        }

        else if (at(gpu_geometry_data, i, j) == *gpu_outflow_id) {

            at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);
        }
    }
}

__global__ void solve_pressure_red_kernel(dtype *gpu_RS, dtype *gpu_P, int *gpu_fluid_id, dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_omega, dtype
*gpu_coeff, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    *gpu_coeff = *gpu_omega / (2.0 * (1.0 / (*gpu_dx * *gpu_dx) + 1.0 / (*gpu_dy * *gpu_dy)));

    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_fluid_id && (i + j)%2 != 0){
        at(gpu_P,i, j) = (1.0 - *gpu_omega) * at(gpu_P,i, j) +
                        (*gpu_coeff) * (sor_helper(gpu_P, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - at(gpu_RS,i, j));
    }

}

__global__ void solve_pressure_black_kernel(dtype *gpu_RS, dtype *gpu_P, int *gpu_fluid_id, dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_omega, dtype
*gpu_coeff, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    *gpu_coeff = *gpu_omega / (2.0 * (1.0 / (*gpu_dx * *gpu_dx) + 1.0 / (*gpu_dy * *gpu_dy)));

    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_fluid_id && (i + j)%2 == 0){
        at(gpu_P,i, j) = (1.0 - *gpu_omega) * at(gpu_P,i, j) +
                        (*gpu_coeff) * (sor_helper(gpu_P, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - at(gpu_RS,i, j));
    }

}

__global__ void res_kernel(dtype *gpu_RS, dtype *gpu_P, int *gpu_fluid_id, dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_res, int *gpu_size_x, int *gpu_size_y, int *gpu_fluid_cells_size, int *gpu_geometry_data){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    dtype val;

    if (i < *gpu_size_x && j < *gpu_size_y){
        at(gpu_res, i, j) = 0.0;
        if (at(gpu_geometry_data, i, j) == *gpu_fluid_id){
            val = laplacian(gpu_P, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - at(gpu_RS,i, j);
            at(gpu_res, i, j) = (val * val);
        }
    }
}

__global__ void calc_rs_kernel(dtype *gpu_RS, dtype *gpu_F, dtype *gpu_G, int *gpu_fluid_id, dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_dt, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_fluid_id)
        at(gpu_RS,i,j) = 1 / (*gpu_dt) * ((at(gpu_F,i, j) - at(gpu_F,i - 1, j)) / (*gpu_dx) + (at(gpu_G,i, j) - at(gpu_G,i, j - 1)) / (*gpu_dy));
}

__global__ void calc_velocities_kernel(dtype *gpu_F, dtype *gpu_G, dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, int *gpu_fluid_id, int *gpu_outflow_id, dtype *gpu_dx, dtype *gpu_dy, int *gpu_size_x, int *gpu_size_y, dtype *gpu_dt, int *gpu_geometry_data)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == *gpu_fluid_id)
    {   
        if (( at(gpu_geometry_data, i+1, j)== *gpu_fluid_id) || (at(gpu_geometry_data, i+1, j) == *gpu_outflow_id)) {
            at(gpu_U,i, j) = at(gpu_F,i, j) - ((*gpu_dt)/(*gpu_dx)) * (at(gpu_P,i + 1, j) - at(gpu_P,i, j));           
        }
        if ((at(gpu_geometry_data, i, j+1)==*gpu_fluid_id) || (at(gpu_geometry_data, i, j+1)==*gpu_outflow_id)) {
            at(gpu_V,i, j) = at(gpu_G,i, j) - ((*gpu_dt)/(*gpu_dy)) * (at(gpu_P,i, j + 1) - at(gpu_P,i, j));
        }
    }
}

void CUDA_solver::initialize(Fields &field, Grid &grid, dtype cpu_UIN, dtype cpu_VIN, dtype cpu_wall_temp_a, dtype cpu_wall_temp_h, dtype cpu_wall_temp_c, dtype cpu_omg) {

    UIN = cpu_UIN;
    VIN = cpu_VIN;
    wall_temp_a = cpu_wall_temp_a;
    wall_temp_h = cpu_wall_temp_h;
    wall_temp_c = cpu_wall_temp_c;
    omg = cpu_omg;

    block_size = dim3(BLOCK_SIZE);
    block_size_2d = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    domain_size = (grid.domain().domain_size_x + 2) * (grid.domain().domain_size_y + 2);
    cudaMalloc((void **)&gpu_geometry_data, domain_size * sizeof(int));

    grid_size = grid.imaxb() * grid.jmaxb();
    grid_size_x = grid.imaxb();
    grid_size_y = grid.jmaxb();
    grid_fluid_cells_size = grid.fluid_cells().size();

    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);

    cudaMalloc((void **)&gpu_T, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_T_temp, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_U, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_V, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_P, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_F, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_G, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_RS, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_res, grid_size * sizeof(dtype));

    cudaMalloc((void **)&gpu_dx, sizeof(dtype));
    cudaMalloc((void **)&gpu_dy, sizeof(dtype));
    cudaMalloc((void **)&gpu_dt, sizeof(dtype));
    cudaMalloc((void **)&gpu_alpha, sizeof(dtype));
    cudaMalloc((void **)&gpu_gamma, sizeof(dtype));
    cudaMalloc((void **)&gpu_beta, sizeof(dtype));
    cudaMalloc((void **)&gpu_nu, sizeof(dtype));
    cudaMalloc((void **)&gpu_tau, sizeof(dtype));
    cudaMalloc((void **)&gpu_gx, sizeof(dtype));
    cudaMalloc((void **)&gpu_gy, sizeof(dtype));
    cudaMalloc((void **)&gpu_omega, sizeof(dtype));

    cudaMalloc((void **)&gpu_coeff, sizeof(dtype));
    cudaMalloc((void **)&gpu_rloc, sizeof(dtype));
    cudaMalloc((void **)&gpu_val, sizeof(dtype));


    cudaMalloc((void **)&gpu_size_x, sizeof(dtype));
    cudaMalloc((void **)&gpu_size_y, sizeof(dtype));

    cudaMalloc((void **)&gpu_fluid_id, sizeof(int));
    cudaMalloc((void **)&gpu_moving_wall_id, sizeof(int));
    cudaMalloc((void **)&gpu_fixed_wall_id, sizeof(int));
    cudaMalloc((void **)&gpu_inflow_id, sizeof(int));
    cudaMalloc((void **)&gpu_outflow_id, sizeof(int));
    cudaMalloc((void **)&gpu_adiabatic_id, sizeof(int));
    cudaMalloc((void **)&gpu_hot_id, sizeof(int));
    cudaMalloc((void **)&gpu_cold_id, sizeof(int));
    cudaMalloc((void **)&gpu_fluid_cells_size, sizeof(int));

    cudaMalloc((void **)&gpu_wall_temp_a, sizeof(dtype));
    cudaMalloc((void **)&gpu_wall_temp_h, sizeof(dtype));
    cudaMalloc((void **)&gpu_wall_temp_c, sizeof(dtype));
    cudaMalloc((void **)&gpu_isHeatTransfer, sizeof(bool));

    cudaMalloc((void **)&gpu_wall_velocity, sizeof(dtype));
    cudaMalloc((void **)&gpu_UIN, sizeof(dtype));
    cudaMalloc((void **)&gpu_VIN, sizeof(dtype));
    cudaMalloc((void **)&gpu_POUT, sizeof(dtype));

    cudaMalloc((void **)&gpu_umax, sizeof(dtype));
    cudaMalloc((void **)&gpu_vmax, sizeof(dtype));


    // Typecasting raw pointer to thrust device pointer
    thrust_U = thrust::device_pointer_cast(gpu_U);
    thrust_V = thrust::device_pointer_cast(gpu_V);
    thrust_res = thrust::device_pointer_cast(gpu_res);
}

void CUDA_solver::pre_process(Fields &field, Grid &grid, Discretization &discretization, dtype cpu_dt) {

    cudaMemcpy(gpu_geometry_data, grid.get_geometry_data().data(), domain_size * sizeof(int), cudaMemcpyHostToDevice);

    bool energycheck = field.isHeatTransfer();
    cudaMemcpy(gpu_isHeatTransfer, &energycheck, sizeof(bool), cudaMemcpyHostToDevice);

    if (energycheck)
        cudaMemcpy(gpu_T, field.t_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_U, field.u_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_V, field.v_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_P, field.p_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_F, field.f_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_G, field.g_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_RS, field.rs_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);

    dtype var = grid.dx();
    cpu_dx = var;
    cudaMemcpy(gpu_dx, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = grid.dy();
    cpu_dy = var;
    cudaMemcpy(gpu_dy, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = cpu_dt;
    cudaMemcpy(gpu_dt, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_alpha();
    cpu_alpha = var;
    cudaMemcpy(gpu_alpha, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_beta();
    cudaMemcpy(gpu_beta, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = discretization.get_gamma();
    cudaMemcpy(gpu_gamma, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_nu();
    cpu_nu = var;
    cudaMemcpy(gpu_nu, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_tau();
    cpu_tau = var;
    cudaMemcpy(gpu_tau, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_gx();
    cudaMemcpy(gpu_gx, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_gy();
    cudaMemcpy(gpu_gy, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = 0;
    cudaMemcpy(gpu_umax, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = 0;
    cudaMemcpy(gpu_vmax, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_size_x, &grid_size_x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size_y, &grid_size_y, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_fluid_cells_size, &grid_fluid_cells_size, sizeof(int), cudaMemcpyHostToDevice);

    int var1 = GEOMETRY_PGM::moving_wall_id;
    cudaMemcpy(gpu_moving_wall_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::fixed_wall_id;
    cudaMemcpy(gpu_fixed_wall_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::inflow_id;
    cudaMemcpy(gpu_inflow_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::outflow_id;
    cudaMemcpy(gpu_outflow_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::hot_id;
    cudaMemcpy(gpu_hot_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::cold_id;
    cudaMemcpy(gpu_cold_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::adiabatic_id;
    cudaMemcpy(gpu_adiabatic_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_wall_temp_a, &wall_temp_a, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wall_temp_h, &wall_temp_h, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wall_temp_c, &wall_temp_c, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_omega, &omg, sizeof(dtype), cudaMemcpyHostToDevice);

    var = LidDrivenCavity::wall_velocity;
    cudaMemcpy(gpu_wall_velocity, &var, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_UIN, &UIN, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_VIN, &VIN, sizeof(dtype), cudaMemcpyHostToDevice);
    var = GEOMETRY_PGM::POUT;
    cudaMemcpy(gpu_POUT, &var, sizeof(dtype), cudaMemcpyHostToDevice);
}

void CUDA_solver::apply_boundary() {


    FixedWallBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_T, gpu_geometry_data, gpu_fluid_id, gpu_moving_wall_id, gpu_fixed_wall_id, gpu_inflow_id, gpu_outflow_id, gpu_adiabatic_id, gpu_hot_id, gpu_cold_id, gpu_wall_temp_a, gpu_wall_temp_h, gpu_wall_temp_c, gpu_isHeatTransfer, gpu_size_x, gpu_size_y);

    MovingWallBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_moving_wall_id, gpu_wall_velocity, gpu_size_x, gpu_size_y, gpu_geometry_data);

    InFlowBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_inflow_id, gpu_UIN, gpu_VIN, gpu_size_x, gpu_size_y, gpu_geometry_data);

    OutFlowBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_outflow_id, gpu_POUT, gpu_size_x, gpu_size_y, gpu_geometry_data);

}

void CUDA_solver::calc_pressure(int max_iter, dtype tolerance, dtype t, dtype dt) {

    dtype res = 0.;
    int iter = 0;

    do{
        apply_boundary();
        solve_pressure_red_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_RS, gpu_P, gpu_fluid_id,gpu_dx, gpu_dy, gpu_omega, gpu_coeff, gpu_size_x, gpu_size_y, gpu_geometry_data);
        solve_pressure_black_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_RS, gpu_P,gpu_fluid_id, gpu_dx, gpu_dy, gpu_omega, gpu_coeff, gpu_size_x, gpu_size_y, gpu_geometry_data);
        res_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_RS, gpu_P, gpu_fluid_id, gpu_dx, gpu_dy, gpu_res, gpu_size_x, gpu_size_y, gpu_fluid_cells_size, gpu_geometry_data);
        res = thrust::reduce(thrust::device, thrust_res, thrust_res + grid_size, (dtype) 0, thrust::plus<dtype>());
        iter++;
    }while(res > tolerance && iter < max_iter);
    
    

    if (iter == max_iter) {
        std::cout << "Max iteration reached at " << t<<" s \n";
    }

    std::cout << "Time = " << std::setw(12) << t << " Residual = "<< std::setw(12) << res <<
    
    " Iter = " << std::setw(8) << iter << " dt = " << std::setw(12) << dt << '\n';

}

void CUDA_solver::calc_T() {
    cudaMemcpy(gpu_T_temp, gpu_T, grid_size * sizeof(dtype), cudaMemcpyDeviceToDevice);
    calc_T_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_T, gpu_T_temp, gpu_U, gpu_V, gpu_fluid_id, gpu_dx, gpu_dy, gpu_dt, gpu_alpha, gpu_gamma, gpu_size_x, gpu_size_y, gpu_geometry_data);
}

void CUDA_solver::calc_fluxes() {
    calc_fluxes_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_F,gpu_G,gpu_U,gpu_V,gpu_T, gpu_fluid_id, gpu_geometry_data,gpu_gx,gpu_gy,gpu_dx,gpu_dy,gpu_size_x, gpu_size_y, gpu_gamma, gpu_beta, gpu_nu, gpu_dt, gpu_isHeatTransfer);
    fluxes_bc_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_F,gpu_G,gpu_U,gpu_V, gpu_fluid_id, gpu_fixed_wall_id, gpu_adiabatic_id, gpu_hot_id, gpu_cold_id, gpu_moving_wall_id, gpu_inflow_id, gpu_outflow_id, gpu_geometry_data,gpu_size_x, gpu_size_y);
}

void CUDA_solver::calc_rs() {
    calc_rs_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_RS, gpu_F,gpu_G, gpu_fluid_id, gpu_dx,gpu_dy, gpu_dt, gpu_size_x, gpu_size_y, gpu_geometry_data);
}

void CUDA_solver::calc_velocities() {
    calc_velocities_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_F, gpu_G, gpu_U, gpu_V, gpu_P, gpu_fluid_id, gpu_outflow_id, gpu_dx, gpu_dy, gpu_size_x, gpu_size_y, gpu_dt, gpu_geometry_data);
}

dtype CUDA_solver::calc_dt() {

    num_blocks = get_num_blocks(grid_size);
    dtype t[4];
    dtype result;

    // Finding max and min element in U and V
    thrust_U_max = thrust::max_element(thrust_U, thrust_U + grid_size);
    thrust_U_min = thrust::min_element(thrust_U, thrust_U + grid_size);
    thrust_V_max = thrust::max_element(thrust_V, thrust_V + grid_size);
    thrust_V_min = thrust::min_element(thrust_V, thrust_V + grid_size);
    
    // Finding Maximum between max element and -(min element) as Abs max needed.
    dtype umax = get_max(*thrust_U_max, -(*thrust_U_min));
    dtype vmax = get_max(*thrust_V_max, -(*thrust_V_min));

    t[0] = 1 / (2 * (cpu_nu) * (1/((cpu_dx)*(cpu_dx)) + 1/((cpu_dy)*(cpu_dy))));
    t[1] = (cpu_dx) / (umax);
    t[2] = (cpu_dy) / (vmax);   
    t[3] = 1 / (2 * (cpu_alpha) * (1/((cpu_dx)*(cpu_dx)) + 1/((cpu_dy)*(cpu_dy))));
    dtype temp_dt =t[0];

    // //Previous implementation
    // max_abs_element_kernel<<<num_blocks, block_size>>>(gpu_U, gpu_size_x, gpu_size_y, d_mutex, gpu_umax);
    // max_abs_element_kernel<<<num_blocks, block_size>>>(gpu_V, gpu_size_x, gpu_size_y, d_mutex, gpu_vmax);
    // cudaMemcpy((void *)&cpu_umax, gpu_umax, sizeof(dtype), cudaMemcpyDeviceToHost);
    // cudaMemcpy((void *)&cpu_vmax, gpu_vmax, sizeof(dtype), cudaMemcpyDeviceToHost);
    // t[0] = 1 / (2 * (cpu_nu) * (1/((cpu_dx)*(cpu_dx)) + 1/((cpu_dy)*(cpu_dy))));
    // t[1] = (cpu_dx) / (cpu_umax);
    // t[2] = (cpu_dy) / (cpu_vmax);   
    // t[3] = 1 / (2 * (cpu_alpha) * (1/((cpu_dx)*(cpu_dx)) + 1/((cpu_dy)*(cpu_dy))));
    // dtype temp_dt =t[0];

    for(int i=1; i<4; i++)
    {
        if(t[i]<temp_dt)
            temp_dt = t[i];
    }

    result = (cpu_tau) * temp_dt;
    cudaMemcpy(gpu_dt, &result, sizeof(dtype), cudaMemcpyHostToDevice);
    return result;

}

void CUDA_solver::post_process(Fields &field) {
    
    if (field.isHeatTransfer())
        cudaMemcpy((void *)field.t_matrix().data(), gpu_T, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    
    cudaMemcpy((void *)field.f_matrix().data(), gpu_F, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.g_matrix().data(), gpu_G, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.rs_matrix().data(), gpu_RS, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.u_matrix().data(), gpu_U, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.v_matrix().data(), gpu_V, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.p_matrix().data(), gpu_P, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
}

CUDA_solver::~CUDA_solver() {
    cudaFree(gpu_geometry_data);
    cudaFree(gpu_T);
    cudaFree(gpu_U);
    cudaFree(gpu_V);
    cudaFree(gpu_P);
    cudaFree(gpu_F);
    cudaFree(gpu_G);
    cudaFree(gpu_RS);
    cudaFree(gpu_T_temp);
    cudaFree(gpu_dx);
    cudaFree(gpu_dy);
    cudaFree(gpu_dt);
    cudaFree(gpu_gx);
    cudaFree(gpu_gy);
    cudaFree(gpu_nu);
    cudaFree(gpu_beta);
    cudaFree(gpu_gamma);
    cudaFree(gpu_alpha);
    cudaFree(gpu_tau);
    cudaFree(gpu_size_x);
    cudaFree(gpu_size_y);
    cudaFree(gpu_fluid_id);
    cudaFree(gpu_fixed_wall_id);
    cudaFree(gpu_moving_wall_id);
    cudaFree(gpu_inflow_id);
    cudaFree(gpu_outflow_id);
    cudaFree(gpu_adiabatic_id);
    cudaFree(gpu_hot_id);
    cudaFree(gpu_cold_id);
    cudaFree(gpu_wall_temp_a);
    cudaFree(gpu_wall_temp_c);
    cudaFree(gpu_wall_temp_h);
    cudaFree(gpu_isHeatTransfer);
    cudaFree(gpu_UIN);
    cudaFree(gpu_VIN);
    cudaFree(gpu_POUT);
    cudaFree(gpu_wall_velocity);
    cudaFree(gpu_umax);
    cudaFree(gpu_vmax);
    cudaFree(gpu_omega);
    cudaFree(gpu_coeff);
    cudaFree(gpu_rloc);
    cudaFree(gpu_val);
    cudaFree(gpu_res);
    cudaFree(gpu_fluid_cells_size);
}
