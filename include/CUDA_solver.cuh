#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Fields.hpp"
#include "Grid.hpp"
#include "Enums.hpp"
#include "Discretization.hpp"
#include "Boundary.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#define BLOCK_SIZE 128
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

class CUDA_solver{

    private:
    
    dtype *gpu_T, *gpu_T_temp;
    dtype *gpu_U;
    dtype *gpu_V;
    dtype *gpu_P;
    dtype *gpu_F;
    dtype *gpu_G;
    dtype *gpu_RS;

    int *gpu_geometry_data;

    int *gpu_fluid_id;
    int *gpu_moving_wall_id; 
    int *gpu_fixed_wall_id;
    int *gpu_inflow_id;
    int *gpu_outflow_id;
    int *gpu_adiabatic_id;
    int *gpu_hot_id;
    int *gpu_cold_id; 
    int *gpu_fluid_cells_size;

    dtype *gpu_POUT;
    dtype *gpu_UIN;
    dtype *gpu_VIN;

    dtype *gpu_omega, *gpu_coeff, *gpu_rloc, *gpu_val, *gpu_res;

    dtype *gpu_umax, *gpu_vmax;
    dtype *gpu_wall_temp_a, *gpu_wall_temp_h, *gpu_wall_temp_c;

    int domain_size, grid_size, grid_size_x, grid_size_y;
    dtype *gpu_dx, *gpu_dy, *gpu_dt, *gpu_gamma, *gpu_alpha, *gpu_beta, *gpu_nu, *gpu_tau;
    dtype *gpu_gx, *gpu_gy;

    int *gpu_size_x, *gpu_size_y;
    dtype *gpu_wall_velocity;

    bool *gpu_isHeatTransfer;

    dim3 block_size, num_blocks, block_size_2d, num_blocks_2d;

    dtype UIN, VIN, wall_temp_a, wall_temp_h, wall_temp_c, omg;
    dtype cpu_umax, cpu_vmax, cpu_dx, cpu_dy, cpu_nu, cpu_alpha, cpu_tau;

    int grid_fluid_cells_size;

    thrust::device_ptr<dtype> thrust_U, thrust_V;
    thrust::device_ptr<dtype> thrust_U_max, thrust_U_min, thrust_V_max, thrust_V_min;
    thrust::device_ptr<dtype> thrust_res;

    thrust::maximum<dtype> get_max;
  
    public:

    void initialize(Fields &, Grid &, dtype, dtype, dtype, dtype, dtype, dtype);
    void pre_process(Fields &, Grid &, Discretization &, dtype);
    void post_process(Fields &);
    void calc_T();
    void apply_boundary();
    void calc_fluxes();
    void calc_rs();
    void calc_pressure(int, dtype, dtype, dtype);
    void calc_velocities();
    dtype calc_dt();
    dim3 get_num_blocks(int);
    dim3 get_num_blocks_2d(int, int);
    
    ~CUDA_solver();

};
