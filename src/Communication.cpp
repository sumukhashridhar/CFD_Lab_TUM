#include <mpi.h>
#include "Communication.hpp"

void Communication::init_parallel(int argc, char **argv){
    MPI_Init(&argc, &argv);
}

double Communication::reduce_min(double rank) {
    double min_value{0};
    MPI_Reduce(&rank, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    return min_value;
}

double Communication::reduce_max(double rank) {
    double max_value{0};
    MPI_Reduce(&rank, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_value;
}

double Communication::reduce_sum(double rank) {
    double sum{0};
    MPI_Reduce(&rank, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return sum;
}

void Communication::broadcast(double umax) {
    MPI_Bcast(&umax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Communication::communicate(Matrix<double> &field){

    auto neighbour = _assign_neighbours(get_rank());

    int data_imax = field.imax();
    int data_jmax = field.jmax();

    MPI_Status status;
    
    /* 
    data_lr_out - array to be sent to neighbouring processes in left / right direction
    data_lr_in - array to be received from neighbouring processes from left / right direction
    data_tb_out - array to be sent to neighbouring processes in top / bottom direction
    data_tb_in - array to be received from neighbouring processes from top / bottom direction
    */

    double data_lr_out[data_jmax], data_lr_in[data_jmax], data_tb_out[data_imax], data_tb_in[data_imax];
    
    // Communication in Left direction
    if (neighbour['L'] != MPI_PROC_NULL){
        
        for (auto k = 0; k < data_jmax; ++k){
            data_lr_out[k] = field(1, k);
        }

        MPI_Sendrecv(&data_lr_out, data_jmax, MPI_DOUBLE, neighbour['L'], 0,
                     &data_lr_in, data_jmax, MPI_DOUBLE, neighbour['L'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_jmax; ++k){
            field(0, k) = data_lr_in[k];
        }
        
    }

    // Communication in Right direction
    if (neighbour['R'] != MPI_PROC_NULL){
        
        for (auto k = 0; k < data_jmax; ++k){
            data_lr_out[k] = field(data_imax - 2, k);
        }

        MPI_Sendrecv(&data_lr_out, data_jmax, MPI_DOUBLE, neighbour['R'], 0,
                     &data_lr_in, data_jmax, MPI_DOUBLE, neighbour['R'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_jmax; ++k){
            field(data_imax - 1, k) = data_lr_in[k];
        }
 
    }


    // Communication in Top direction
    if (neighbour['T'] != MPI_PROC_NULL){
        
        for (auto k = 0; k < data_imax; ++k){
            data_tb_out[k] = field(k, data_jmax - 2);
        }

        MPI_Sendrecv(&data_tb_out, data_imax, MPI_DOUBLE, neighbour['T'], 0,
                     &data_tb_in, data_imax, MPI_DOUBLE, neighbour['T'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_imax; ++k){
            field(k, data_jmax - 1) = data_tb_in[k];
        }
        
    }

    // Communication in Bottom direction
    if (neighbour['B'] != MPI_PROC_NULL){
        
        for (auto k = 0; k < data_imax; ++k){
            data_tb_out[k] = field(k, 1);
        }

        MPI_Sendrecv(&data_tb_out, data_imax, MPI_DOUBLE, neighbour['B'], 0,
                     &data_tb_in, data_imax, MPI_DOUBLE, neighbour['B'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_imax; ++k){
            field(k, 0) = data_tb_in[k];
        }
        
    }    
}

int Communication::get_rank(){
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    return my_rank;
}

int Communication::get_size(){
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void Communication::finalize() {

    MPI_Finalize();

}

std::map<char, int> Communication::_assign_neighbours(int rank){
    std::map<char, int> neighbour;

    int i = rank % iproc;
    int j = (rank - i) / iproc;

    // L - Left, R - Right, T - Top, B - Bottom
    if (i - 1 >= 0)
        neighbour['L'] =  (i - 1) + j * iproc;
    else
        neighbour['L'] = MPI_PROC_NULL;

    if (i + 1 < iproc)
        neighbour['R'] =  (i + 1) + j * iproc;
    else
        neighbour['R'] = MPI_PROC_NULL;

    if (j + 1 < jproc)
        neighbour['T'] =  i + (j + 1) * iproc;
    else
        neighbour['T'] = MPI_PROC_NULL;

    if (j - 1 >= 0)
        neighbour['B'] =  i + (j - 1) * iproc;
    else
        neighbour['B'] = MPI_PROC_NULL;

    return neighbour;
}
