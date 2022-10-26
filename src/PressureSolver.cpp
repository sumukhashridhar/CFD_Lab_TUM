#include "PressureSolver.hpp"

#include <cmath>
#include <iostream>
#include "Communication.hpp"

SOR::SOR(double omega) : _omega(omega) {}

double SOR::solve(Fields &field, Grid &grid, const std::vector<std::unique_ptr<Boundary>> &boundaries) {

    double dx = grid.dx();
    double dy = grid.dy();

    double rloc_sum;
    int size;

    double coeff = _omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy))); // = _omega * h^2 / 4.0, if dx == dy == h

    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();

        field.p(i, j) = (1.0 - _omega) * field.p(i, j) +
                        coeff * (Discretization::sor_helper(field.p_matrix(), i, j) - field.rs(i, j));
    }

    Communication::communicate(field.p_matrix());

    double res = 0.0;
    double rloc = 0.0;

    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();

        double val = Discretization::laplacian(field.p_matrix(), i, j) - field.rs(i, j);
        rloc += (val * val);
    }

    // Reduction of sum from all processes
    rloc_sum = Communication::reduce_sum(rloc);

    // Reduction of number of fluid cells from all processes
    size = Communication::reduce_sum(grid.fluid_cells().size());

    // Residual on whole domain computed by rank 0 and broadcasted to other processes
    if(Communication::get_rank() == 0)
    {
        res = rloc_sum / (size);
        res = std::sqrt(res);
    }
    MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return res;
}
