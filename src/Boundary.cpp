#include "Boundary.hpp"
#include <cmath>
#include <iostream>

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : _cells(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : _cells(cells), _wall_temperature(wall_temperature) {}

/*****************************************************************************************
 * This function applies boundary conditions to fixed wall as given in equation (15)-(17)
****************************************************************************************/
void FixedWallBoundary::apply(Fields &field) {

    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();
        
        // obstacles B_NE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and East directions 

        if(currentCell->is_border(border_position::TOP) && currentCell->is_border(border_position::RIGHT)){
            field.u(i, j) = 0.0;
            field.u(i - 1, j) = -field.u(i - 1, j + 1);
            field.v(i, j) = 0.0;
            field.v(i, j - 1) = -field.v(i + 1, j - 1);
            field.p(i, j) = (field.p(i, j + 1) + field.p(i + 1, j))/2;

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                   field.t(i, j) = (field.t(i + 1, j) + field.t(i, j + 1))/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j + 1) + field.t(i + 1, j) )/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j + 1) + field.t(i + 1, j) )/2;
            }

        }

        // obstacles B_SE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and East directions 

        else if(currentCell->is_border(border_position::BOTTOM) && currentCell->is_border(border_position::RIGHT)){
            field.u(i, j) = 0.0;
            field.u(i - 1, j) = -field.u(i - 1, j - 1);
            field.v(i, j - 1) = 0.0;
            field.v(i, j) = -field.v(i + 1, j);
            field.p(i, j) = (field.p(i + 1, j) + field.p(i, j - 1))/2;

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                   field.t(i, j) = (field.t(i + 1, j) + field.t(i, j - 1))/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j - 1) + field.t(i + 1, j) )/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j - 1) + field.t(i + 1, j) )/2;
            }

        }

        // obstacle B_NW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and West directions 

        else if(currentCell->is_border(border_position::TOP) && currentCell->is_border(border_position::LEFT)){
            field.u(i - 1, j) = 0.0;
            field.u(i, j) = -field.u(i, j + 1);
            field.v(i, j) = 0.0;
            field.v(i, j - 1) = -field.v(i - 1, j - 1);
            field.p(i,j) = (field.p(i - 1, j) + field.p(i, j + 1))/2;

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                   field.t(i, j) = (field.t(i - 1, j) + field.t(i, j + 1))/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j + 1) + field.t(i - 1, j) )/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j + 1) + field.t(i - 1, j) )/2;
            }

        }

        // obstacle B_SW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and West directions 

        else if(currentCell->is_border(border_position::BOTTOM) && currentCell->is_border(border_position::LEFT)){
            field.u(i - 1, j) = 0.0;
            field.u(i, j) = field.u(i, j - 1);
            field.v(i, j - 1) = 0.0;
            field.v(i, j) = -field.v(i - 1, j);
            field.p(i, j) = (field.p(i - 1, j) + field.p(i, j - 1))/2;
           
            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                   field.t(i, j) = (field.t(i - 1, j) + field.t(i, j - 1))/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j - 1) + field.t(i - 1, j) )/2;
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - (field.t(i, j - 1) + field.t(i - 1, j) )/2;
            }
            
        }

        // Bottom Wall B_N (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North direction

        else if(currentCell->is_border(border_position::TOP)){
            field.u(i, j) = -field.u(i, j + 1);
            field.v(i, j) = 0.0;
            field.p(i, j) = field.p(i, j + 1);

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                    field.t(i, j) = field.t(i, j + 1);
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2*wall_temp - field.t(i, j + 1);
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - field.t(i, j + 1);
            }
        }

        

        // Top Wall B_S (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South direction

        else if(currentCell->is_border(border_position::BOTTOM)){
            field.u(i, j) = -field.u(i, j - 1);
            field.v(i, j) = 0.0;
            field.p(i, j) = field.p(i, j - 1);

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                    field.t(i, j) = field.t(i, j - 1);
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2 * wall_temp - field.t(i, j - 1);
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2 * wall_temp - field.t(i, j - 1);
            }
        }

        

        // Left Wall B_E (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the East direction

        else if(currentCell->is_border(border_position::RIGHT)){
            field.u(i, j) = 0.0;
            field.v(i, j) = -field.v(i + 1, j);
            field.p(i, j) = field.p(i + 1, j);

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                    field.t(i, j) = field.t(i + 1, j);
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id) {
                    field.t(i, j) = 2*wall_temp - field.t(i + 1, j);
            }        
        
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - field.t(i + 1, j);
            }
        }

        
        /***********************************************************************************************
        * Right Wall B_W (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on the West direction *
        ***********************************************************************************************/

        else if(currentCell->is_border(border_position::LEFT)){
            //Since u grid is staggered, the u velocity of cells to left of ghost layer should be set to 0.
            field.u(i - 1, j) = 0.0; 
            field.v(i, j) = -field.v(i - 1, j);
            field.p(i, j) = field.p(i - 1, j);

            if(field.isHeatTransfer()){
                int id = _wall_temperature.begin()->first;
                double wall_temp = _wall_temperature.begin()->second;
                if(currentCell->wall_id() == GEOMETRY_PGM::adiabatic_id)
                    field.t(i, j) = field.t(i - 1, j);
                else if (currentCell->wall_id() == GEOMETRY_PGM::hot_id && id == GEOMETRY_PGM::hot_id)
                    field.t(i, j) = 2*wall_temp - field.t(i - 1, j);
                else if (currentCell->wall_id() == GEOMETRY_PGM::cold_id && id == GEOMETRY_PGM::cold_id)
                    field.t(i, j) = 2*wall_temp - field.t(i - 1, j);
            }
        }
        
        
    }

}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, double wall_velocity) : _cells(cells) {
    _wall_velocity.insert(std::pair(LidDrivenCavity::moving_wall_id, wall_velocity));
}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_velocity,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _wall_velocity(wall_velocity), _wall_temperature(wall_temperature) {}

/***********************************************************************************************
 * This function applies boundary conditions to moving wall as given in equation (15)-(17)
 * The u velocity of moving wall is set such that average at the top boundary is wall velocity.
 **********************************************************************************************/

void MovingWallBoundary::apply(Fields &field) {

    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();
        field.u(i, j) = 2 * (_wall_velocity.begin()->second) - field.u(i, j-1);
        //Since v grid is staggered, the v velocity of cells to below of ghost layer should be set to 0.
        field.v(i,j - 1) = 0.0;
        field.p(i,j) = field.p(i, j-1);
    }
}

// Inflow Boundary

InFlowBoundary::InFlowBoundary(std::vector<Cell *> cells, double UIN, double VIN) : _cells(cells) {
    _UIN.insert(std::pair(GEOMETRY_PGM::inflow_id, UIN));
    _VIN.insert(std::pair(GEOMETRY_PGM::inflow_id, VIN));
}

InFlowBoundary::InFlowBoundary(std::vector<Cell *> cells, std::map<int, double> UIN, std::map<int, double> VIN,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _UIN(UIN), _VIN(VIN), _wall_temperature(wall_temperature) {}


/*****************************************************************************************
 * This function applies boundary conditions to Inflow Boundary *
****************************************************************************************/


void InFlowBoundary::apply(Fields &field) {
    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();
            field.u(i,j) = _UIN.begin()->second;
            field.v(i,j) = 2 * _VIN.begin()->second - field.v(i + 1, j);
            field.p(i,j) = field.p(i + 1, j);
    }
}

OutFlowBoundary::OutFlowBoundary(std::vector<Cell *> cells, double POUT) : _cells(cells) {
    _POUT.insert(std::pair(GEOMETRY_PGM::outflow_id, POUT));
}

OutFlowBoundary::OutFlowBoundary(std::vector<Cell *> cells, std::map<int, double> POUT,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _POUT(POUT), _wall_temperature(wall_temperature) {}


/*****************************************************************************************
 * This function applies boundary conditions to Outflow Boundary *
****************************************************************************************/

void OutFlowBoundary::apply(Fields &field) {
    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();

            field.u(i,j) = field.u(i - 1,j);
            field.v(i,j) = field.v(i - 1,j);
            field.p(i,j) = 2 * _POUT.begin()->second - field.p(i - 1, j);

    }
}
