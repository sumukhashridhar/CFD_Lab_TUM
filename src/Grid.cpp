#include "Grid.hpp"
#include "Enums.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>

#include<Communication.hpp>
Grid::Grid(std::string geom_name, Domain &domain) {
    _domain = domain;
    _cells = Matrix<Cell>(_domain.size_x + 2, _domain.size_y + 2);

    if (geom_name.compare("NONE")) {
        std::vector<std::vector<int>> geometry_data(_domain.domain_size_x + 2,
                                                    std::vector<int>(_domain.domain_size_y + 2, 0));
        parse_geometry_file(geom_name, geometry_data); //do job done by build_lid.. cavity here
        assign_cell_types(geometry_data);
    } else {
        if (Communication::get_rank() == 0)
            std::cout<<"No geometry file found. Lid driven cavity case will be simulated";
        build_lid_driven_cavity();
    }
}

void Grid::build_lid_driven_cavity() {
    std::vector<std::vector<int>> geometry_data(_domain.domain_size_x + 2,
                                                std::vector<int>(_domain.domain_size_y + 2, 0));

    for (int i = 0; i < _domain.domain_size_x + 2; ++i) {
        for (int j = 0; j < _domain.domain_size_y + 2; ++j) {
            // Bottom, left and right walls: no-slip
            if (i == 0 || j == 0 || i == _domain.domain_size_x + 1) {
                geometry_data.at(i).at(j) = LidDrivenCavity::fixed_wall_id;
            }
            // Top wall: moving wall
            else if (j == _domain.domain_size_y + 1) {
                geometry_data.at(i).at(j) = LidDrivenCavity::moving_wall_id;
            }
        }
    }
    assign_cell_types(geometry_data);
}

void Grid::assign_cell_types(std::vector<std::vector<int>> &geometry_data) {

    int i = 0;
    int j = 0;

    for (int j_geom = _domain.jmin; j_geom < _domain.jmax; ++j_geom) {
        { i = 0; }
        for (int i_geom = _domain.imin; i_geom < _domain.imax; ++i_geom) {
            
            // Halo cells 
            if (i_geom == _domain.imin && i_geom != 0 && j_geom != 0 && j_geom != _domain.domain_size_y + 1){
                _cells(i, j) = Cell(i, j, cell_type::HALO);
                _halo_cells.push_back(&_cells(i, j));
            }
            else if (i_geom == _domain.imax - 1 && i_geom != _domain.domain_size_x + 1 && j_geom != 0 && j_geom != _domain.domain_size_y + 1){
                _cells(i, j) = Cell(i, j, cell_type::HALO);
                _halo_cells.push_back(&_cells(i, j));                
            }
            else if (j_geom == _domain.jmin && j_geom != 0 && i_geom != 0 && i_geom != _domain.domain_size_x + 1){
                _cells(i, j) = Cell(i, j, cell_type::HALO);
                _halo_cells.push_back(&_cells(i, j));
            }
            else if (j_geom == _domain.jmax - 1  && j_geom != _domain.domain_size_y + 1 && i_geom != 0 && i_geom != _domain.domain_size_x + 1){
                _cells(i, j) = Cell(i, j, cell_type::HALO);
                _halo_cells.push_back(&_cells(i, j));
            }


            // Non - Halo Cells
            else if (geometry_data.at(i_geom).at(j_geom) == 0) {
                _cells(i, j) = Cell(i, j, cell_type::FLUID);
                _fluid_cells.push_back(&_cells(i, j));
            } 
            else if (geometry_data.at(i_geom).at(j_geom) == LidDrivenCavity::moving_wall_id || geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::moving_wall_id) {
                _cells(i, j) = Cell(i, j, cell_type::MOVING_WALL, geometry_data.at(i_geom).at(j_geom));
                _moving_wall_cells.push_back(&_cells(i, j));
            } 
            else if (geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::inflow_id) {
                _cells(i, j) = Cell(i, j, cell_type::INFLOW, geometry_data.at(i_geom).at(j_geom));
                _inflow_cells.push_back(&_cells(i, j));
            } 
            else if (geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::outflow_id) {
                _cells(i, j) = Cell(i, j, cell_type::OUTFLOW, geometry_data.at(i_geom).at(j_geom));
                _outflow_cells.push_back(&_cells(i, j));
            }      
            else if (geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::fixed_wall_id) {
                _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                _fixed_wall_cells.push_back(&_cells(i, j));
            }
            else if (geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::adiabatic_id) {
                _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                _fixed_wall_cells.push_back(&_cells(i, j));
            }
            else if (geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::hot_id) {
                _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                _fixed_wall_cells.push_back(&_cells(i, j));
            }
            else if (geometry_data.at(i_geom).at(j_geom) == GEOMETRY_PGM::cold_id) {
                _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                _fixed_wall_cells.push_back(&_cells(i, j));
            }              
            else {
                if (i_geom == 0 or j_geom == 0 or i_geom == _domain.domain_size_x + 1 or j_geom == _domain.domain_size_y + 1) {
                    // Outer walls
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                    _fixed_wall_cells.push_back(&_cells(i, j));
                }
            }

            ++i;
        }
        ++j;
    }

    // Corner cell neighbour assignment
    // Bottom-Left Corner
    i = 0;
    j = 0;
    _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
    _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
    if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::TOP);
    }
    if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::RIGHT);
    }

    // Top-Left Corner
    i = 0;
    j = _domain.size_y + 1;
    _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
    _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
    if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::BOTTOM);
    }
    if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::RIGHT);
    }

    // Top-Right Corner
    i = _domain.size_x + 1;
    j = Grid::_domain.size_y + 1;
    _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
    _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
    if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::BOTTOM);
    }
    if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::LEFT);
    }

    // Bottom-Right Corner
    i = Grid::_domain.size_x + 1;
    j = 0;
    _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
    _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
    if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::TOP);
    }
    if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::LEFT);
    }

    // Bottom cells
    j = 0;
    for (int i = 1; i < _domain.size_x + 1; ++i) {
        _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
        _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
        _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
        if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::RIGHT);
        }
        if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::LEFT);
        }
        if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::TOP);
        }
    }

    // Top Cells
    j = Grid::_domain.size_y + 1;

    for (int i = 1; i < _domain.size_x + 1; ++i) {
        _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
        _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
        _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
        if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::RIGHT);
        }
        if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::LEFT);
        }
        if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::BOTTOM);
        }
    }

    // Left Cells
    i = 0;
    for (int j = 1; j < _domain.size_y + 1; ++j) {
        _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
        _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
        _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
        if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::RIGHT);
        }
        if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::BOTTOM);
        }
        if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::TOP);
        }
    }
    // Right Cells
    i = Grid::_domain.size_x + 1;
    for (int j = 1; j < _domain.size_y + 1; ++j) {
        _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
        _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
        _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
        if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::LEFT);
        }
        if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::BOTTOM);
        }
        if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::TOP);
        }
    }

    // Inner cells
    for (int i = 1; i < _domain.size_x + 1; ++i) {
        for (int j = 1; j < _domain.size_y + 1; ++j) {
            _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
            _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
            _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
            _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);

            if (_cells(i, j).type() != cell_type::FLUID) {
                if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::LEFT);
                }
                if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::RIGHT);
                }
                if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::BOTTOM);
                }
                if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::TOP);
                }
            }
        }
    }

    for (int i = 1; i < _domain.size_x + 1; ++i) {
        for (int j = 1; j < _domain.size_y + 1; ++j) {

            int num_fluid_neighbours{0};

            if (_cells(i, j).type() != cell_type::FLUID) {
                    if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
                        num_fluid_neighbours++;
                    }
                    if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
                        num_fluid_neighbours++;
                    }
                    if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
                        num_fluid_neighbours++;
                    }
                    if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
                        num_fluid_neighbours++;
                    }
            }
            // Checking for invalid cells    
            if (num_fluid_neighbours > 2) {
                std::cout << "Found invalid cell at " << i << "," << j << "\n" << "Exiting the simulation" << "\n";
                std::cout << "The cell has three fluid neighbour\n";
                exit (EXIT_FAILURE); 
            }
        }
    }
}

void Grid::parse_geometry_file(std::string filedoc, std::vector<std::vector<int>> &geometry_data) {

    int numcols, numrows, depth;

    std::ifstream infile(filedoc);
    std::stringstream ss;
    std::string inputLine = "";

    // First line : version
    getline(infile, inputLine);
    if (inputLine.compare("P2") != 0) {
        std::cerr << "First line of the PGM file should be P2" << std::endl;
    }

    // Second line : comment
    getline(infile, inputLine);

    // Continue with a stringstream
    ss << infile.rdbuf();
    // Third line : size
    ss >> numrows >> numcols;
    // Fourth line : depth
    ss >> depth;

    for (int col = numcols - 1; col > -1; --col) {
        for (int row = 0; row < numrows; ++row) {
            ss >> geometry_data[row][col];
        }
    }

    infile.close();

    //Scaling

    if ((_domain.domain_size_x) !=(numrows - 2) || (_domain.domain_size_y) !=(numcols - 2)) {
        
        if (((_domain.domain_size_x) % (numrows - 2) != 0 && (_domain.domain_size_y) % (numcols - 2) != 0)) {
            std::cout << "Error: Improper Scaling, you can only do integer scaling\n";
            exit(EXIT_FAILURE);
        }

        auto geometry_data_temperary = geometry_data;
        int x_factor = ((_domain.domain_size_x)) / (numrows - 2);
        int y_factor = ((_domain.domain_size_y)) / (numcols - 2);

        int i_temp, j_temp;
        // Interior cells
        for (int col = numcols - 2; col > 0; col--) {
            for (int row = 1; row < numrows - 1; row++){
                i_temp = (row -1) * x_factor + 1;
                j_temp = (col -1) * y_factor + 1;
                for(int j = j_temp; j < y_factor + j_temp; j++)
                    for (int i = i_temp; i < x_factor + i_temp; ++i)
                        geometry_data[i][j] = geometry_data_temperary[row][col];
            }
        }

        //Top and bottom edges
        for (int row = 1; row < numrows - 1; ++row){
            i_temp = (row -1) * x_factor + 1;
            for (int i = i_temp; i < x_factor + i_temp; ++i){
                geometry_data[i][0] = geometry_data_temperary[row][0];
                geometry_data[i][_domain.domain_size_y + 1] = geometry_data_temperary[row][numcols - 1];;
            }
        }

        //Left and right edges
        for (int col = numcols - 2; col > 0; col--){
            j_temp = (col -1) * y_factor + 1;
            for(int j = j_temp; j < y_factor + j_temp; j++){
                geometry_data[0][j] = geometry_data_temperary[0][col];
                geometry_data[_domain.domain_size_x + 1][j] = geometry_data_temperary[numrows - 1][col];
            }
        }

        //Corner cells
        geometry_data[0][0] = geometry_data_temperary[0][0];
        geometry_data[0][_domain.domain_size_y + 1] = geometry_data_temperary[0][numcols - 1];
        geometry_data[_domain.domain_size_x + 1][0] = geometry_data_temperary[numrows - 1][0];
        geometry_data[_domain.domain_size_x + 1][_domain.domain_size_y + 1] = geometry_data_temperary[numrows - 1][numcols - 1];
        }

}

int Grid::imax() const { return _domain.size_x; }
int Grid::jmax() const { return _domain.size_y; }

int Grid::imaxb() const { return _domain.size_x + 2; }
int Grid::jmaxb() const { return _domain.size_y + 2; }

Cell Grid::cell(int i, int j) const { return _cells(i, j); }

double Grid::dx() const { return _domain.dx; }

double Grid::dy() const { return _domain.dy; }

const Domain &Grid::domain() const { return _domain; }

const std::vector<Cell *> &Grid::fluid_cells() const { return _fluid_cells; }

const std::vector<Cell *> &Grid::fixed_wall_cells() const { return _fixed_wall_cells; }

const std::vector<Cell *> &Grid::moving_wall_cells() const { return _moving_wall_cells; }

const std::vector<Cell *> &Grid::inflow_cells() const { return _inflow_cells; }

const std::vector<Cell *> &Grid::outflow_cells() const { return _outflow_cells; }

const std::vector<Cell *> &Grid::halo_cells() const { return _halo_cells; }