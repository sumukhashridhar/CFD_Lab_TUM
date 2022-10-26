#pragma once
#include "Datastructures.hpp"
// If no geometry file is provided in the input file, lid driven cavity case
// will run by default. In the Grid.cpp, geometry will be created following
// PGM convention, which is:
// 0: fluid, 3: fixed wall, 4: moving wall
// namespace LidDrivenCavity

namespace LidDrivenCavity {
const int moving_wall_id = 8;
const int fixed_wall_id = 3;
const dtype wall_velocity = 1.0;
} 

// If geometry file is provided in the input file, the PGM convention is:
// 0: fluid, 1: inflow 2: outflow 3: fixed wall 5: adiabatic wall
// 6: hot fixed wall 7: cold fixed wall 8: moving wall
// namespace GEOMETRY_PGM

namespace GEOMETRY_PGM {
const int moving_wall_id = 8; 
const int fixed_wall_id = 3;
const int inflow_id = 1;
const int outflow_id = 2;
const dtype POUT = 0.0;
const int adiabatic_id = 5;
const int hot_id = 6;
const int cold_id = 7; 
}

enum class border_position {
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,
};

namespace border {
const int TOP = 0;
const int BOTTOM = 1;
const int LEFT = 2;
const int RIGHT = 3;
} // namespace border

enum class cell_type {

    FLUID,
    FIXED_WALL,
    MOVING_WALL,
    INFLOW,
    OUTFLOW,
    DEFAULT
};
