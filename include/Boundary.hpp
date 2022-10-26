#pragma once

#include <vector>

#include "Cell.hpp"
#include "Fields.hpp"

/**
 * @brief Abstact of boundary conditions.
 *
 * This class patches the physical values to the given field.
 */
class Boundary {
  public:
    /**
     * @brief Main method to patch the boundary conditons to given field and
     * grid
     *
     * @param[in] Field to be applied
     */
    virtual void apply(Fields &field) = 0;
    virtual ~Boundary() = default;
};

/**
 * @brief Fixed wall boundary condition for the outer boundaries of the domain.
 * Dirichlet for velocities, which is zero, Neumann for pressure
 */
class FixedWallBoundary : public Boundary {
  public:
    FixedWallBoundary(std::vector<Cell *> cells);
    FixedWallBoundary(std::vector<Cell *> cells, std::map<int, dtype> wall_temperature);
    virtual ~FixedWallBoundary() = default;
    virtual void apply(Fields &field);

  private:
    std::vector<Cell *> _cells;
    std::map<int, dtype> _wall_temperature;
};

/**
 * @brief Moving wall boundary condition for the outer boundaries of the domain.
 * Dirichlet for velocities for the given velocity parallel to the fluid,
 * Neumann for pressure
 */
class MovingWallBoundary : public Boundary {
  public:
    MovingWallBoundary(std::vector<Cell *> cells, dtype wall_velocity);
    MovingWallBoundary(std::vector<Cell *> cells, std::map<int, dtype> wall_velocity,
                       std::map<int, dtype> wall_temperature);
    virtual ~MovingWallBoundary() = default;
    virtual void apply(Fields &field);

  private:
    std::vector<Cell *> _cells;
    std::map<int, dtype> _wall_velocity;
    std::map<int, dtype> _wall_temperature;
};

/**
 * @brief Inflow boundary condition for the inlet boundaries of the domain.
 * Dirichlet for velocities for the given velocity parallel to the fluid,
 * Neumann for pressure
 */

class InFlowBoundary : public Boundary {
  public:
    InFlowBoundary(std::vector<Cell *> cells, dtype UIN, dtype VIN);
    InFlowBoundary(std::vector<Cell *> cells, std::map<int, dtype> UIN, std::map<int, dtype> VIN,
                       std::map<int, dtype> wall_temperature);
    virtual ~InFlowBoundary() = default;
    virtual void apply(Fields &field);

  private:
    std::vector<Cell *> _cells;
    std::map<int, dtype> _UIN;
    std::map<int, dtype> _VIN;
    std::map<int, dtype> _wall_temperature;
};

/**
 * @brief Outflow boundary condition for the outlet boundaries of the domain.
 * Neumann for velocities for the given velocity parallel to the fluid,
 * Dirichlet for pressure
 */

class OutFlowBoundary : public Boundary {
  public:
    OutFlowBoundary(std::vector<Cell *> cells, dtype POUT);
    OutFlowBoundary(std::vector<Cell *> cells, std::map<int, dtype> POUT,
                       std::map<int, dtype> wall_temperature);
    virtual ~OutFlowBoundary() = default;
    virtual void apply(Fields &field);

  private:
    std::vector<Cell *> _cells;
    std::map<int, dtype> _POUT;
    std::map<int, dtype> _wall_temperature;
};
