#include "Case.hpp"
#include "Enums.hpp"

#include <algorithm>
#ifdef GCC_VERSION_9_OR_HIGHER
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#ifdef GCC_VERSION_9_OR_HIGHER
namespace filesystem = std::filesystem;
#else
namespace filesystem = std::experimental::filesystem;
#endif

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridWriter.h>
#include <vtkTuple.h>

Case::Case(std::string file_name, int argn, char **args) {
    // Read input parameters
    const int MAX_LINE_LENGTH = 1024;
    std::ifstream file(file_name);
    dtype nu;      /* viscosity   */
    dtype UI;      /* velocity x-direction */
    dtype VI;      /* velocity y-direction */
    dtype PI;      /* pressure */
    dtype GX;      /* gravitation x-direction */
    dtype GY;      /* gravitation y-direction */
    dtype xlength; /* length of the domain x-dir.*/
    dtype ylength; /* length of the domain y-dir.*/
    dtype dt;      /* time step */
    int imax;       /* number of cells x-direction*/
    int jmax;       /* number of cells y-direction*/
    dtype gamma;   /* uppwind differencing factor*/
    dtype omg;     /* relaxation factor */
    dtype tau;     /* safety factor for time step*/
    int itermax;    /* max. number of iterations for pressure per time step */
    dtype eps;     /* accuracy bound for pressure*/

    dtype UIN;     /* inlet velocity */
    dtype VIN;     /* outlet velocity */
    int num_of_walls; /* number of walls */

    dtype TI;      /* initial temperature */
    dtype beta;    /* thermal expansion co-efficient */
    dtype alpha;   /* thermal diffusivity */

   
   dtype wall_temp_a; // a is adiabatic(5)
   dtype wall_temp_h; // h is hot wall (6)
   dtype wall_temp_c; // c is cold wall (7)
    
    std::vector<dtype> wall_temp;
    if (file.is_open()) {

        std::string var;
        while (!file.eof() && file.good()) {
            file >> var;
            if (var[0] == '#') { /* ignore comment line*/
                file.ignore(MAX_LINE_LENGTH, '\n');
            } else {
                if (var == "xlength") file >> xlength;
                if (var == "ylength") file >> ylength;
                if (var == "nu") file >> nu;
                if (var == "t_end") file >> _t_end;
                if (var == "dt") file >> dt;
                if (var == "omg") file >> omg;
                if (var == "eps") file >> eps;
                if (var == "tau") file >> tau;
                if (var == "gamma") file >> gamma;
                if (var == "dt_value") file >> _output_freq;
                if (var == "UI") file >> UI;
                if (var == "VI") file >> VI;
                if (var == "GX") file >> GX;
                if (var == "GY") file >> GY;
                if (var == "PI") file >> PI;
                if (var == "itermax") file >> itermax;
                if (var == "imax") file >> imax;
                if (var == "jmax") file >> jmax;
                if (var == "UIN") file >> UIN;
                if (var == "VIN") file >> VIN;
                if (var == "geo_file") file >> _geom_name;
                if (var == "num_of_walls") file >> num_of_walls;
                if (var=="wall_temp_5") {file>>wall_temp_a; wall_temp.push_back(wall_temp_a);}
                if (var=="wall_temp_6") {file>>wall_temp_h; wall_temp.push_back(wall_temp_h);}
                if (var=="wall_temp_7") {file>>wall_temp_c; wall_temp.push_back(wall_temp_c);}
                if (var == "TI") file >> TI;
                if (var == "beta") file >> beta;
                if (var == "alpha") file >> alpha;
                
            }
        }
    }
    
    file.close();

    // bool variable isHeatTransfer checks if heat transfer occurs in the system
    bool isHeatTransfer = false;
    std::map<int, dtype> wall_temp_a_map, wall_temp_h_map, wall_temp_c_map;
    if (!wall_temp.empty()){
        isHeatTransfer = true;
        wall_temp_a_map.insert(std::pair<int, dtype>(GEOMETRY_PGM::adiabatic_id, wall_temp_a));
        wall_temp_h_map.insert(std::pair<int, dtype>(GEOMETRY_PGM::hot_id, wall_temp_h));
        wall_temp_c_map.insert(std::pair<int, dtype>(GEOMETRY_PGM::cold_id, wall_temp_c));
    }
      
    std::map<int, dtype> wall_vel;
    if (_geom_name.compare("NONE") == 0) {
        wall_vel.insert(std::pair<int, dtype>(LidDrivenCavity::moving_wall_id, LidDrivenCavity::wall_velocity));
    }


    // Set file names for geometry file and output directory
    set_file_names(file_name);

    // Build up the domain
    Domain domain;
    domain.dx = xlength / static_cast<dtype>(imax);
    domain.dy = ylength / static_cast<dtype>(jmax);
    domain.domain_size_x = imax;
    domain.domain_size_y = jmax;

    build_domain(domain, imax, jmax);

    _grid = Grid(_geom_name, domain);
    _field = Fields(nu, dt, tau, _grid.domain().size_x, _grid.domain().size_y, UI, VI, PI, TI, _grid, alpha, beta, isHeatTransfer, GX, GY); 

    _discretization = Discretization(domain.dx, domain.dy, gamma);
    _pressure_solver = std::make_unique<SOR>(omg);
    _max_iter = itermax;
    _tolerance = eps;

    // Construct boundaries
    if (not _grid.moving_wall_cells().empty()) {
        _boundaries.push_back(
            std::make_unique<MovingWallBoundary>(_grid.moving_wall_cells(), LidDrivenCavity::wall_velocity));
    }
    if (not _grid.fixed_wall_cells().empty()) {
        if (_field.isHeatTransfer()) {
            _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_cells(), wall_temp_a_map));
            _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_cells(), wall_temp_h_map));
            _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_cells(), wall_temp_c_map));
        }
        else
            _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_cells()));
    }
    if (not _grid.inflow_cells().empty()) {
        _boundaries.push_back(std::make_unique<InFlowBoundary>(_grid.inflow_cells(), UIN, VIN));
    }
    if (not _grid.outflow_cells().empty()) {
        _boundaries.push_back(std::make_unique<OutFlowBoundary>(_grid.outflow_cells(), GEOMETRY_PGM::POUT));
    }

    cuda_solver.initialize(_field, _grid, UIN, VIN, wall_temp_a, wall_temp_h, wall_temp_c, omg);
}

void Case::set_file_names(std::string file_name) {
    std::string temp_dir;
    bool case_name_flag = true;
    bool prefix_flag = false;

    for (int i = file_name.size() - 1; i > -1; --i) {
        if (file_name[i] == '/') {
            case_name_flag = false;
            prefix_flag = true;
        }
        if (case_name_flag) {
            _case_name.push_back(file_name[i]);
        }
        if (prefix_flag) {
            _prefix.push_back(file_name[i]);
        }
    }

    for (int i = file_name.size() - _case_name.size() - 1; i > -1; --i) {
        temp_dir.push_back(file_name[i]);
    }

    std::reverse(_case_name.begin(), _case_name.end());
    std::reverse(_prefix.begin(), _prefix.end());
    std::reverse(temp_dir.begin(), temp_dir.end());

    _case_name.erase(_case_name.size() - 4);
    _dict_name = temp_dir;
    _dict_name.append(_case_name);
    _dict_name.append("_Output");

    if (_geom_name.compare("NONE") != 0) {
        _geom_name = _prefix + _geom_name;
    }

    // Create output directory
    filesystem::path folder(_dict_name);
    try {
        filesystem::create_directory(folder);
    } catch (const std::exception &e) {
        std::cerr << "Output directory could not be created." << std::endl;
        std::cerr << "Make sure that you have write permissions to the "
                     "corresponding location"
                  << std::endl;
    }
}

/**
 * This function is the main simulation loop. In the simulation loop, following steps are required
 * - Calculate and apply boundary conditions for all the boundaries in _boundaries container
 *   using apply() member function of Boundary class
 * - Calculate fluxes (F and G) using calculate_fluxes() member function of Fields class.
 *   Flux consists of diffusion and convection part, which are located in Discretization class
 * - Calculate right-hand-side of PPE using calculate_rs() member function of Fields class
 * - Iterate the pressure poisson equation until the residual becomes smaller than the desired tolerance
 *   or the maximum number of the iterations are performed using solve() member function of PressureSolver class
 * - Calculate the velocities u and v using calculate_velocities() member function of Fields class
 * - Calculat the maximal timestep size for the next iteration using calculate_dt() member function of Fields class
 * - Write vtk files using output_vtk() function
 *
 * Please note that some classes such as PressureSolver, Boundary are abstract classes which means they only provide the
 * interface. No member functions should be defined in abstract classes. You need to define functions in inherited
 * classes such as MovingWallBoundary class.
 *
 * For information about the classes and functions, you can check the header files.
 */
void Case::simulate() {

    dtype t = 0.0;
    dtype dt = *(_field.dt());
    int timestep = 0;
    dtype output_counter = _output_freq;
    dtype res; //Residual for Pressure SOR
    int iter;

    output_vtk(timestep); // write the zeroth timestep

    cuda_solver.pre_process(_field, _grid, _discretization, dt);

    while(t < _t_end){

        dt = cuda_solver.calc_dt();
        t = t + dt;
        ++timestep;

        cuda_solver.apply_boundary();
        if (_field.isHeatTransfer()) { 
            cuda_solver.calc_T();
        }
        cuda_solver.calc_fluxes();
        cuda_solver.calc_rs();
        cuda_solver.calc_pressure(_max_iter, _tolerance, t, dt);
        //cuda_solver.calc_pressure_direct_solve();
        //cuda_solver.apply_boundary();
        cuda_solver.calc_velocities();

        if(t >= output_counter) {
            cuda_solver.post_process(_field);
            output_vtk(timestep);
            output_counter += _output_freq;
        }

        // // Original Implementation
        // dt = _field.calculate_dt(_grid);
        // t = t + dt;
        // ++timestep;
        // for (int i = 0; i < _boundaries.size(); i++) {
        //     _boundaries[i]->apply(_field);
        // }
        // _field.calculate_temperatures(_grid);
        // _field.calculate_fluxes(_grid);
        // _field.calculate_rs(_grid);
        // iter = 0;
        // do{
        //     for (int i = 0; i < _boundaries.size(); i++) {
        //         _boundaries[i]->apply(_field);
        //     }
        //     res = _pressure_solver->solve(_field, _grid, _boundaries);
        //     iter++;
        // }while(res > _tolerance && iter < _max_iter);
        // if (iter == _max_iter) {
        //     std::cout << "Max iteration reached at " << t<<" s \n";
        // }
        // std::cout << "Time = " << std::setw(12) << t << " Residual = "<< std::setw(12) << res <<
        // " Iter = " << std::setw(8) << iter << " dt = " << std::setw(12) << dt << '\n';
        // _field.calculate_velocities(_grid);
        // if(t >= output_counter) {
        //     output_vtk(timestep);
        //     output_counter += _output_freq;
        // }
    }
    std::cout<<"\n";

}

void Case::output_vtk(int timestep, int my_rank) {
    // Create a new structured grid
    vtkSmartPointer<vtkStructuredGrid> structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();

    // Create grid
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    dtype dx = _grid.dx();
    dtype dy = _grid.dy();

    dtype x = _grid.domain().imin * dx;
    dtype y = _grid.domain().jmin * dy;

    { y += dy; }
    { x += dx; }

    dtype z = 0;
    for (int col = 0; col < _grid.domain().size_y + 1; col++) {
        x = _grid.domain().imin * dx;
        { x += dx; }
        for (int row = 0; row < _grid.domain().size_x + 1; row++) {
            points->InsertNextPoint(x, y, z);
            x += dx;
        }
        y += dy;
    }

    // Specify the dimensions of the grid
    structuredGrid->SetDimensions(_grid.domain().size_x + 1, _grid.domain().size_y + 1, 1);
    structuredGrid->SetPoints(points);

    // Pressure Array
    vtkDoubleArray *Pressure = vtkDoubleArray::New();
    Pressure->SetName("pressure");
    Pressure->SetNumberOfComponents(1);

    // Temparature Array
    vtkDoubleArray *Temparature = vtkDoubleArray::New();
    Temparature->SetName("temparature");
    Temparature->SetNumberOfComponents(1);

    // Velocity Array
    vtkDoubleArray *Velocity = vtkDoubleArray::New();
    Velocity->SetName("velocity");
    Velocity->SetNumberOfComponents(3);

    // Print pressure and temperature from bottom to top
    for (int j = 1; j < _grid.domain().size_y + 1; j++) {
        for (int i = 1; i < _grid.domain().size_x + 1; i++) {
            dtype pressure = _field.p(i, j);
            Pressure->InsertNextTuple(&pressure);

            if (_field.isHeatTransfer()) {
            dtype temparature = _field.t(i, j);
            Temparature->InsertNextTuple(&temparature);
            }
        }
    }

    // Temp Velocity
    dtype vel[3];
    vel[2] = 0; // Set z component to 0

    // Print Velocity from bottom to top
    for (int j = 0; j < _grid.domain().size_y + 1; j++) {
        for (int i = 0; i < _grid.domain().size_x + 1; i++) {
            vel[0] = (_field.u(i, j) + _field.u(i, j + 1)) * 0.5;
            vel[1] = (_field.v(i, j) + _field.v(i + 1, j)) * 0.5;
            Velocity->InsertNextTuple(vel);
        }
    }

    // Add Pressure to Structured Grid
    structuredGrid->GetCellData()->AddArray(Pressure);

    // Add Temparature to Structured Grid
    if (_field.isHeatTransfer())
    structuredGrid->GetCellData()->AddArray(Temparature);

    // Add Velocity to Structured Grid
    structuredGrid->GetPointData()->AddArray(Velocity);


    /* This section hides all the obstacles (fixed wall cells) on the vtk file */

    for (auto currentCell: _grid.fixed_wall_cells()){
        int i = currentCell->i();
        int j = currentCell->j();

        if (j > 0 && i > 0 && j <= _grid.jmax() && i <= _grid.imax()){
            int id = (j - 1) * _grid.imax() + (i - 1);
            structuredGrid->BlankCell(id);
        }
    }

    // Write Grid
    vtkSmartPointer<vtkStructuredGridWriter> writer = vtkSmartPointer<vtkStructuredGridWriter>::New();

    // Create Filename
    std::string outputname =
        _dict_name + '/' + _case_name + "_" + std::to_string(timestep) + ".vtk";

    writer->SetFileName(outputname.c_str());
    writer->SetInputData(structuredGrid);
    writer->Write();
}

void Case::build_domain(Domain &domain, int imax_domain, int jmax_domain) {
    domain.imin = 0;
    domain.jmin = 0;
    domain.imax = imax_domain + 2;
    domain.jmax = jmax_domain + 2;
    domain.size_x = imax_domain;
    domain.size_y = jmax_domain;
}
