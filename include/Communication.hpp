#pragma once

#include"Datastructures.hpp"
#include <map>

/**
 * @brief Methods to hadle communication
 *
 */
class Communication{
    public:

    /**
     * @brief Method to initialize communication
     *
     * @param[in] Number of arguments in command line
     * @param[in] Arguments in command line
     */
    static void init_parallel(int argc, char **argv);

    /**
     * @brief Method to exchange field data among processes
     *
     * @param[in] Field variable to be exchanged
     */
    static void communicate(Matrix<double> &);

    /**
     * @brief Method to find minimum across all processes
     *
     * @param[in] rank of the process
     * @param[out] minimum value across all processes
     */
    static double reduce_min(double);

    /**
     * @brief Method to find maximum across all processes
     *
     * @param[in] rank of the process
     * @param[out] maximum value across all processes
     */
    static double reduce_max(double);

    /**
     * @brief Method to find sum of a variable across all processes
     *
     * @param[in] rank of the process
     * @param[out] Total sum across all processes
     */
    static double reduce_sum(double);

    /**
     * @brief Method to broadcast a value to all processes
     *
     * @param[in] variable to be broadcasted
     */
    static void broadcast(double);

    /**
     * @brief Returns rank of current process
     *
     * @param[out] rank of current process
     */
    static int get_rank();

    /**
     * @brief Returns number of processors
     *
     * @param[out] number of processors
     */
    static int get_size();
    
    /**
     * @brief Method to finalize communication
     */
    static void finalize();

    inline static int iproc, jproc;

    private:

    /**
     * @brief Method to assign neighbours to each process
     *
     * @param[in] rank of the process
     * @param[out] A map of it neighbours
     */
    static std::map<char, int> _assign_neighbours(int);

};