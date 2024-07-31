// DSMC Collision Test
//   Usage:
//     srun ./DSMC_collision
//                  <nx> [<ny>...] <Np> <Nt> <collisionModel>
//                  <timeStep> --bufferSize <buffSize> --info <verbosity>
//     nx             = Number of cell-centered points in the x-direction
//     ny          = Number of cell-centered points in the y-direction
//     nz             = Number of cell-centered points in the z-direction
//     totalP             = Total number of macro-particles in the simulation
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//     sover            = Field solver to use (FFT, FFTW, or FFTS)
// computeSelfField = Whether to compute the self-field

//     Example:
//     srun ./Nanbu 64 64 64 100000 40 0.01 FFT false  --info 10
//     srun  ./Nanbu nr[0] nr[1] nr[2] totalP solver computeSelfField

constexpr unsigned Dim = 3;       // Simulation dimension
using T                = double;  // Type for simulation precision
const char* TestName   = "Nanbu_collision";

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "Manager/PicManager.h"
#include "datatypes.h"
#include "NanbuManager.h"

long long startTimestamp = timestamp();
// std::string startFormattedTimestamp = formattedTimestamp();


void convergenceTest(int arg, char* argv[]) {
    Inform msg(TestName);

    int realizations    = std::atoi(argv[arg++]); // how often to run each simulation
    double nu0t_final   = std::atof(argv[arg++]);
    //int timesteps       = std::atoi(argv[arg++]);
    std::string method  = argv[arg++]; // should be Nanbu or TakAbe
    int NParticles      = std::atoi(argv[arg++]);
    double nu0dT        = std::atof(argv[arg++]);
    bool output_debug   = std::string(argv[arg++]) == "true";

    double T_total = 0.008/3 + 2*0.01/3;
    double rho     = 0.768; // 10^18m^-3 = 0.768 eV^3 (natural units) //1e6 in lnL, since lnL uses n in cm^-3
    double lnL     = std::sqrt(T_total / (4*pi*rho)); // 10.0; //23 - std::log(std::sqrt(rho)*std::pow(T_total, -1.5)); //   // 
    double nu0     = pi*std::sqrt(2.0)*rho*lnL/std::pow(T_total, 1.5); // uses lnL=10.0, 246
    msg << "nu0: " << nu0 << endl;

    // First run for different dt values
    int every_it = realizations / 10;
    for (int i = 0; i < realizations; ++i) {
        // Calculate nu0 and therefore necessary steps (and set timestepsize) to reach nu0t_final
        // TODO
        // Now create the manager and start the simulation
        //double t_final             = nu0t_final/nu0;
        double timestepsize        = nu0dT; // nu0
        int timesteps              = nu0t_final/nu0dT;
        // timestepsize        = 2.6e-3; //nu0dT/nu0;
        // timesteps           = 100; //nu0t_final/nu0dT;
        Vector_t<int, Dim> nr_grid = {1, 1, 1}; // not really needed in this case
        double lbt                 = 0.01; // Not used here
        bool computeSelfField      = false;
        bool computeCollisions     = true;
        std::string solver         = "FFT";
        double tau                 = 1.0; // not relevant here
        double timestep_fraction   = 1.0; // not relevant here (only Nanbu2 and Bird...)
        double mass_per_particle   = 1.0;

        NanbuManager<T, Dim> manager(NParticles, timesteps, nr_grid, lbt, solver, computeSelfField, computeCollisions, method, tau, 
                                    timestep_fraction, mass_per_particle, "convergence", 42+i, i, output_debug, 1.0, false,
                                     startTimestamp, 1.0); 
        // manager.setSamplingVelocityScale(1.0); // Not relevant here
        manager.setTimestepsize(timestepsize); // Overwrite timestepzsize
        manager.pre_run();

        if (output_debug || every_it==0 || i%every_it==0) {
            std::cout << "Running simulation " << i << "/" << realizations << " with timestepsize: " << timestepsize << std::endl;
        }
        manager.run(manager.getNt());
    }


    // Then run everything for different N values

}

void deltaTest(int arg, char* argv[]) {
    Inform msg(TestName);

    int gridsize           = std::atoi(argv[arg++]); 
    int realizations       = std::atoi(argv[arg++]); // how often to run each simulation
    double t_final         = std::atof(argv[arg++]);
    //int timesteps       = std::atoi(argv[arg++]);
    std::string method     = argv[arg++]; // should be Nanbu or TakAbe
    int NParticles         = std::atoi(argv[arg++]);
    double dx_ratio        = std::atof(argv[arg++]); // dx = L * dx_ratio (initial sampling cube size)
    int timesteps          = std::atoi(argv[arg++]);
    bool adjust_field_dims = std::string(argv[arg++]) == "true";
    bool computeCollisions = std::string(argv[arg++]) == "true";
    bool output_debug      = std::string(argv[arg++]) == "true";
    // initial gridsize  realization t_final algo  N_part N_steps dx_ratio use_collision debug_outp

    // 32 1 5.0 TakAbe 2000 0.01 200 true false --info 10

    //double T_total = 0.008/3 + 2*0.01/3;
    //double rho     = 0.768; // 10^18m^-3 = 0.768 eV^3 (natural units) //1e6 in lnL, since lnL uses n in cm^-3
    //double lnL     = std::sqrt(T_total / (4*pi*rho)); // 10.0; //23 - std::log(std::sqrt(rho)*std::pow(T_total, -1.5)); //   // 
    //double nu0     = pi*std::sqrt(2.0)*rho*lnL/std::pow(T_total, 1.5); // uses lnL=10.0, 246
    //msg << "nu0: " << nu0 << endl;
    double timestepsize = t_final / timesteps;

    // First run for different dt values
    int every_it = realizations / 10;
    for (int i = 0; i < realizations; ++i) {
        // Calculate nu0 and therefore necessary steps (and set timestepsize) to reach nu0t_final
        // TODO
        // Now create the manager and start the simulation
        //double t_final             = nu0t_final/nu0;
        //double timestepsize        = nu0dT; // nu0
        //int timesteps              = nu0t_final/nu0dT;
        // timestepsize        = 2.6e-3; //nu0dT/nu0;
        // timesteps           = 100; //nu0t_final/nu0dT;
        Vector_t<int, Dim> nr_grid = {gridsize, gridsize, gridsize}; // not really needed in this case
        double lbt                 = 0.01; // Not used here
        bool computeSelfField      = true;
        std::string solver         = "FFT";
        double tau                 = 1.0; // not relevant here
        double timestep_fraction   = 1.0; // not relevant here (only Nanbu2 and Bird...)
        double mass_per_particle   = 1.0;

        NanbuManager<T, Dim> manager(NParticles, timesteps, nr_grid, lbt, solver, computeSelfField, computeCollisions, method, tau, 
                                     timestep_fraction, mass_per_particle, "delta", 42*i, i, output_debug, dx_ratio, adjust_field_dims,
                                     startTimestamp, 1.0); 
        // manager.setSamplingVelocityScale(1.0); // Not relevant here
        manager.setTimestepsize(timestepsize); // Overwrite timestepzsize
        manager.pre_run();

        if (output_debug || every_it==0 || i%every_it==0) {
            std::cout << "Running simulation " << i+1 << "/" << realizations << " with timestepsize: " << timestepsize << std::endl;
        }
        manager.run(manager.getNt());
    }
}

void runSphereTest(int arg, char* argv[]) {
    Inform msg(TestName);
    Vector_t<int, Dim> nr = std::atoi(argv[arg++]);
    // 1. input grids number : nr[0] nr[1] nr[2] 64 64 64
    //for (unsigned d = 0; d < Dim; d++) {
    //    nr[d] = std::atoi(argv[arg++]); //64 64 64
    //}
    // 2. particle number total P 
    size_type totalP           = std::atoll(argv[arg++]); //10000
    // 3. time steps
    int nt                     = std::atoi(argv[arg++]); //100

    // 4. lbm 
    double lbt = 0.01; // std::atof(argv[arg++]); //0.01

    // 5.  field solver
    std::string solver = "FFT"; // argv[arg++]; 

    // 6. true or false
    bool computeSelfField = std::string(argv[arg++]) == "true";
    
    // 6.5 true or false
    bool computeCollisions = std::string(argv[arg++]) == "true";

    bool output_debug      = std::string(argv[arg++]) == "true";

    bool adjust_field_dims = std::string(argv[arg++]) == "true";

    bool useDebyeLengthCollisionCutoff = std::string(argv[arg++]) == "true";

    // 7. Nanbu / Bird / Nanbu2
    std::string methods = argv[arg++];

    // 8. tau
    double tau = 1.0; // std::atof(argv[arg++]);

    // 8.5 custom timestepsize
    double timestepsize = std::atof(argv[arg++]);

    // 8.55
    double sampling_vector_scale  = std::atof(argv[arg++]);

    double confinementForceAdjust = std::atof(argv[arg++]);

    int realizations              = std::atoi(argv[arg++]); // how often to run each simulation

    // 9. Set a timestep fraction to define rho \propto tau/rho
    double timestep_fraction = 1.0;
    //if (methods == "Bird") { timestep_fraction = 1.0; }

    // 10. Set the mass_per_particle constant
    double mass_per_particle = 510936.6237; //9.1e-31; // electron
    int every_it = realizations / 10;
    for (int i = 0; i < realizations; ++i) {
        NanbuManager<T, Dim> manager(totalP, nt, nr, lbt, solver, computeSelfField, computeCollisions, methods, tau, 
                                     timestep_fraction, mass_per_particle, "sphere", 42*i, i, output_debug, 1.0, adjust_field_dims,
                                     startTimestamp, confinementForceAdjust);  // FFT

        manager.setSamplingVelocityScale(sampling_vector_scale);
        manager.setCollisionOverwrite(useDebyeLengthCollisionCutoff);
        manager.setTimestepsize(timestepsize); // Overwrite timestepzsize
        manager.pre_run();

        if (output_debug || every_it==0 || i%every_it==0) {
            std::cout << "Running simulation " << i+1 << "/" << realizations << " with timestepsize: " << timestepsize << std::endl;
        }
        manager.run(manager.getNt());
    }
    // Create an instance of the collision manager
    

    // Set scale by which the velocity vectors with initial zero momentum spread are set (TODO: remove later...)!
    

    // Overwrite timestepzsize (set to )
    //manager.setTimestepsize(timestepsize);
    
    // Initialize particles and prepare for the simulation
    //manager.pre_run();

    // msg << "Starting collision computations ..." << endl;

    // Run the DSMC collision computations
    //manager.run(manager.getNt());  // Simulate for 'nt' time steps
}


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        // Parse command-line arguments
        int arg = 1;

        std::string initial_distr = argv[arg++];
        msg << "Initial distribution: " << initial_distr << endl;
        if (initial_distr == "sphere") {
            runSphereTest(arg, argv);
        } else if (initial_distr == "convergence") {
            convergenceTest(arg, argv);
        } else if (initial_distr == "delta") {
            deltaTest(arg, argv);
        } else {
            msg << "Unknown initial distribution: " << initial_distr << endl;
        }
        
        msg << "Simulation completed." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timingNB.dat"));

    }
    ippl::finalize();

    return 0;
}



