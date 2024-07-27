#ifndef IPPL_NANBU_MANAGER_H
#define IPPL_NANBU_MANAGER_H

#include <algorithm>  // For std::shuffle
#include <cmath>
#include <memory> // 
#include <random>   // For std::mt19937 and std::random_device
#include <utility>  // For std::pair
#include <vector> 

#include "AllManager.h"
#include "DSMCHelpers.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

#include "DSMCHelpers.h" // Includes e.g. getA(s)...

#include <typeinfo>

using view_type      = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using mass_view_type = typename ippl::ParticleAttrib<double>::view_type; // ippl::detail::ViewType<double, 1>::view_type;

// define functions used in sampling particles
/*struct InducedHeatingDistributionFunctionsR {
  struct CDF{
       KOKKOS_INLINE_FUNCTION double operator()(double r, unsigned int d, const double *params_p) const {
           return x + (params_p[d * 2 + 0] / params_p[d * 2 + 1]) * Kokkos::sin(params_p[d * 2 + 1] * x);
       }
  };

  struct PDF{
       KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d, double const *params_p) const {
           return 1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x);
       }
  };

  struct Estimate{
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d, double const *params_p) const {
            return u + params_p[d] * 0.;
	}
  };
};*/

template <typename T, unsigned Dim>
class NanbuManager : public AllManager<T, Dim> {
protected:
    bool computeSelfField_m;
    bool computeCollisions_m;
    bool use_disorder_induced_heating_m = false; // only set to true if mentioned in initialize()
    std::string methods_m;
    double tau_m;
    double timestep_fraction_m;
    double mass_per_particle_m;
    // double normalization_reference_density = 1.0; // set in initialize() to the overall density for normalization of rho 

    double sampling_vector_scale = 0.01;
    std::string initial_distr;
    int seed;
    int realization_counter;
    double dx_ratio_m; // only used on delta testcase!
    bool adjust_field_dims;
    long long startTimestamp;
    double cellVolume = 1.0;
    double confinementForceAdjust = 1.0;

    // Vector_t<double, Dim> E_rescale = {1.0, 1.0, 1.0}; // Rescale E-field for the disorder induced heating process
    //double E_rescale = 1.0;

public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    NanbuManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
                 std::string& solver_, bool computeSelfField_, bool computeCollisions_,
                 std::string methods_, double tau_, double timestep_fraction_m_,
                 double mass_per_particle_m_, std::string initial_distr_, int seed_, int realization_counter_,
                 bool output_debug_, double dx_ratio_, bool adjust_field_dims_, long long startTimestamp_,
                 double confinementForceAdjust_)
        : AllManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, output_debug_)
        , computeSelfField_m(computeSelfField_)
        , computeCollisions_m(computeCollisions_)
        , methods_m(methods_) 
        , tau_m(tau_)
        , timestep_fraction_m(timestep_fraction_m_)
        , mass_per_particle_m(mass_per_particle_m_)
        , initial_distr(initial_distr_) 
        , seed(seed_)
        , realization_counter(realization_counter_)
        , dx_ratio_m(dx_ratio_)
        , adjust_field_dims(adjust_field_dims_)
        , startTimestamp(startTimestamp_) 
        , confinementForceAdjust(confinementForceAdjust_) {}
    ~NanbuManager() {}

    void pre_run() override {
        Inform m("Pre Run");
        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }
        this->decomp_m.fill(true);
        this->kw_m   = 2 * pi;                     // 2 * pi
        this->rmin_m = 0.0;                        // 0
        if (this->initial_distr == "sphere") {
            this->rmax_m = 506.84; //100e-6; // std::pow(this->totalP_m/1e6, 1.0/3); // std::pow(this->totalP_m, 1.0/3); // 4.64; // 506.842372; //100e-6; 
            this->Q_m    = -0.302822*this->totalP_m; // 1.0 * this->totalP_m; //-0.302822*this->totalP_m; //-1.6e-19 * this->totalP_m;
        } else if (this->initial_distr == "delta") {
            this->rmax_m = 1.0;                     // 100
            this->Q_m    = 1.0*this->totalP_m;
        } else if (this->initial_distr == "convergence") {
            this->rmax_m = std::pow(this->totalP_m/0.768, 1.0/3);
            this->Q_m    = 1.0*this->totalP_m;
        } else {
                std::cerr << "No collision model selected" << std::endl;
        }
        
        this->hr_m = this->rmax_m / this->nr_m;  // mesh spacing hr = rmax_m/[128,128,128]
        cellVolume = std::reduce(this->hr_m.begin(), this->hr_m.end(), 1., std::multiplies<double>());

        // Q = -\int\int f dx dv
         
                         //std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1.0, std::multiplies<double>()); // Set average charge "density" to -1.0 
                         //this->charge_per_particle_m * this->totalP_m; // Set total charge to N_particle*charge_per_particle
                         //std::reduce(this->rmax_m.begin(), this->rmax_m.end(), -1.,
                         //            std::multiplies<double>());  // Q = -1 // total charge
        this->origin_m = this->rmin_m;
        //this->setTimestepsize(1.0); // 2.15623e-13 // Set to some value, should be set in Nanbu.cpp using manager.setTimestepsize(...)
                         //this->timestep_fraction_m * this->tau_m / (this->totalP_m*this->mass_per_particle_m 
                         //               / std::reduce(this->rmax_m.begin(), this->rmax_m.end(), 1.0, std::multiplies<double>()));
        this->it_m     = 0;
        this->time_m   = 0.0;

        if (this->output_debug) {
            m << "Discretization:" << endl
            << "nt " << this->nt_m << " Np= " << this->totalP_m << " grid = " << this->nr_m << endl;
        }

        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver(std::make_shared<FieldSolver_t>(
            this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getE(),
            &this->fcontainer_m->getPhi()));

        this->fsolver_m->initSolver();

        this->setLoadBalancer(std::make_shared<LoadBalancer_t>(
            this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m));
        if (this->output_debug) m << "Mesh, solver, load balancer initialized." << endl;

        initializeParticles();
        //ippl::Comm->barrier();

        if (this->computeSelfField_m) {
            static IpplTimings::TimerRef DummySolveTimer = IpplTimings::getTimer("solveWarmup");
            IpplTimings::startTimer(DummySolveTimer);
            
            //applyExtendedBC(); // see function description for why this is important
            //if (this->output_debug) m << "Applied extended BC for warmup." << endl;
            if (this->adjust_field_dims) adjustFieldMeshDimensions(); // Adjust mesh sizing before solve

            this->fcontainer_m->getRho() = 0.0;
            //std::cout << "Yes1" << std::endl;
            this->fsolver_m->runSolver();
            if (this->output_debug) m << "Solver warmup: #1 completed." << endl;
            //std::cout << "Yes2" << std::endl;
            //Kokkos::fence();
            //ippl::Comm->barrier();
            //this->pcontainer_m->update(); 
            
            IpplTimings::stopTimer(DummySolveTimer);
            this->pcontainer_m->update();
            this->par2grid();
            if (this->output_debug) m << "Solver warmup: this->par2grid() completed." << endl;
            //ippl::Comm->barrier();
            //std::cout << "Yes3" << std::endl;

            static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
            IpplTimings::startTimer(SolveTimer);
            this->fsolver_m->runSolver();
            //std::cout << "Yes4" << std::endl;
            IpplTimings::stopTimer(SolveTimer);
            if (this->output_debug) m << "Solver warmup: #2 completed." << endl;

            this->grid2par();
            if (this->output_debug) m << "Solver warmup: this->grid2par() completed." << endl;

            resetBoundaries();
            //std::cout << "Yes5" << std::endl;
            //this->pcontainer_m->update(); // update since we have changed the particle positions
            //resetFieldDimensions(); // reset mesh sizing after solve and before calling .update()!
        }

        this->dump();
        //std::cout << "Yes6" << std::endl;

        Inform logs(NULL, "data/logs.txt", Inform::APPEND);
        if (this->realization_counter == 0) { //  && this->it_m == 0
            logs << "\n\n------------------------------------------\n" << startTimestamp << ", " << formattedTimestamp(startTimestamp) << endl;
            logs << "\t" << "Rmin:                      " << this->rmin_m << endl;
            logs << "\t" << "Rmax:                      " << this->rmax_m << endl;
            logs << "\t" << "Origin:                    " << this->origin_m << endl;
            logs << "\t" << "Total number of particles: " << this->totalP_m << endl;
            logs << "\t" << "Total number of timesteps: " << this->nt_m << endl;
            logs << "\t" << "Grid dimensions:           " << this->nr_m << endl;
            logs << "\t" << "Loadbalancer threshold:    " << this->lbt_m << endl;
            logs << "\t" << "Field solver:              " << this->solver_m << endl;
            logs << "\t" << "Compute self-field:        " << this->computeSelfField_m << endl;
            logs << "\t" << "Compute collisions:        " << this->computeCollisions_m << endl;
            logs << "\t" << "Collision Method:          " << this->methods_m << endl;
            logs << "\t" << "Nanbu2-tau parameter:      " << this->tau_m << endl;
            logs << "\t" << "Nanbu2 stepsize fraction:  " << this->timestep_fraction_m << endl;
            logs << "\t" << "Mass per particle:         " << this->mass_per_particle_m << endl;
            logs << "\t" << "Initial distribution:      " << this->initial_distr << endl;
            logs << "\t" << "Initial Seed               " << this->seed << " (may change with realization)" << endl;
            logs << "\t" << "Delta sampling ratio:      " << this->dx_ratio_m << endl;
            logs << "\t" << "Field solver grid adjust.: " << this->adjust_field_dims << endl;
        }

        if (this->output_debug) m << "Done";
    }

    void setSamplingVelocityScale(double scale_) { this->sampling_vector_scale = scale_; } // only used for "cold sphere" test case ("deprecated")

    void initializeParticles() {
        Inform m("Initialize Particles");
        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> rmax = this->rmax_m;
        auto* mesh                 = &this->fcontainer_m->getMesh();
        auto* FL                   = &this->fcontainer_m->getFL();

        // calculation the time for initializing particles
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        IpplTimings::startTimer(particleCreation);

        // total number of particles
        size_type totalP = this->totalP_m;
        // number of particles per processor
        size_type nlocal = totalP / ippl::Comm->size();
        // create particles
        this->pcontainer_m->create(nlocal);

        // can imideately calculate reference density (total number/total volume)
        // this->normalization_reference_density = totalP / mesh->getMeshVolume();

        // get gridsize in dimension 0

        // (1) set position : uniform random distribution
        // ----------------------------------------------------------------
        // double dx0 = mesh->getGridsize(0); 
        // std::mt19937_64 eng;
        // eng.seed(42);  
        // std::uniform_real_distribution<double> unif(dx0/2, (this->rmax_m[0])-dx0/2);
        // typename ParticleContainer_t::particle_position_type::HostMirror R_host =
        //     this->pcontainer_m->R.getHostMirror();
        // double sum_coord = 0.0;

        // for (unsigned long int i = 0; i < nlocal; i++) {
        //     ippl::Vector<double, 3> tmp_rand_R{unif(eng), unif(eng), unif(eng)};
        //     R_host(i) = tmp_rand_R;
        //     sum_coord += tmp_rand_R(0) + tmp_rand_R(1) + tmp_rand_R(2);
        // }

        // double global_sum_coord = 0.0;
        // ippl::Comm->reduce(sum_coord, global_sum_coord, 1, std::plus<double>());

        // if (ippl::Comm->rank() == 0) {
        //     std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
        // }
        // Kokkos::deep_copy(this->pcontainer_m->R.getView(), R_host);

        // ----------------------------------------------------------------

        // (2) delta function to initial particle position condition
        // ----------------------------------------------------------------

        // std::mt19937_64 eng;
        // eng.seed(42);
        // std::normal_distribution<double> gauss(0.0, 1.0);
        // typename ParticleContainer_t::particle_position_type::HostMirror R_host =
        //     this->pcontainer_m->R.getHostMirror();
        // double sum_coord = 0.0;

        // for (unsigned long int i = 0; i < nlocal; i++) {
        //     ippl::Vector<double, 3> tmp_rand_R{gauss(eng), gauss(eng), gauss(eng)};
        //     R_host(i) = tmp_rand_R;
        //     sum_coord += tmp_rand_R(0) + tmp_rand_R(1) + tmp_rand_R(2);
        // }

        // double global_sum_coord = 0.0;
        // ippl::Comm->reduce(sum_coord, global_sum_coord, 1, std::plus<double>());

        // if (ippl::Comm->rank() == 0) {
        //     std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
        // }
        // Kokkos::deep_copy(this->pcontainer_m->R.getView(), R_host);


        // ----------------------------------------------------------------

        // (3) how to set position function to initial particle position condition
        // ----------------------------------------------------------------
        // have some problem. address error need to be fix
        if (this->initial_distr == "sphere") {
            double r = 90.11657; // 17.78e-6;
            //double avg_rho = totalP / (4/3 * pi * std::pow(r, 3));
            std::mt19937_64 eng;
            eng.seed(this->seed);
            std::uniform_real_distribution<double> unif(-r, r);
            typename ParticleContainer_t::particle_position_type::HostMirror R_host =
                    this->pcontainer_m->R.getHostMirror();

            for (unsigned long int i = 0; i < nlocal; i++) {
                    double x, y, z, r_squared;
                    do {
                    x = unif(eng);
                    y = unif(eng);
                    z = unif(eng);
                    r_squared = x * x + y * y + z * z;
                    } while (r_squared > r * r);
                    
                    ippl::Vector<double, 3> tmp_rand_R{x, y, z};
                    R_host(i) = tmp_rand_R + this->rmax_m/2;
                }
            Kokkos::deep_copy(this->pcontainer_m->R.getView(), R_host);
            Kokkos::fence();
            
            this->pcontainer_m->update(); // update since particle sampling might be ober the domain boundaries???
            ippl::Comm->barrier();
            // ----------------------------------------------------------------


            //Vector_t<double, Dim> kw     = this->kw_m;
            //Vector_t<double, Dim> hr     = this->hr_m;
            //Vector_t<double, Dim> origin = this->origin_m;
            /*using DistR_t = ippl::random::Distribution<double, Dim, 2 * Dim, CustomDistributionFunctions>;
            double parR[2 * Dim];
            for(unsigned int i=0; i<Dim; i++){
                parR[i * 2   ]  = 0.05;
                parR[i * 2 + 1] = 0.5;
            }
            DistR_t distR(parR);*/
            // static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        
            // (1) set velocity Guassian center is  : Tx =4, Ty = 0.8, Tz =0.8 ; set  Vx =2, Vy =
            // 0.8944, Vz = 0.8944;
            // ----------------------------------------------------------------------
            // view_type* P = &(this->pcontainer_m->P.getView());
            // double mu[Dim];
            // double sd[Dim];
            // double vval = 1.37055e7;

            // for (unsigned i = 0; i < Dim; i++) {
            //     if (i == 0)
            //         mu[i] = 0, sd[i] = vval;  // initial Vx = 2
            //     else
            //         mu[i] = 0, sd[i] = vval; // 5e5 * std::sqrt(0.8)/2;  // initial Vy = 0.8944, Vz = 0.8944
            // }
            // int seed = 42;
            // Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));
            // Kokkos::parallel_for(nlocal, ippl::random::randn<double, Dim>(*P, rand_pool64, mu, sd));
            // Kokkos::fence();
            // ippl::Comm->barrier();
            // ----------------------------------------------------------------------

            // (2)set velocity 0
            // ----------------------------------------------------------------------
            // double vval = 1.37055e7;

            // vector_scale is now set as a member variable
            // double vector_scale = 416390; // Expected velocity magnitude per particle to get kB*T=196meV --> gives lambda_D = 1.56um, gridsize 1.5um
                                                // std::sqrt( 2*(4.005e-11/this->totalP_m)/this->mass_per_particle_m ); // scaling for velocity to match (energy per particle, 250MeV=4.005e-7J)  
            this->use_disorder_induced_heating_m = true; // important for additional "confinement kick"!
            std::uniform_real_distribution<double> unifo(-1.0, 1.0);
            unifo(eng);
            //std::normal_distribution<double> gauss(vval, 5e6);
            // 4.6e6 m/s is the velocity one electron needs to cross 1/100 of the simulation domain in one timestep.
            typename ParticleContainer_t::particle_position_type::HostMirror v_host = this->pcontainer_m->P.getHostMirror();
            for (unsigned long int i = 0; i < nlocal; i++) {
                Vector_t<double, Dim> vel{unifo(eng), unifo(eng), unifo(eng)}; // Create random vector
                vel /= normIPPLVector(vel, false); // normalize vector
                v_host(i) = vel * sampling_vector_scale; // assign vector and scale it according to overall energy

                // Different approach: sample in R-direction (so the sphere expands...)
                //Vector_t<double, Dim> vel       = R_host(i) - this->rmax_m/2; // use R_host from position sampling above...
                //v_host(i)                       = vel / normIPPLVector(vel, false) * sampling_vector_scale;
            }
            Kokkos::deep_copy(this->pcontainer_m->P.getView(), v_host);
            Kokkos::fence();
            //this->pcontainer_m->P = 0.0;
            // ----------------------------------------------------------------------

            IpplTimings::stopTimer(particleCreation);
        } else if (this->initial_distr == "convergence") {
            // First sample position uniformly 
            std::mt19937_64 eng;
            eng.seed(this->seed);  
            std::uniform_real_distribution<double> unif(0.0, this->rmax_m[0]);
            typename ParticleContainer_t::particle_position_type::HostMirror R_host = this->pcontainer_m->R.getHostMirror();
            for (unsigned long int i = 0; i < nlocal; i++) {
                ippl::Vector<double, 3> tmp_rand_R{unif(eng), unif(eng), unif(eng)};
                R_host(i) = tmp_rand_R;
            }
            Kokkos::deep_copy(this->pcontainer_m->R.getView(), R_host);

            // Then sample velocities according to anisotropic temperature distribution T_parallel=0.008
            double T_parallel = 0.008, T_perp = 0.01;
            std::normal_distribution<double> dist_parallel(0.0, std::sqrt(T_parallel));
            std::normal_distribution<double> dist_perp(0.0, std::sqrt(T_perp));
            typename ParticleContainer_t::particle_position_type::HostMirror P_host = this->pcontainer_m->P.getHostMirror();
            for (unsigned long int i = 0; i < nlocal; i++) {
                double v_parallel = dist_parallel(eng);
                double v_perp1 = dist_perp(eng);
                double v_perp2 = dist_perp(eng); //std::pow(1.0 / (2 * pi), 1.5) / (std::sqrt(T_parallel)*T_perp) * 
                                 //std::exp(-0.5 * (std::pow(v_parallel / std::sqrt(T_parallel), 2) + 
                                 //                 std::pow(v_perp1 / std::sqrt(T_perp), 2))); // respect the normalization of the distribution...
                P_host(i) = {v_perp1, v_perp2, v_parallel};
            }
            Kokkos::deep_copy(this->pcontainer_m->P.getView(), P_host);
        } else if (this->initial_distr == "delta") { 
            std::mt19937_64 eng;
            eng.seed(this->seed);  
            // First sample position uniformly in small cube around center
            std::uniform_real_distribution<double> unif(this->rmax_m[0]/2 *(1-this->dx_ratio_m), this->rmax_m[0]/2 *(1+this->dx_ratio_m));
            typename ParticleContainer_t::particle_position_type::HostMirror R_host = this->pcontainer_m->R.getHostMirror();
            for (unsigned long int i = 0; i < nlocal; i++) {
                ippl::Vector<double, 3> tmp_rand_R{unif(eng), unif(eng), unif(eng)};
                R_host(i) = tmp_rand_R;
            }
            Kokkos::deep_copy(this->pcontainer_m->R.getView(), R_host);

            // Then sample velocities (T_0 = 0)
            //typename ParticleContainer_t::particle_position_type::HostMirror P_host = this->pcontainer_m->P.getHostMirror();
            this->pcontainer_m->P = 0.0;
            //Kokkos::fence();
            //ippl::Comm->barrier();
            //for (unsigned long int i = 0; i < nlocal; i++) {
            //    P_host(i) = {unif(eng)*10, 10*unif(eng), 10*unif(eng)};
            //}
            //Kokkos::deep_copy(this->pcontainer_m->P.getView(), P_host);
        }
        //if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
        //    if (this->output_debug) m << "Starting first repartition" << endl;
            // IpplTimings::startTimer(domainDecomposition);
            //bool isFirstRepartition_m      = true;
            //const ippl::NDIndex<Dim>& lDom = FL->getLocalNDIndex();
            //const int nghost               = this->fcontainer_m->getRho().getNghost();
        //auto rhoview                   = this->fcontainer_m->getRho().getView();  
        //rhoview = 0.0; // enough for now...
            /*using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", this->fcontainer_m->getRho().getFieldRangePolicy(),
                KOKKOS_LAMBDA (const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec =
                        (args + lDom.first() - nghost + 0.5) * hr + origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = (normIPPLVector(xvec-this->rmax_m/2, false))>r ? 0.0 : 1/(4*pi*std::pow(r,3)/3); // avg_rho;??
                });
            Kokkos::fence();*/
        bool isFirstRepartition_m      = true;
        this->loadbalancer_m->initializeORB(FL, mesh);
        this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition_m);
            // IpplTimings::stopTimer(domainDecomposition);
        //}

        this->pcontainer_m->q = this->Q_m / this->totalP_m;
        this->pcontainer_m->m = this->mass_per_particle_m;

        if (this->output_debug) m << "particles created and initial conditions assigned " << endl;

        // Can only use this when compiling with CUDA
        /*cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        }*/

        this->pcontainer_m->update(); // because otherwise, the scatter(...) does not want to work -.-
        // this one line is the error that was missing and left our team in CSP awake for many nights. arrrgghhh D:

        Kokkos::fence();
        ippl::Comm->barrier();
    }

    void advance() override {
        Inform m("Advance");
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;
        // particle collision update velocity

        static IpplTimings::TimerRef adv = IpplTimings::getTimer("AdvanceComplete");
        IpplTimings::startTimer(adv);
        if (this->computeCollisions_m) {
            if (methods_m == "Nanbu") {
                AllCollisions(false);
                //NanbuCollision();  // binary collision model
            } else if (methods_m == "Bird") {
                BirdCollision();  // binary collision model
            } else if (methods_m == "Nanbu2") {
                Nanbu2Collision();  // binary collision model
            } else if (methods_m == "TakAbe") {
                AllCollisions(true);
            } else {
                std::cerr << "No collision model selected" << std::endl;
            }
        }

        double dt = this->dt_m; 

        // kick
        if (this->computeSelfField_m) {
            pc->P = pc->P + dt/2 * (pc->q / pc->m) * pc->E/e0;
        }
        // add another "kick" for the space confinement of the disorder induced heating process!
        //double R0 = 90.11657; // 17.78e-6;
        if (this->use_disorder_induced_heating_m) {
            disorderInducedHeatingConfinementUpdate(dt);
        }

        // solve
        size_type totalP        = this->totalP_m;
        int it                  = this->it_m;
        if (this->computeSelfField_m) {
            bool isFirstRepartition = false;
            if (this->loadbalancer_m->balance(totalP, it + 1 - it)) {
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL   = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
            }
            
            if (this->adjust_field_dims) adjustFieldMeshDimensions(); // need to adjust everytime. Otherwise streaming particles go out of the domain...
            this->par2grid();
            // applyExtendedBC(); // see function description for why this is important
            this->fsolver_m->runSolver();
            //pc->E = pc->E*this->E_rescale; // rescale because of domain rescaling???
            this->grid2par();
            resetBoundaries();
            //resetFieldDimensions(); // reset mesh before calling .update() (to properly apply bc)!

        }

        // drift --> also automatically includes disorder induced heating "update" (if necessary)
        pc->R = pc->R + pc->P * dt;  // R = R + P * dt
        pc->update(); // domain changed, so update...
        applyExtendedBC();
        if (this->use_disorder_induced_heating_m) applyReflectingBoundaryConditions(&(pc->R.getView()), &(pc->P.getView()), 90.11657); // to keep particles inside the "sphere"

        // kick 
        if (this->computeSelfField_m) {
            pc->P = pc->P + dt/2 * (pc->q / pc->m) * pc->E / e0;
            // if (this->output_debug && pc->E.extent(0)>=100) m << "E(0) = " << pc->E(0) << ", E(100) = " << pc->E(100) << endl;
        }
        // add another "kick" for the space confinement of the disorder induced heating process!
        if (this->use_disorder_induced_heating_m) {
            disorderInducedHeatingConfinementUpdate(dt);
        }

        IpplTimings::stopTimer(adv);

        //Kokkos::fence();
        //ippl::Comm->barrier();
    }

    void dump() override {
        size_type nlocal                           = this->pcontainer_m->getLocalNum();
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");

        IpplTimings::startTimer(dumpDataTimer);

        if (this->initial_distr == "sphere") {
            //bool save_velocity = false;

            //Inform velocityOut(NULL, "data/NB_velocity.csv", Inform::APPEND);
            //Inform positionOut(NULL, "data/NB_position.csv", Inform::APPEND);
            //Inform Eout(NULL, "data/NB_E.csv", Inform::APPEND);
            //Inform Xemitout(NULL, "data/NB_Xemit.csv", Inform::APPEND);
            //Inform dtout(NULL, "data/NB2_dt.csv", Inform::APPEND);
            //positionOut.precision(10);
            //velocityOut.precision(10);
            //Eout.precision(10);
            //dtout.precision(5);
            //Xemitout.precision(10);
            //velocityOut.setf(std::ios::scientific, std::ios::floatfield);
            //positionOut.setf(std::ios::scientific, std::ios::floatfield);
            //Eout.setf(std::ios::scientific, std::ios::floatfield);
            //Xemitout.setf(std::ios::scientific, std::ios::floatfield);
            //dtout.setf(std::ios::scientific, std::ios::floatfield);
            // output every particle velocity to file (V)
            view_type* P  = &(this->pcontainer_m->P.getView());
            view_type* R  = &(this->pcontainer_m->R.getView());
            //int time_step = this->it_m;

            /*
            if (save_velocity) {
                velocityOut << "time step " << time_step << ",V[x]" << ",V[y]" << ",V[z]" << endl;
                for (unsigned long int i = 0; i < this->pcontainer_m->getLocalNum(); i++) {
                    for (unsigned j = 0; j < Dim; j++) {
                        velocityOut << "," << (*P)(i)[j];
                    }
                    velocityOut << endl;
                }
            }

            if (save_velocity) {
                positionOut << "time step " << time_step << ",R[x]" << ",R[y]" << ",R[z]" << endl;
                // output every paritcle position to file (R)
                for (unsigned long int i = 0; i < this->pcontainer_m->getLocalNum(); i++) {
                    for (unsigned j = 0; j < Dim; j++) {
                        positionOut << "," << (*R)(i)[j];
                    }
                    positionOut << endl;
                }
            }*/

            // output energy x, y, z, total energy (Energy)
            double Energy = 0.0, Enery_x, Energy_y, Energy_z;

            Kokkos::parallel_reduce(
                "Particle Energy", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = dot((*P)(i), (*P)(i)).apply();
                    valL += myVal;
                },
                Kokkos::Sum<double>(Energy));

            Energy *= 0.5 * this->mass_per_particle_m;

            Kokkos::parallel_reduce(
                "Particle Energy x", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[0] * (*P)(i)[0];
                    valL += myVal;
                },
                Kokkos::Sum<double>(Enery_x));

            Enery_x *= 0.5 * this->mass_per_particle_m;

            Kokkos::parallel_reduce(
                "Particle Energy y", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[1] * (*P)(i)[1];
                    valL += myVal;
                },
                Kokkos::Sum<double>(Energy_y));

            Energy_y *= 0.5 * this->mass_per_particle_m;

            Kokkos::parallel_reduce(
                "Particle Energy z", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[2] * (*P)(i)[2];
                    valL += myVal;
                },
                Kokkos::Sum<double>(Energy_z));

            Energy_z *= 0.5 * this->mass_per_particle_m;

            //Eout << " time step " << time_step << " " << Energy << " " << Enery_x << " " << Energy_y
            //    << " " << Energy_z << endl;


            // Calculate mean R, P
            //size_t nlocal = this->pcontainer_m->getLocalNum();
            double xmean, vxmean;
            Kokkos::parallel_reduce(
                "Particle x Mean", (*R).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*R)(i)[0];
                    valL += myVal;
                },
                Kokkos::Sum<double>(xmean));
            xmean /= nlocal;

            Kokkos::parallel_reduce(
                "Particle x Mean", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[0];
                    valL += myVal;
                },
                Kokkos::Sum<double>(vxmean));
            vxmean /= nlocal;



            // calcualte X emittance
            double Xemit_x = 0.0, Xemit_R2, Xemit_V2, Xemit_RV;

            Kokkos::parallel_reduce(
                "Particle Xemit V", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = ((*P)(i)[0] - vxmean) * ((*P)(i)[0] - vxmean);
                    valL += myVal;
                },
                Kokkos::Sum<double>(Xemit_V2));
            Xemit_V2 /= nlocal;

            Kokkos::parallel_reduce(
                "Particle Xemit R", (*R).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = ((*R)(i)[0] - xmean) * ((*R)(i)[0] - xmean);
                    valL += myVal;
                },
                Kokkos::Sum<double>(Xemit_R2));
            Xemit_R2 /= nlocal;

            Kokkos::parallel_reduce(
                "Particle Xemit R", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = ((*R)(i)[0] - xmean) * ((*P)(i)[0] - vxmean);
                    valL += myVal;
                },
                Kokkos::Sum<double>(Xemit_RV));
            Xemit_RV /= nlocal;

            // Xemitx = sqrt(<x^2> * <V^2> - <x * V> ^2)
            // calcutate the total number

            Xemit_x = std::sqrt((Xemit_R2 * Xemit_V2 - Xemit_RV * Xemit_RV));

            //Xemitout << " time step " << time_step << " " << Xemit_x << endl;
            // output dt 
            //dtout << " time step " << time_step << " " << this->dt_m << endl;
            std::string velocitySavePath = "data/" + std::to_string(startTimestamp) + "_" + this->methods_m + "_" + this->initial_distr + ".csv";
            // std::string velocitySavePath = "data/" + std::to_string(this->nt_m) + "_" + this->methods_m + "_" + this->initial_distr + "_" + std::to_string(this->totalP_m) + ".csv"; // + std::to_string(this->realization_counter) + ".csv";
            //Inform velocityExpOut(NULL, "data/NB_velocity.csv", Inform::APPEND);
            if (this->realization_counter == 0 && this->it_m == 0) {
                Inform velocityExpOut(NULL, velocitySavePath.c_str(), Inform::OVERWRITE);
                velocityExpOut << "realization;time;Energy;Enery_x;Energy_y;Energy_z;Xemit_x;cell_vol" << endl;
            }
            Inform velocityExpOut(NULL, velocitySavePath.c_str(), Inform::APPEND);
            velocityExpOut.precision(10);
            
            velocityExpOut << this->realization_counter << ";" << this->time_m << ";" << Energy << ";" << Enery_x << ";" << Energy_y << ";" << Energy_z 
                           << ";" << Xemit_x << ";" << this->cellVolume*std::reduce(this->nr_m.begin(), this->nr_m.end(), 1., std::multiplies<double>()) << endl;


        } else if (this->initial_distr == "convergence") {
            // In this experiment, only save T_perp and T_parallel at every timestep
            view_type* P  = &(this->pcontainer_m->P.getView());

            double v_parallel_sq = 0.0, v_perp_sq = 0.0, totalE = 0.0;
            
            // Calculate mean parallel velocity
            Kokkos::parallel_reduce(
                "v_parallel_sq", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[2] * (*P)(i)[2];
                    valL += myVal;
                },
                Kokkos::Sum<double>(v_parallel_sq));
            v_parallel_sq /= (nlocal);

            // Calculate mean perpendicular velocity
            Kokkos::parallel_reduce(
                "v_perp_sq", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[1] * (*P)(i)[1] + (*P)(i)[0] * (*P)(i)[0];
                    valL += myVal;
                },
                Kokkos::Sum<double>(v_perp_sq));
            v_perp_sq /= (2 * nlocal);

            // Calculate total energy
            Kokkos::parallel_reduce(
                "Particle Energy", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = dot((*P)(i), (*P)(i)).apply();
                    valL += myVal;
                },
                Kokkos::Sum<double>(totalE));
            totalE *= 0.5 * this->mass_per_particle_m;

            // Finally dump the values into the file
            std::string velocitySavePath = "data/" + this->methods_m + "_" + this->initial_distr + "_parallel" + std::to_string(this->totalP_m) + ".csv"; // + std::to_string(this->realization_counter) + ".csv";
            //Inform velocityExpOut(NULL, "data/NB_velocity.csv", Inform::APPEND);
            if (this->realization_counter == 0 && this->it_m == 0) {
                Inform velocityExpOut(NULL, velocitySavePath.c_str(), Inform::OVERWRITE);
                velocityExpOut << "realization;time;v_parallel_sq;v_perp_sq;E" << endl;
            }
            Inform velocityExpOut(NULL, velocitySavePath.c_str(), Inform::APPEND);
            velocityExpOut.precision(10);
            
            velocityExpOut << this->realization_counter << ";" << this->time_m << ";" << v_parallel_sq << ";" << v_perp_sq << ";" << totalE << endl;
        } else if (this->initial_distr == "delta") {
            view_type* P = &(this->pcontainer_m->P.getView());
            view_type* E = &(this->pcontainer_m->E.getView()); 

            double v_x_sq = 0.0, v_y_sq = 0.0, v_z_sq = 0.0, E_mean = 0.0;
            
            // Calculate mean x velocity^2
            Kokkos::parallel_reduce(
                "v_x_sq", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[0] * (*P)(i)[0];
                    valL += myVal;
                },
                Kokkos::Sum<double>(v_x_sq));
            v_x_sq /= nlocal;

            // Calculate mean y velocity^2
            Kokkos::parallel_reduce(
                "v_y_sq", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[1] * (*P)(i)[1];
                    valL += myVal;
                },
                Kokkos::Sum<double>(v_y_sq));
            v_y_sq /= nlocal;

            // Calculate mean z velocity^2
            Kokkos::parallel_reduce(
                "v_z_sq", (*P).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = (*P)(i)[2] * (*P)(i)[2];
                    valL += myVal;
                },
                Kokkos::Sum<double>(v_z_sq));
            v_z_sq /= nlocal;

            // Calculate mean E field
            Kokkos::parallel_reduce(
                "E_mean", (*E).extent(0),
                KOKKOS_LAMBDA(const int i, double& valL) {
                    double myVal = std::sqrt((*E)(i)[0]*(*E)(i)[0] + (*E)(i)[1]*(*E)(i)[1] + (*E)(i)[2]*(*E)(i)[2]);
                    valL += myVal;
                },
                Kokkos::Sum<double>(E_mean));
            E_mean /= nlocal;

            // Finally dump the values into the file
            std::string velocitySavePath = "data/" + std::to_string(startTimestamp) + "_" + this->methods_m + "_" + this->initial_distr + ".csv";
            // std::string velocitySavePath = "data/" + std::to_string(this->nt_m) + "_" + this->methods_m + "_" + this->initial_distr + "_" + std::to_string(this->totalP_m) + ".csv"; // + std::to_string(this->realization_counter) + ".csv";
            //Inform velocityExpOut(NULL, "data/NB_velocity.csv", Inform::APPEND);
            if (this->realization_counter == 0 && this->it_m == 0) {
                Inform velocityExpOut(NULL, velocitySavePath.c_str(), Inform::OVERWRITE);
                velocityExpOut << "realization;time;v_x_sq;v_y_sq;v_z_sq;<E>;cell_vol;rho_norm" << endl;
            }
            Inform velocityExpOut(NULL, velocitySavePath.c_str(), Inform::APPEND);
            velocityExpOut.precision(10);

            velocityExpOut << this->realization_counter << ";" << this->time_m << ";" << v_x_sq << ";" << v_y_sq << ";" << v_z_sq << ";" << E_mean 
                           << ";" << this->cellVolume*std::reduce(this->nr_m.begin(), this->nr_m.end(), 1., std::multiplies<double>())
                           << ";" << this->rhoNorm_m << endl;

            ippl::Comm->barrier();
        }

        IpplTimings::stopTimer(dumpDataTimer);
    }

    void AllCollisions(bool useTakizukaAbe) {
        Inform msg("AllCollisions");
        static IpplTimings::TimerRef NB = IpplTimings::getTimer("AllCollision");
        IpplTimings::startTimer(NB);

        // Create uniform random engine...
        // std::random_device rd;
        std::mt19937 gen; // rd()
        gen.seed(this->seed+this->it_m); // +1 to not have the same as for the sampling

        std::uniform_real_distribution<> dis(0.0, 1.0);

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        view_type* P                            = &(pc->P.getView());
        mass_view_type* m                       = &(pc->m.getView()); // Mass type...
        mass_view_type* q                       = &(pc->q.getView()); // Charge type.

        std::unordered_map<size_t, std::vector<size_t>> mesh_decomp                               = genParticleMeshDecomposition();
        std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> mesh_compisition_pairs = genPairsFromMeshComposition(mesh_decomp, gen);

        unsigned int nan_c = 0;
        double temperature = getTemperature(P, this->totalP_m);
        //double mass_test = 0;
        for (const auto& [cellID, collision_pairs] : mesh_compisition_pairs) {
            // 1997 Nanbu collision model, all pairs collide in each cells
            // Nanbu2 below, add Nc to the loop to only collide Nc pairs in each cell

            const std::vector<size_t> current_cell = mesh_decomp[cellID];
            double total_mass = current_cell.size(); // 0.0;
            //mass_test += total_mass;
            /*Kokkos::parallel_reduce(
                "Total Cell Mass", current_cell.size(),
                KOKKOS_LAMBDA(const size_t i, double& valL) {
                    double myVal = (*m)(current_cell[i]); // (*m)(collision_pairs(i))
                    valL += myVal;
                }, Kokkos::Sum<double>(total_mass));
            */

            //auto& dx = (this->fcontainer_m->getMesh()).getMeshSpacing();
            double vol_per_cell = (this->fcontainer_m->getMesh()).getCellVolume();
            //double totalM       = 1.0 * mesh_decomp[cellID].size();  // TODO set mass=1 for now...
            double rho          = total_mass / vol_per_cell; // (dx(0) * dx(1) * dx(2));  // particle NUMBER density
            
            for (const auto& pair : collision_pairs) {
                size_type i = pair.first;
                size_type j = pair.second;

                // Calculate relative velocity
                Vector_t<double, Dim> dV =  (*P)(i) - (*P)(j);  // also named "q" in the Nanbu/Bird comparison paper

                // Nanbu collision model (1997)

                Vector_t<double, Dim> dU = useTakizukaAbe ? getTakizukaAbeVelocityUpdate(i, j, m, q, P, rho, gen, dis, nan_c, temperature)
                                                          : getNanbuVelocityUpdate(i, j, m, q, P, rho, gen, dis, nan_c, temperature);

                // Update particle velocities
                (*P)(i) = (*P)(i) - dU / 2 * (useTakizukaAbe ? -1 : 1);
                (*P)(j) = (*P)(j) + dU / 2 * (useTakizukaAbe ? -1 : 1);

                //Vector_t<double, Dim> dV2 =  (*P)(i) - (*P)(j);
                //std::cout << dV << "  ----   " << dV2 << std::endl;
            }
        }
        if (this->output_debug) {
            msg << this->realization_counter << "." << this->it_m << " No collision counter = " << nan_c << "/" << mesh_compisition_pairs.size() << endl; // << ", m_tot = " << mass_test << endl;
        }

        IpplTimings::stopTimer(NB); 
    }

    void BirdCollision() {
        // TODO calculate A

        /* do for each cells
                tc=0
                dtc = 2*tau/(rho*totalM) = dt/ Nc
                    while tc < (Nc+1) * dtc
                        pick a pair of particles i,j in one cell---eg (1,4), (1,7), (1,6) indices
           [1, 4,6,7 ...] do collision tc += dtc
         */
        // calculate time for bird collision
        static IpplTimings::TimerRef BirdTimer = IpplTimings::getTimer("BirdCollision");
        IpplTimings::startTimer(BirdTimer);

        // Create uniform random engine...
        //std::random_device rd;
        std::mt19937 gen;
        gen.seed(this->seed+this->it_m);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        view_type* P                            = &(pc->P.getView());
        mass_view_type* m                       = &(pc->m.getView()); // Mass type...
        mass_view_type* q                       = &(pc->q.getView()); // Charge type.
        double temperature                      = getTemperature(P, this->totalP_m);

        std::unordered_map<size_t, std::vector<size_t>> mesh_decomp = genParticleMeshDecomposition();
        //std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> mesh_compisition_pairs = genPairsFromMeshComposition(mesh_decomp, gen);

        // Generate densities inside mesh and update "Delta t" like for Nanbu2
        std::unordered_map<size_t, double> rho_values = updateTimeStepReturnRhoDistribution(mesh_decomp);

        // Iterate over each cells mesh_decomposition
        unsigned int nan_c = 0;
        for (const auto& [cellID, cell_particles_indices] : mesh_decomp) {
            // set collision time dt_c, time counter tc, calculate each cell rho, totalM, Nc

            /*
            Careful: We use t_c "per cell". Similarly, every cell has it's own Delta t_c.
            */
            double t_c  = 0.0; // Set initial time counter to 0
            double dt_c = 2 * this->dt_m / cell_particles_indices.size(); // Basically get "Nc collisions" per time tc
            double rho  = rho_values[cellID]; // get density of current cell

            //std::vector<size_t> indices(particleIndices);

            // Do the Bird routine
            do {
                // output i，j  two different indices from praticleIndices
                /*std::vector<size_t> sample_pair;

                std::sample(indices.begin(), indices.end(), std::back_inserter(sample_pair), 2,
                            gen);
                size_t i = sample_pair[0];
                size_t j = sample_pair[1];

                if (i == j) {
                    continue;
                }*/
                auto [i, j] = randElementPair(cell_particles_indices, gen);

                // Do the usual collision from Nanbu
                Vector_t<double, Dim> dU = getNanbuVelocityUpdate(i, j, m, q, P, rho, gen, dis, nan_c, temperature);


                // Update particle velocities
                (*P)(i) = (*P)(i) - dU / 2;
                (*P)(j) = (*P)(j) + dU / 2;

                /*
                Update the time counter.
                Iterate everytime "Nc/2" times. This means, iterate until the
                current dt per cell is greater than the "global dt_m". This (just 
                using a counter per cell and per step) is simpler than creating 
                a "global time" counter as suggested by the paper.
                This makes it also easier to account for different timestep sizes
                after every step.
                */ 
                t_c += dt_c;
            } while (t_c < this->dt_m);
        }
        std::cout << "No collision counter = " << nan_c << std::endl;

        IpplTimings::stopTimer(BirdTimer);
    }

    void Nanbu2Collision() {
        // TODO: A not solve

        // calculate Nc collision pairs in one step  Nc: = round (\rho_cell * N_cell / (2 *tau))

        /*
        How to get the "new" Delta t.
        We want to find the global maximum of the density field.
        1. Find local maximum (calculate all densities, save in map with cellID).
        2. Use...
            localExNorm = max(rho...);
            double ExAmp = 0.0;
            ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());
        ....to get the global maximum!
        3. Calculate the timestepsize for this step. Maybe rewrite advancce() method to split
        step/Nanbu stuff and space advancement! Want Dt = tau/rho * fraction, where fraction=1 gives
        Bobylev and <=1 gives Babovsky.
        4. Don't forget to save Delta t using dump().
        */

        static IpplTimings::TimerRef NB2 = IpplTimings::getTimer("Nanbu2Collision");
        IpplTimings::startTimer(NB2);

        // Create uniform random engine...
        // std::random_device rd;
        std::mt19937 gen;
        gen.seed(this->seed+this->it_m);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        view_type* P                            = &(pc->P.getView());
        mass_view_type* m                       = &(pc->m.getView()); // Mass type...
        mass_view_type* q                       = &(pc->q.getView()); // Charge type.
        double temperature                      = getTemperature(P, this->totalP_m);

        std::unordered_map<size_t, std::vector<size_t>> mesh_decomp                               = genParticleMeshDecomposition();
        std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> mesh_compisition_pairs = genPairsFromMeshComposition(mesh_decomp, gen);

        /*
        For this algorithm we need to update "Delta t" every step. In order to do this, we need the maximum possible
        rho over all processes/nodes to make it consistens. This means, in every timestep, we will first calculate 
        all the densities per cell, then calculate the maximum and finally use the communicator to get the maximum
        over all processes. 
        */
        std::unordered_map<size_t, double> rho_values = updateTimeStepReturnRhoDistribution(mesh_decomp);

        unsigned int nan_c = 0;
        for (const auto& [cellID, collision_pairs] : mesh_compisition_pairs) {
            /* 
            If N =2000000, grids 32 32 32, rmax =m , actually max rho = mass* N_cell/ V_cell <<2
            time interval  <= tau / rho <= 1/ 2 <= 0.5.
            also time interval should satisfy not smaller engouh to make sure all cells at least
            have one pair to collision now our set is 0.1
            */

            double rho    = rho_values[cellID]; // particle density in this cell
            double N_cell = mesh_decomp[cellID].size(); // curren cell size
            size_type Nc  = std::round(rho * N_cell * this->dt_m / (2 * this->tau_m));

            // judage Nc is larger than the number of collision pairs or not
            if (Nc > collision_pairs.size()) {
                Nc = collision_pairs.size();
                std::cout << "Nc is larger than the number of collision pairs!" << std::endl;
            }

            unsigned int iteration_counter = 0; // want only Nc iterations!
            for (const auto& pair : collision_pairs) {
                // Only continue if we had less than Nc collisions
                if (++iteration_counter >= Nc) { break; }

                size_type i = pair.first;
                size_type j = pair.second;

                Vector_t<double, Dim> dU = getNanbuVelocityUpdate(i, j, m, q, P, rho, gen, dis, nan_c, temperature);

                // Update particle velocities
                (*P)(i) = (*P)(i) - dU / 2;
                (*P)(j) = (*P)(j) + dU / 2;
            }
        }
        std::cout << "No collision counter = " << nan_c << std::endl;

        IpplTimings::stopTimer(NB2);
    }

    std::unordered_map<size_t, double> updateTimeStepReturnRhoDistribution(const std::unordered_map<size_t, std::vector<size_t>> mesh_decomp) {
        double vol_per_cell = (this->fcontainer_m->getMesh()).getCellVolume();
        //std::cout << vol_per_cell << std::endl;
        
        std::unordered_map<size_t, double> rho_values; 
        for (const auto& [cellID, current_cell] : mesh_decomp) {
            // Calculate total mass using Kokkos (maybe faster for many many particles?) NOPE
            double total_mass = current_cell.size(); // use size, since only number density is relevant!
            /*Kokkos::parallel_reduce(
                "Total Cell Mass", current_cell.size(),
                KOKKOS_LAMBDA(const size_t i, double& valL) {
                    double myVal = (*m)(current_cell[i]); // (*m)(collision_pairs(i))
                    valL += myVal;
                }, Kokkos::Sum<double>(total_mass));
            */

            // Calculate density and save it with corresponding cellID, update 14.05.2024: ...and normalize it using "initial density"
            rho_values[cellID] = (total_mass / vol_per_cell); // / this->normalization_reference_density;
        }
        double local_max_rho = (rho_values.size() == 0) ? 0.0 : getMapMax(rho_values).second;

        // Now ask ippl::Comm to give me the greatest of the local_max_rho available
        double global_max_rho = 0.0;
        ippl::Comm->reduce(local_max_rho, global_max_rho, 1, std::greater<double>());

        // Finally use global_max_rho to set a new Delta t for this run!
        //if (useBird) {
            // Update for the Bird algorithm...
        //    this->dt_m = 2 * this->tau_m / (global_max_rho * mesh_decomp.size());
        //} else {
            // Use Nanbu2
        //std::cout << global_max_rho << " - " << local_max_rho << std::endl;
        this->setTimestepsize(this->timestep_fraction_m * this->tau_m / global_max_rho);
        //}

        return rho_values;
    }

    Vector_t<double, Dim> getNanbuVelocityUpdate(const size_type& i, const size_type& j,
                                                 const mass_view_type* m, const mass_view_type* q, const view_type* P, const double& rho,
                                                 std::mt19937& gen, std::uniform_real_distribution<>& dis, unsigned int& nan_c,
                                                 double& temperature) { 

        // Calculate relative velocity
        Vector_t<double, Dim> dV = (*P)(i) - (*P)(j);  // also named "q" in the Nanbu/Bird comparison paper
        
        // Calculate velocity components
        double V_n   = std::sqrt(dV[0] * dV[0] + dV[1] * dV[1] + dV[2] * dV[2]);
        double V_tau = std::sqrt(dV[1] * dV[1] + dV[2] * dV[2]);

        double invTau1 = getInvTau1(i, j, m, q, rho, V_n, this->initial_distr, temperature);
        double s       = this->tau_m * invTau1 / rho * this->dt_m; // V_n/(rho * rho); // this one or not, i don't know, just use annotation
        //s = 1.1;
        double A       = getA(s);
        //std::cout << s << " - " << A << std::endl;
        //std::cout << s << " - " << 10*this->tau_m/(pi*std::pow(V_n, 3)) << " - " << rho << " - " << V_n << std::endl;

        // Sample U uniformly from [0, 1]
        double U1 = dis(gen);
        double U2 = dis(gen);

        // Calculate collision angles
        double psi  = 2 * pi * U2;
        double cosx = (std::isnan(s) || (s < s_thresholds.first) || (s > s_thresholds.second)) ? 
                        1.0 : (std::log(std::exp(-A) + 2 * U1 * std::sinh(A)) / A);
        double sinx = std::sqrt(1 - cosx * cosx);
        //if (std::abs(sinx) < 1e-12) {
        //    return Vector_t<double, Dim> {0.0, 0.0, 0.0};
        //}

        // Calculate velocity change
        Vector_t<double, Dim> dU =
            (1 - cosx) * dV
            + (sinx / V_tau)
                    * Vector_t<double, Dim>{
                        V_tau * V_tau * std::cos(psi),
                        -1 * (dV[0] * dV[1] * std::cos(psi) + dV[2] * V_n * std::sin(psi)),
                        -1 * (dV[2] * dV[0] * std::cos(psi) - dV[1] * V_n * std::sin(psi))};
        //std::cout << dU << std::endl;
        
        // Outputs parameter if we get a NaN velocity update!
        if (std::isnan(dU[0]) || std::isnan(dU[1]) || std::isnan(dU[2]) || std::isnan(s)) { // (s < s_thresholds.first) || (s > s_thresholds.second)
            //Inform msg(TestName);
            dU = 0.0; // Set dU=0 for the case dU=NaN --> no update (since it comes from V_tau=0)!
            nan_c++;
            //std::cout << "Encountered small/nan velocity update: " << dU << " " << s << " " << A << " " << U1 << " " << U2 << " " << invTau1 << " " << V_n << " " << V_tau << " " << rho << std::endl;
        }

        return dU;
    }

    Vector_t<double, Dim> getTakizukaAbeVelocityUpdate(const size_type& i, const size_type& j,
                                                       const mass_view_type* m, const mass_view_type* q, const view_type* P, const double& rho,
                                                       std::mt19937& gen, std::uniform_real_distribution<>& dis, unsigned int& nan_c,
                                                       double& temperature) {
        Vector_t<double, Dim> dV = (*P)(i) - (*P)(j); // Relative velocity
        double u_n               = normIPPLVector(dV, false);
        double u_p               = std::sqrt(std::pow(dV[0], 2) + std::pow(dV[1], 2)); // Careful: Nanbu and this have different def's of "perpendicular"
        
        auto [sinPhi, cosPhi, sinTh, cosTh] = genPhiThetaTakizukaAbe(this->dt_m, u_n, i, j, m, q, rho, gen, dis, temperature, this->initial_distr); 

        double ux = dV[0], uy = dV[1], uz = dV[2]; 

        // Calculate velocity update according to formula
        Vector_t<double, Dim> dU = {ux/u_p*uz*sinTh*cosPhi - uy/u_p*u_n*sinTh*sinPhi - ux*(1-cosTh),
                                    uy/u_p*uz*sinTh*cosPhi + ux/u_p*u_n*sinTh*sinPhi - uy*(1-cosTh),
                                    -u_p*sinTh*cosPhi - uz*(1-cosTh)};
        
        // Catch "no collision" or zero velocity spread...
        if (std::isnan(dU[0]) || std::isnan(dU[1]) || std::isnan(dU[2])) {
            dU = 0.0; // Set dU=0 for the case dU=NaN --> no update (since it comes from V_...=0)!
            nan_c++;
        }
        // std::cout << sinPhi << " - " << cosPhi << " - " << sinTh << " - " << cosTh << std::endl;

        return dU;
    }

    std::unordered_map<size_t, std::vector<size_t>> genParticleMeshDecomposition() {
        using vector_type = typename Mesh_t<Dim>::vector_type;

        size_type nlocal = this->pcontainer_m->getLocalNum();
        view_type* R     = &(this->pcontainer_m->R.getView());

        // Get mesh for
        const Mesh_t<Dim>& mesh = this->fcontainer_m->getMesh();

        // Get spacings
        const vector_type& dx       = mesh.getMeshSpacing();
        const vector_type& origin   = mesh.getOrigin();
        const vector_type& gridsize = mesh.getGridsize();  // Number of cells per dimension
        const vector_type invdx     = 1.0 / dx;

        std::unordered_map<size_t, size_t> mesh_sizes;  // contains (cellID, #Particles) --> want to avoid .push_back(...)
        for (size_t idx = 0; idx < nlocal; ++idx) {
            Vector_t<size_t, Dim> l = ((*R)(idx)-origin) * invdx;  // add +0.5 to allocate to grid points, leave it out for grid CELLS
            size_t cellID           = getUniqueParticleID(l, gridsize);
            mesh_sizes[cellID]      = (mesh_sizes.count(cellID) == 0) ? 1 : mesh_sizes[cellID] + 1;
            // mesh_decomposition[cellID].push_back(idx);
        }

        // Now create a map and allocate the storage for the particles
        std::unordered_map<size_t, std::vector<size_t>> mesh_decomposition;
        for (auto& [cellID, num_particles] : mesh_sizes) {
            mesh_decomposition[cellID].reserve(num_particles);
        }

        // Now actually fill the mesh_decomposition with particle indices doing the same as
        // before...
        for (size_t idx = 0; idx < nlocal; ++idx) {
            Vector_t<size_t, Dim> l =
                ((*R)(idx)-origin)
                * invdx;  // add +0.5 to allocate to grid points, leave it out for grid CELLS
            size_t cellID = getUniqueParticleID(l, gridsize);
            mesh_decomposition[cellID].push_back(idx);
        }

        return mesh_decomposition;
    }

    void disorderInducedHeatingConfinementUpdate(const double& dt) {
        static IpplTimings::TimerRef bunchConfinement = IpplTimings::getTimer("DisorderInducedBunchConfinement");
        IpplTimings::startTimer(bunchConfinement);
        // Calculate the forces for the disorder induced heating confinement
        // and update the particle velocities accordingly.

        // access instance like this (same as in initialize)? getting a reference in the function head gives an illegal access error...
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        // Use a view of the particle velocity, since otherwise Kokkos might do some inefficient stuff (citation: Mohsen...)?
        view_type* viewP      = &(pc->P.getView());
        view_type* viewR      = &(pc->R.getView());
        mass_view_type* viewm = &(pc->m.getView());
        mass_view_type* viewq = &(pc->q.getView());
        view_type* viewE      = &(pc->E.getView());
        
        // pc->P is the particle velocity attribute and forces is a Kokkos::View
        view_type forces = getDisorderInducedHeatingForcesView(viewR, viewq, viewE, this->confinementForceAdjust); // this->rmin_m, this->rmax_m, 
        
        // Perform the "drift" on each element for linear inwards facing confinement force.
        // Needs to be done elementwise, since I can't multiply perform view_type and particleAttribute calculations elementwise... 
        Kokkos::parallel_for("UpdatePConfinementForce", viewP->extent(0), KOKKOS_LAMBDA(const int i) {
            (*viewP)(i) = (*viewP)(i) + (dt/2) * forces(i) / (*viewm)(i); // * ((*viewq)(i) / (*viewm)(i));
        });
        Kokkos::fence();

        IpplTimings::stopTimer(bunchConfinement);
    }

    void adjustFieldMeshDimensions() { // const unsigned int& it, const unsigned int& adjustEvery
        Inform m("AdjustFieldMeshDimensions");
        // if (it % adjustEvery != 0) { return; }

        // ...otherwise try to adjust the mesh dimensions
        static IpplTimings::TimerRef fieldAdjust = IpplTimings::getTimer("MeshDimensionAdjustment");
        IpplTimings::startTimer(fieldAdjust);

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        auto *mesh = &this->fcontainer_m->getMesh();
        auto *FL = &this->fcontainer_m->getFL();

        // Calculate the maximum and minimum of all particle coordinates using Kokkos
        view_type* R               = &(pc->R.getView());
        // double minR[Dim], maxR[Dim];
        MinMaxReducer<Dim> minMax;
        findMinMax(R, minMax); 
        Vector_t<double, Dim> maxR = minMax.max_val, minR = minMax.min_val;
        if (this->output_debug) m << "MaxR: " << maxR << " MinR: " << minR << endl;
        /*Kokkos::View<MinMaxReducer<Dim>> minMax("minMax");

        Kokkos::parallel_reduce("FindMinMax", 100, KOKKOS_LAMBDA(const int i, MinMaxReducer<Dim>& reducer) {
            for (int d = 0; d < Dim; ++d) {
                reducer.min_val[d] = std::min(reducer.min_val[d], data(i, d));
                reducer.max_val[d] = std::max(reducer.max_val[d], data(i, d));
            }
        }, minMax);*/


        /*maxR[0] = 1.0;
        Kokkos::parallel_for("MaxMinR", pc->getLocalNum(), KOKKOS_LAMBDA(const int i) {
            for (size_t d = 0; d < Dim; ++d) {
                Kokkos::atomic_max(&maxR(d), (*R)(i)[d]);
                Kokkos::atomic_min(&minR(d), (*R)(i)[d]);
                //if ((*R)(i)[d] > maxR[d]) { maxR[d] = (*R)(i)[d]; }
                //if ((*R)(i)[d] < minR[d]) { minR[d] = (*R)(i)[d]; }
            }
        });*/
        
        /*Kokkos::Min<double[Dim]> min_reducer;
        Kokkos::Max<double[Dim]> max_reducer;
        //Kokkos::MinMax<...> ;
        Kokkos::parallel_reduce("MaxMinR", pc->getLocalNum(), 
            KOKKOS_LAMBDA(size_t i, double min_val[Dim], double max_val[Dim]) {
                for (size_t d = 0; d < Dim; ++d) {
                    min_val[d] = std::min(min_val[d], (*R)(i)[d]);
                    max_val[d] = std::max(max_val[d], (*R)(i)[d]);
                }
            }, min_reducer, max_reducer
        );
        for (size_t d = 0; d < Dim; ++d) {
            minR(d) = min_reducer.reference()[d];
            maxR(d) = max_reducer.reference()[d];
        }*/

        // std::cout << "MaxR: " << maxR << " MinR: " << minR << std::endl;

        // Now figure out what the componentwise global min/max values are using the ippl::Comm
        Vector_t<double, Dim> globalMaxR, globalMinR;
        for (size_t i = 0; i < Dim; i++) {
            ippl::Comm->reduce(&maxR[i], &globalMaxR[i], 1, std::greater<double>());
            ippl::Comm->reduce(&minR[i], &globalMinR[i], 1, std::less<double>());
        }
        if (this->output_debug) m << "GlobalMaxR: " << globalMaxR << " GlobalMinR: " << globalMinR << endl;
        
        //Vector_t rescale_vector = (this->rmax_m-this->rmin_m) / (globalMaxR-globalMinR);
        
        // Calculate new mesh spacing 
        Vector_t<double, Dim> hr = (globalMaxR-globalMinR) / mesh->getGridsize(); // divide by number of cells in each dimension
        //this->E_rescale = 1.0;
        //for (size_t i = 1; i < Dim; ++i) this->E_rescale *= this->hr_m[i] / hr[i];

        this->cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());

        // set the origin and mesh spacing of the mesh via
        mesh->setMeshSpacing(hr);
        mesh->setOrigin(globalMinR); // becomes the new origin, since the domain starts from (0, 0, 0)
        if (this->output_debug) m << "MeshSpacing: " << hr << " Origin: " << globalMinR << endl;

        // AFTER layout update, since otherwise the BC will be applied to the wrong computation domain
        this->rmin_m = globalMinR;
        this->origin_m = globalMinR;
        this->rmax_m = globalMaxR;
        this->hr_m = hr;

        extLayoutUpdate(FL, mesh);
        pc->update();
        if (this->output_debug) m << "Extended layout update completed." << endl;
        
        //this->rmin_m = globalMinR;
        //this->rmax_m = globalMaxR;
        //this->origin_m = globalMinR;
        //this->hr_m   = this->rmax_m / this->nr_m;  // mesh spacing hr = rmax_m/[128,128,128]
        //this->setFieldContainer(std::make_shared<FieldContainer_t>(
        //    this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
        //    this->isAllPeriodic_m));
        //this->fcontainer_m->setHr(this->rmax_m / this->nr_m);
        //this->fcontainer_m->setRMin(globalMinR);
        //this->fcontainer_m->setRMax(globalMaxR);


        // then, update the field layout
        //pc->getLayout().updateLayout(*FL, *mesh);
        //this->fcontainer_m->getRho().initialize(*mesh, *FL);
        //pc->update(); 
        //bool isFirstRepartition_m      = false;
        //this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition_m); // better than updating manually...
        
        // Need to set this too, otherwise: the scaling of E will be off -.-
        //bool first_rep = false; 
        //this->loadbalancer_m->updateLayout(FL, mesh, first_rep);
        //this->fcontainer_m->initializeFields(this->solver_m);
        //Kokkos::fence();
        //ippl::Comm->barrier();
        
        IpplTimings::stopTimer(fieldAdjust);
    }

    /*void resetFieldDimensions() {
        # * Has to be called after using the solver! Otherwise, the particles will just move infinitely
        # * (like open boundary conditions...).
        # * Otherwise, .update() doesn't seem to work properly and does not apply boundary conditions...
        # * I thought, the domain and the mesh are different, but they seem to be interlinked or smth
        
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        auto *mesh = &this->fcontainer_m->getMesh();
        auto *FL = &this->fcontainer_m->getFL();

        mesh->setMeshSpacing(this->hr_m);
        mesh->setOrigin(this->origin_m); // reset domain starts usually from (0, 0, 0)
        pc->getLayout().updateLayout(*FL, *mesh);
        pc->update(); 
    }*/

    void applyExtendedBC() {
        /*
        Problem: The usual .applyBC() function only works for particles that are up to
        "one domain size away" from the domain. If particle velocities are high enough,
        they migh overshoot that limit and the algorithm breaks.

        Therefore: Use modulo instead of one simple subtraction!
        */
        static IpplTimings::TimerRef applyBC = IpplTimings::getTimer("ApplyExtendedBC");
        IpplTimings::startTimer(applyBC);
        
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        view_type* R                            = &(pc->R.getView());
        const Vector_t<double, Dim>& origin = this->origin_m;
        const Vector_t<double, Dim>& rmax   = this->rmax_m;
        
        //const Vector_t<double, Dim> origin = {0.0, 0.0, 0.0};
        //const Vector_t<double, Dim> rmax   = {1.0, 1.0, 1.0};

        // pc->R = this->origin_m + ippl::Mod((pc->R - this->origin_m), pc->R);

        Kokkos::parallel_for("ApplyPeriodicBC", R->extent(0), KOKKOS_LAMBDA(const int i) {
            //(*R)(i) = origin + fmod((*R)(i) - origin, rmax);
            for (size_t d = 0; d < Dim; ++d) {
                // Needs this weired fmod(fmod(a, b) + b, b), since (-3.141)%1.0 == -0.141, but we want 0.859 ("Python modulo")
                (*R)(i)[d] = origin[d] + fmod(fmod((*R)(i)[d] - origin[d], rmax[d]) + rmax[d], rmax[d]);
            }
        });
        Kokkos::fence();

        // Check for errors
        /*cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        }*/
        
        IpplTimings::stopTimer(applyBC);
    }

    /*void rescaleE() {
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        view_type* E                            = &(pc->E.getView());

        Kokkos::parallel_for("ERescale", E->extent(0), KOKKOS_LAMBDA(const int i) {
            //(*R)(i) = origin + fmod((*R)(i) - origin, rmax);
            (*E)(i) = (*E)(i) * this->E_rescale;
        });
    }*/

    void extLayoutUpdate(ippl::FieldLayout<Dim>* fl, ippl::UniformCartesian<T, Dim>* mesh) {
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        Field_t<Dim>* rho_m   = &(fc->getRho());
        VField_t<T, Dim>* E_m = &(fc->getE());

        rho_m->updateLayout(*fl);
        E_m->updateLayout(*fl);

        //PLayout_t<T, Dim>* layout = &pc->getLayout();
        pc->getLayout().updateLayout(*fl, *mesh);
        //(*layout).updateLayout(*fl, *mesh);

        std::get<FFTSolver_t<T, Dim>>(this->fsolver_m->getSolver()).setRhs(*rho_m);
    }

    void resetBoundaries() {
        this->origin_m = 0.0;
        this->rmin_m   = this->origin_m;
        if (this->initial_distr == "sphere") {
            this->rmax_m = 506.84;
        } else {
            this->rmax_m = 1.0;
        }
        this->hr_m     = this->rmax_m / this->nr_m;

        //this->cellVolume = std::reduce(this->hr_m.begin(), this->hr_m.end(), 1., std::multiplies<double>());
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        auto *FL = &this->fcontainer_m->getFL();
        auto *mesh = &this->fcontainer_m->getMesh();
        // set the origin and mesh spacing of the mesh via
        mesh->setMeshSpacing(this->hr_m);
        mesh->setOrigin(this->rmin_m); // becomes the new origin, since the domain starts from (0, 0, 0)
        

        extLayoutUpdate(FL, mesh);
        //std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        // view_type* E                            = &(pc->E.getView());
        // pc->E = pc->E / this->cellVolume;
    }


};
#endif
