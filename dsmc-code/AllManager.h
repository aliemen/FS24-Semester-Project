#ifndef IPPL_ALLMANAGER_H
#define IPPL_ALLMANAGER_H

#include <memory>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h" 


using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class AllManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t = LoadBalancer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
protected:
    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    double lbt_m;
    std::string solver_m;
    bool output_debug;
public:
    AllManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_,
               std::string& solver_, bool output_debug_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                           LoadBalancer<T, Dim>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , output_debug(output_debug_) {}
    ~AllManager(){}

protected:
// time related variables
    double time_m;
    double dt_m;
    int it_m;
// grid related variables
    Vector_t<double, Dim> kw_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> origin_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
// self-field related variables
    double rhoNorm_m; // norm of the charge density
    double Q_m; // total charge

public:
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    double getLoadBalanceThreshold() const { return lbt_m; }

    void setLoadBalanceThreshold(double lbt_) { lbt_m = lbt_; }

    const std::string& getSolver() const { return solver_m; }

    void setSolver(const std::string& solver_) { solver_m = solver_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }
    
    void setTimestepsize(double timestepsize_) { dt_m = timestepsize_; }

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
        if (this->output_debug) {
            Inform m("Pre-step");
            m << "Done" << endl;
        }
    }

    void post_step() override {
        // Update time
        this->time_m += this->dt_m; 
        this->it_m++;  //
        // wrtie solution to output file
        this->dump();
        if (this->output_debug) {
            Inform m("Post-step:");
            m << "Finished time step: " << this->it_m << " time: " << this->time_m << endl;
        }
    }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        gather(this->pcontainer_m->E, this->fcontainer_m->getE(), this->pcontainer_m->R);
    }

    void par2grid() override { scatterCIC(); }

    void scatterCIC() {
        Inform m("scatter ");
        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double> *q = &this->pcontainer_m->q;
        typename Base::particle_position_type *R = &this->pcontainer_m->R;
        Field_t<Dim> *rho               = &this->fcontainer_m->getRho();
        double Q                        = Q_m;
        Vector_t<double, Dim> rmin	= rmin_m;
        Vector_t<double, Dim> rmax	= rmax_m;
        Vector_t<double, Dim> hr        = hr_m;

        scatter(*q, *rho, *R);
        // m << "Scatter done" << endl;
        double relError = std::fabs((Q-(*rho).sum())/Q);
        
        if (this->output_debug) m << relError << endl;

        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || relError > 1e-10) {
                m << "Time step: " << it_m << endl;
                m << "Total particles in the sim. " << totalP_m << " "
                  << "after update: " << TotalParticles << endl;
                m << "Rel. error in charge conservation: " << relError << endl;
                ippl::Comm->abort();
            }
	    }

        double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)            = (*rho) / cellVolume;

        rhoNorm_m = norm(*rho);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (Q / size);
        }
   }
};
#endif
