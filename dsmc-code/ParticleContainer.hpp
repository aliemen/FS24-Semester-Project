#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class

// parameter input 
// particle attribute: q, P, E 

// we dont need to add R (position), ID , since particlebase have already defined it. Particle

        //! view of particle positions
        // particle_position_type R;

        //! view of particle IDs
        //  particle_index_type ID;



template <typename T, unsigned Dim = 3> // T particle attribute like( rho double), Dim 
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>{
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>; 

    public:
        ippl::ParticleAttrib<double> q;                 // scalar: charge
        ippl::ParticleAttrib<double> m;   // add particle mass
        typename Base::particle_position_type P;  // vector: particle velocity
        typename Base::particle_position_type E;  // vector: electric field at particle position
    private:
        PLayout_t<T, Dim> pl_m; // About how particles are laid out in space (such as meshing, spatial relationships between particles, etc.)
    public:
        ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL) 
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
        }

        ~ParticleContainer(){} 

        std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; } //By returning a shared pointer to pl_m, other objects or functions can access and use the particle's layout information. 
        void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; } // allow external code updates the pl_m member variable.
    


	void registerAttributes() {
		// register the particle attributes
		this->addAttribute(q); // add attributes charge
		this->addAttribute(m); // add mass attribute
		this->addAttribute(P); // add attributes velocity
		this->addAttribute(E); // add attributes fields at particle position 

	}
	void setupBCs() { setBCAllPeriodic(); }

    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); } // 
};

#endif
