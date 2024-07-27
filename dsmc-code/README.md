# Current `run.sh` content and environment variables
```
#!/bin/bash
#SBATCH -n 2 # Number of cores
#SBATCH --time=90:00 # Runtime in minutes
#SBATCH --mem-per-cpu=3000              # memory per CPU core
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=nanbu_coulomb_sim   # Descriptive job name
#SBATCH --output=results/nanbu_sim_%j.out  # Write stdout to results/ dir with job ID
#SBATCH --error=results/nanbu_sim_%j.err   # Write stderr to results/ dir with job ID

cd ../../.. # TODO CHANGE THIS
module load gcc/11.4.0 cmake/3.26.3 cuda/12.1.1 openmpi/4.1.4

# ./ippl-build-scripts/999-build-everything -t serial -k -f -i -u # TODO CHANGE THIS
cd ippl/build_serial/05.03.dsmc/dsmc-code-simple # TODO CHANGE THIS
# cd ippl/build_openmp/05.03.dsmc/dsmc-code-simple # (this for multi threading)

# clean up data directory and timing file
rm -rf data 
rm -rf timingNB.dat
rm -rf timingDSMC_Based.dat
mkdir -p data
chmod 777 data

# add "mpirun -np 4" before the command to run in parallel (with 4 cores)

# mpirun -np 2 ./Nanbu 64 64 64 156055 1000 0.01 FFT false true Nanbu 1 2.15623e-13 63699.28 --info 10

# mpirun -np 2 ./Nanbu 64 64 64 156055 1000 0.01 FFT true true Nanbu 1 2.15623e-13 63699.28 --info 10

./Nanbu 64 64 64 156055 1000 0.01 FFT true true Nanbu 1 2.15623e-13 63699.28 --info 10 # 2.4e7 --> initial 150MeV bunch (doesn't work lol)
```
The arguments are:
| Parameter Value | Description                                            |
|-----------------|--------------------------------------------------------|
| 64 64 64        | Grid                                                   |
| 156055          | Number of particles                                    |
| 1000            | Timesteps                                              |
| 0.01            | Loadbalancer threshold                                 |
| true true       | Field solver? Binary Collisions?                       |
| Nanbu           | Method (Nanbu, Nanbu2, Bird)                           |
| 1               | $\tau$                                                 |
| 2.15623e-13     | Overwrite timestepsize (important for task 4)          |
| 63699.28        | Overwrite initial particle velocity magnitude (task 4) |
| --info 10       | Output                                                 |




# DSMC Collision Overview 

we use indexes, meshes, and fields in the IPPL framework to study the Nanbu Coulomb collision model

- Manager hpp document  "DManager.h", "DSMCManager.h", "NanbuManager.h"
- datetpyes: " datatypes.h" 
- container: "ParticleContainer.h", "FieldContainer.h" 
- main cpp file: "Nanbu.cpp"

first we use ` pre_run() ` to initialize the particles, then we use ` run() ` to iterate over the time steps, for each time step we use ` pre_step(), advance(), post_step()` to update the particles and fields （Here we use a Boolean value “computeSelfField_mcomputeSelfField_m"  to determine whether it has self_field solver , if we set it false, it will not use fieldsolver and grid to particle）. ` pre_step()` function override in DSMCManager.h only print "pre_step()" ` advance()` function override in NanbuManager.h to implement the Nanbu algorithm (coulumb collision), ` post_step()` function override in DSMCManager.h to update the time and dump the data.

- ` pre_run() ` we use to initial particle velocity and position and particle ID  
   - if computeSelfField_mcomputeSelfField_m = false we dont use getE() and getPhi(), par2grid() and grid2par() functions
   - we call ` initialzeParticles() ` to get the initial particle velocity and position and ID 
     -  particle ojects is created, it is initially empty. we need to use  create() (in particle bases) to create the particles. (eg, this ->pcontainer_m->create(100);
     - initialze ... 
        - R uniform distribution
        - V Gaussian distribution 
     - update()

# DSMC Collision Implementation

1. "DManager.h" in src directory
 
 - inherit from BaseManager class, which have pre_run(), ` pre_step(), advance(), post_step()` virtual function 

 - input parameters: 

 - private or protected parameters : ` fcontainer_m, pcontainer_m, fsolver_m;`  
 - functions:

    ```cpp
        // virtual function: 
        grid2par(), par2grid()
        // set and get function: 
        getParticleContainer(), setParticleContainer(), getFieldContainer(), setFieldContainer(), getFieldSolver(), setFieldSolver() 
    ```
- virtual function override in DSMCManager.h 
    
    ```cpp
        void pre_run() override; // initial particles 
        void pre_step() override; // // print something " pre_step()
        void advance() override; // Nanbu algorithm 
        void post_step() override; // update time + dump()
    ```
2. ParticleContainer:

  - input: `ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)`
  - private member: `ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>> pl_m`; // pl_m(FL, mesh)
  - attributes:  velocity (P), charge (q), electric field (E) 
  - addAttribute and periodic boundary condition: `this->addAttribute(q)` , `this->SetPeriodicBC(ippl:BC: PERIODIC)`
  - flag to compute the self-consistent field: `bool computeSelfField_m;`
   

3. FieldContainer: 

- input:   

```cpp
// hr meshing space
// rmin and rmax: the minimum and maximum coordinates of the domain
// decomp: the decomposition of the domain
// domain: the domain of the mesh 

FieldContainer(Vector_t<T, Dim>& hr, Vector_t<T, Dim>& rmin, Vector_t<T, Dim>& rmax, std::array<bool, Dim> decomp,
         ippl::NDIndex<Dim> domain, Vector_t<T, Dim> origin, bool isAllPeriodic)
```
- private memeber: hr_m, rmin_m, rmax_m, decomp_m, mesh_m, fl_m  (E_m, rho_m, Phi_m if not have self-consistent field) 
// mesh_m( domain, hr, origin)
// fl_m( MPI_COMM_WORLD, mesh_m, domain, decomp, isAllPeriodic)
- set and get functions: ...

4. DSMCManager.h

- input parameters:  `size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, std::string& solver_, bool  computeSelfField_  `

- override  `pre_step(), post_step(), grid2par(), par2grid() functions `

- protected memeber

```cpp

    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    std::string solver_m;
    std::string computeSelfField_m;

```

- protected parameters: 

```cpp
    // time related variables
    double time_m;
    double dt_m;
    int it_m;
// grid related variables
    Vector_t<double, Dim> kw_m; // wave vector
    Vector_t<double, Dim> rmin_m; // minimum coordinates of the domain
    Vector_t<double, Dim> rmax_m; // maximum coordinates of the domain rmax_m = 2 * pi / Kw_m
    Vector_t<double, Dim> hr_m; // mesh spacing 
    Vector_t<double, Dim> origin_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
// self-field related variables
    double rhoNorm_m;
    double Q_m;
```

5. NanbuManager.h

6. Nanbu.cpp 

- main function 

- input parameters: 

    ```cpp
    //  totalP_: total number of particles
    //  nt_: number of time steps
    // nr_: number of cells in each direction
    // solver_ : FFT solver or Poisson solver
    // bool: computeSelfField_: compute the self-consistent field or not  

    ```
- run 
```cpp
//     srun ./DSMC_collision 64 64 64 100000 500 FFT false  --info 10
//     srun  ./DSMC_collision nr[0] nr[1] nr[2] totalP solver computeSelfField

```

7. FieldSolver.cpp 

- input : solverm rho, E, phi
- function: initiSolver() , runSolver()， initSolverWithParams
- private member: solver_m, rho_m, E_m, phi_m 

# RUN DSMC_COLLISION 


Change CMakeLists.txt 

```cmake
add_executable (Nanbu Nanbu.cpp)
target_link_libraries (Nanbu ${IPPL_LIBS})
```
 Go to the terminal and run the following commands

```bash 
module load gcc/11.4.0 cmake/3.26.3 cuda/12.1.1 openmpi/4.1.4 
./ippl-build-scripts/999-build-everything -t serial -k -f -i -u
```

Go to the build_serial and run the following command

```
cd /cluster/home/zhanghang/csp/IPPL_sing
cd /cluster/home/zhanghang/csp/IPPL_sing/ippl/build_serial/dsmc_collision 

mkdir data 

chmod 777 data 



./Nanbu 64 64 64 1000 10 FFT --info 10

./DSMC_Based 64 64 64 1000 10 0.01 FFT --info 10

```
