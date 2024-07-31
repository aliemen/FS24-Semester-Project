#!/bin/bash
#SBATCH -n 2 # Number of cores
#SBATCH --time=90:00 # Runtime in minutes
#SBATCH --mem-per-cpu=2000              # memory per CPU core
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=nanbu_coulomb_sim   # Descriptive job name
#SBATCH --output=results/nanbu_sim_%j.out  # Write stdout to results/ dir with job ID
#SBATCH --error=results/nanbu_sim_%j.err   # Write stderr to results/ dir with job ID

cd ../..

module load gcc/11.4.0 cmake/3.26.3 cuda/12.1.1 openmpi/4.1.4 

# ./ippl-build-scripts/999-build-everything -t serial -k -f -i -u

# cd ippl/build_serial/05.03.dsmc/dsmc-code-simple
#cd ippl/build_openmp/05.03.dsmc/dsmc-code-simple # this for multi threading
cd build_openmp/dsmc-code-clone/dsmc-code
#cd build_serial/dsmc-code-clone/dsmc-code

export OMP_PROC_BIND=spread # set for multi threading to indicate that mpi can use ALL available cores
export OMP_PLACES=threads

# clean up data directory and timing file

#rm -rf data 

#rm -rf timingNB.dat

#rm -rf timingDSMC_Based.dat

mkdir -p data

chmod 777 data

# for diffent n paricles
# add "mpirun -np 4" before the command to run in parallel (with 4 cores)
# mpirun -np 4 ./Nanbu 64 64 64 156055 1000 0.01 FFT false true Nanbu 1 2.15623e-13 63699.28 --info 10

# ./Nanbu 16 16 16 2000000 5 0.01 FFT true Nanbu 1 --info 10
# ./Nanbu 64 64 64 156055 1000 0.01 FFT true Nanbu 1 2.15623e-13 --info 10

#For the cold sphere test case
#       grid     totalP N    balance       collisions    dt_m        vector_scale
#mpirun ./Nanbu 64 64 64 160000 1000 0.01 FFT true true Nanbu 1 2.15623e-13 63699.28 --info 10 # 156055; 160000 = 20^4 (for 4 ranks...)
# ./Nanbu 64 64 64 160000 1000 0.01 FFT true true TakAbe 1 327.59496 0.0 --info 10 # natural units # 7.9e-5
./Nanbu sphere 32 156055 1000 true true false true Nanbu 327.59496 0.0 1.0 10 --info 10 # 2.4e7 --> initial 150MeV bunch, dt=2.15623e-13s=327.59496/eV

#./Nanbu 32 32 32 2000000 40 0.01 FFT true Nanbu2 1 --info 10

#./Nanbu 32 32 32 2000000 40 0.01 FFT true Bird 1 --info 10


# For test case number one (error and convergence analysis) (Nanbu or TakAbe)
# ./Nanbu convergence 5    5.0  Nanbu 200   0.49049 false      --info 10
#         initial     real time coll  NPart nu0*dT  outp_debug 


# For the delta test case
#./Nanbu delta 32 1 5.0 TakAbe 2000 0.01 200 false true false --info 10
#./Nanbu delta   64        5           5.0     Nanbu 200    0.001    200     false       false         false      --info 10
#        initial gridsize  realization t_final algo  N_part dx_ratio N_steps adjust_dims use_collision debug_outp


# python plot.py 

