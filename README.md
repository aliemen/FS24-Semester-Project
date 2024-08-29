# Monte Carlo Solution to Coulomb Collisions using IPPL
The repository to my semester project in computational physics in FS24.


# RUN THE CODE

After installing IPPL, paste the `dsmc-code` folder inside the downloaded `ippl` folder. In that folder add the code to the CMakeLists.txt:

```cmake
..............
    FetchContent_MakeAvailable(googletest)

    add_subdirectory (unit_tests)
endif ()

# add these to lines relatively at the end
add_subdirectory (05.03.dsmc/dsmc-code-simple)
message (STATUS "Added 05.03.dsmc source.")

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake )

install (
..............
```
 Go to the terminal and run the following commands

```bash 
module load gcc/11.4.0 cmake/3.26.3 cuda/12.1.1 openmpi/4.1.4 
./ippl-build-scripts/999-build-everything -t serial -k -f -i -u
./ippl-build-scripts/999-build-everything -t openmp -k -f -i -u
./ippl-build-scripts/999-build-everything -t cuda -k -f -i -u
```
Choose how you want to compile the code. Then there are three possibilities to run the code (inside the `build_...` folder):
1. `./Nanbu convergence 5 5.0 Nanbu 200 0.49049 false --info 10`: Choose the Trubnikov test (with "convergence"), `5` is the number of realizations, `5.0` is the final time of the simulation, `Nanbu` is the collision model (also possible: `TakAbe`, `Nanbu2` and `Bird`), `200` is the number of particles, `0.49049` is $\nu_0\Delta t$ and `false` inidicates no debug output.
2. `./Nanbu delta 64 5 5.0 Nanbu 200 0.001 200 false false false --info 10`: Choose delta "function" as initial distribution, a `64^3` grid, `5` realizations, `5.0` as the final time, `Nanbu` as the collision algorithm, `200` as number of particles, `0.001` as $d_x$ (explained in the report), and three bools (use adaptive mesh grid, use collisions and output debug).
3. `./Nanbu sphere 32 156055 1000 true true false true Nanbu 327.59496 0.0 1.0 10 --info 10`: Choose the "Cold Sphere Heating" testcase, a `32^3` grid, `156055` particles, `1000` timesteps, four bools (use self consisten electrical field, compute collisions, output debug, use adaptive mesh), `Nanbu` is the collision algorithm, `327.59496` is the timestepsize, `0.0` is a initial velocity scaling factor, `1.0` is a multipliert to the confinement force and `10` is the number of realizations.

Make sure to create a `data` folder in the same folder as the `Nanbu` executable.
