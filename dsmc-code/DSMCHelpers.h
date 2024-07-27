#ifndef IPPL_DSMC_HELPERS_H
#define IPPL_DSMC_HELPERS_H

#include "Ippl.h"
// #include <Kokkos_Core.hpp> // for the MinMaxReducer (should already be implemented inside Ippl.h)

#include <cmath>  // for std::exp
#include <random>
#include <vector>
#include <iterator> // for std::inserter
#include <tuple> 
#include <chrono> // for timestamp

// add __constant__ for cuda compilation (use "const" for openmp and serial)
//__constant__  double e0 = 1.0; // 8.85e-12; // Vacuum permittivity
//__constant__  double kB = 1.0; // 1.38e-23; // Boltzmann constant
const double e0 = 1.0; // 8.85e-12; // Vacuum permittivity
const double kB = 1.0; // 1.38e-23; // Boltzmann constant

/*
Need to set a threshold for the s value, since otherwise we get nan velocity updates.
NaN velocity updates fed into the field solver results in segmentation faults!
*/
const std::pair<double, double> s_thresholds = {0.0005, 15}; //  use 0.0005 instead of 0.005 to avoid weired Nanbu non-convergence 

std::vector<double> s_value_lookup = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                                      0.09, 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,
                                      0.8,  0.9,  1,    2,    3,    4};
std::vector<double> A_value_lookup = {100.5, 50.50, 33.84, 25.50,  20.50,  17.17,  14.79, 13.01,
                                      11.62, 10.51, 5.516, 3.845,  2.987,  2.448,  2.067, 1.779,
                                      1.551, 1.363, 1.207, 0.4105, 0.1496, 0.05496};

template <typename value_t>
value_t getA(const value_t& s) {
    static IpplTimings::TimerRef solve_A_timer = IpplTimings::getTimer("getA");
    IpplTimings::startTimer(solve_A_timer);

    if (s < 0.01) { return 1.0 / s; }
    if (s >= 4)   { return 3 * std::exp(-s); }

    // If we have 0.01 <= s <= 4, we can use the lookup table and just interpolate
    // linearly between the table values.

    // First iterate through s values and find smin, smax
    double smin = 0.01, smax = 4;
    double Amin = 100.5, Amax = 0.05496;
    for (long unsigned int i = 1; i < s_value_lookup.size(); ++i) {
        if (s <= s_value_lookup[i]) {
            smin = s_value_lookup[i - 1];
            smax = s_value_lookup[i];
            Amin = A_value_lookup[i - 1];
            Amax = A_value_lookup[i];
            break;
        }
    }

    // Then stop timer and interpolate linearly between smin and smax
    double m = (Amax - Amin) / (smax - smin);  // slope of linear interpolation
    IpplTimings::stopTimer(solve_A_timer);
    return m * s + (Amin - m * smin);
}


std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> genPairsFromMeshComposition(
    std::unordered_map<size_t, std::vector<size_t>>& mesh_decomp, std::mt19937& eng) { 
    std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> mesh_pairs;
    // First shuffle all the indices per cell!
    for (auto& [cellID, cell_vec] : mesh_decomp) {
        // std::vector<size_t>& cell_vec = x.second;
        std::shuffle(cell_vec.begin(), cell_vec.end(),
                        eng);  // Shuffle the indices using the random number generator

        std::vector<std::pair<size_t, size_t>> paired_idx;
        for (size_t i = 0; i < cell_vec.size() - 1; i += 2) {  // Pair the indices
            paired_idx.push_back({cell_vec[i], cell_vec[i + 1]});
        }

        mesh_pairs[cellID] = paired_idx;
    }

    return mesh_pairs;
}

size_t getUniqueParticleID(Vector_t<size_t, Dim> vec, const Mesh_t<Dim>::vector_type& gridsize) {
    size_t id = 0;
    for (size_t i = 0; i < Dim; ++i) {
        id = id * gridsize[i] + vec(i);
    }
    return id;
}


/*
Here we have some helper functions to calcualte 1/tau_1 which is used for s
in solving f(A) = exp(-s) for A (which is used for angle sampling).
*/
using vector_view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using scalar_view_type = typename ippl::ParticleAttrib<double>::view_type; // ippl::detail::ViewType<double, 1>::view_type;

//double getCellT(const vector_t& p_indices, scalar_view_type& masses, ) {
//    return 1.0;
//}


double getTemperature(vector_view_type* P, int NPart) {
    double temp = 0.0;
    Kokkos::parallel_reduce(
        "v_y_sq", (*P).extent(0),
        KOKKOS_LAMBDA(const int i, double& valL) {
            double myVal = dot((*P)(i), (*P)(i)).apply(); // (*P)(i)[1] * (*P)(i)[1];
            valL += myVal;
        },
        Kokkos::Sum<double>(temp));
    temp /= 3 * NPart;
    return temp;
}

double coulombLog(double& b0) { // const double& temperature, const double& rho, 
    //double lmdb_D = 7.4e-6/1.973e-7; // according to Nanbu paper (4649, right side E.))
    //return std::log(lmdb_D/b0);
    b0 *= 1.0;
    //return 23 - std::log(std::sqrt(rho)*std::pow(temperature, -1.5));
    //return std::sqrt(temperature / (4*pi*rho));
    return 10.0; // suggested by https://en.wikipedia.org/wiki/Coulomb_collision#Coulomb_logarithm
}

double getInvTau1(const size_t& i1, const size_t& i2, 
                  const scalar_view_type* masses, const scalar_view_type* charges,
                  const double& rho, const double& V_n,
                  const std::string& test_case, double& T_total) {
    /*
    V_n:    Absolute value of relative velocity |v1-v2|.
    i1, i2: Indices of both colliding particles, used to access charges, masses view.
    rho:    Particle (mass) density in our current cell.
    */
    double m1  = (*masses)(i1),  m2 = (*masses)(i2);
    double q1  = (*charges)(i1), q2 = (*charges)(i2);
    double m_r = m1*m2/(m1+m2);     // reduced mass
    //std::cout << m1 << " - " << m2 << ", " << q1 << " - " << q2;
    double lnLambda;
    if (test_case == "convergence") {
        double T_total = 0.008/3 + 2*0.01/3;
        lnLambda = std::sqrt(T_total / (4*pi*rho)); // coulombLog(); // Coulomb logarithm    
        return lnLambda/(4*pi)*std::pow(q1*q2 / m_r, 2) * rho*rho / std::pow(V_n, 3);
    } else {
        // lnLambda = 
        double b0 = std::abs(q1*q2) / (2*pi*e0*m_r * T_total*3);
        lnLambda = coulombLog(b0); // T_total, rho, 
        return 4*pi*std::pow(q1*q2 / (4*pi*e0*m_r), 2) *rho*rho*lnLambda/std::pow(V_n, 3);
    }
}

/*
Find maximum tupel in a map.
Taken from https://stackoverflow.com/a/34937216
*/
template<typename KeyType, typename ValueType> 
std::pair<KeyType,ValueType> getMapMax( const std::unordered_map<KeyType,ValueType>& x ) {
  using pairtype=std::pair<KeyType,ValueType>; 
  return *std::max_element(x.begin(), x.end(), [] (const pairtype & p1, const pairtype & p2) {
        return p1.second < p2.second;
  }); 
}

/*
A function to sample to not equal values from a vector
of indices.
*/
std::pair<size_t, size_t> randElementPair(const std::vector<size_t>& ind_arr, std::mt19937& gen) {
    std::vector<size_t> output_pair(2);
    std::sample(ind_arr.begin(), ind_arr.end(), std::inserter(output_pair, output_pair.begin()), 2, gen);
    return {output_pair[0], output_pair[1]};
}

// Calculate norm of a ippl vector
//double normIPPLVector(const ippl::Vector<double, 3> v, bool perp);
double normIPPLVector(const ippl::Vector<double, 3> v, bool perp) {
    return perp ? std::sqrt(std::pow(v[1], 2) + std::pow(v[2], 2))
                : std::sqrt(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2));
}

// Calculate dot product of two IPPL vectors
double dotIPPLVector(const ippl::Vector<double, 3>& v1, const ippl::Vector<double, 3>& v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

ippl::Vector<double, 3> centerOfMass(const vector_view_type* R) {
    int NPart = (*R).extent(0);
    ippl::Vector<double, 3> com = {0.0, 0.0, 0.0};
    Kokkos::parallel_reduce(
        "CenterOfMass", NPart,
        KOKKOS_LAMBDA(const int i, ippl::Vector<double, 3>& com) {
            com += (*R)(i);
        }, com);
    return com / NPart;
} 

// A function for the disorder induced heating process. Calculates the "linear confinement force" using the position and the total charge,
// since the force is dependent on the "average induced force" by a homogeneous charge distribution/particle cloud
vector_view_type getDisorderInducedHeatingForcesView(const vector_view_type* R, const scalar_view_type* q,
                                                     //const ippl::Vector<double, 3>& rmin, const ippl::Vector<double, 3>& rmax,
                                                     const vector_view_type* E, const double& confinementForceAdjust) {
    //ippl::Vector<double, 3> pre_factor = {pre_Factor_double, pre_Factor_double, pre_Factor_double};
    //vector_view_type forces = pre_factor; 

    // Required by the test case of the disorder induced heating process (unit meters)
    // const double bunch_R = 90.11657; // 17.78e-6;
    //ippl::Vector<double, 3> bunch_center = centerOfMass(R); // 0.5 * (rmin + rmax);
    ippl::Vector<double, 3> bunch_center = 506.84 / 2;
    
    ippl::Vector<double, 3> E_average_foc = centerOfMass(E) * confinementForceAdjust;
    //ippl::Vector<double, 3> test = (*R)(0)-bunch_center;
    //std::cout << test << " " << (*R)(0) << " ";
    //test /= normIPPLVector(test, false);
    //std::cout << test << std::endl;

    //double pre_factor_double = -totalQ / (4*pi*e0); // this parameter is the same for every component
    vector_view_type forces("forces", R->extent(0)); // Create a Kokkos::View with size equal to the number of particles
    // Assign the values from pre_factor to the elements of forces
    Kokkos::parallel_for("ConfinementForcesInit", forces.extent(0), KOKKOS_LAMBDA(const int i) {
        ippl::Vector<double, 3> relative_r = (*R)(i) - bunch_center; // position minus center of particle bunch (as given by the initialization, which is middle of the grid)
        double distance                    = normIPPLVector(relative_r, false); // distance from the center of the bunch
        
        if (distance < 1e-10) { distance = 1e-10; } // Avoid division by zero
        relative_r /= distance; // unit vector pointing from the center of the bunch to the particle^
        
        forces(i) = -relative_r * dotIPPLVector(relative_r, (*E)(i)) * (*q)(i) / e0 * confinementForceAdjust; // pre_factor_double;
        //forces(i) = -relative_r * -47256.887/(std::pow(distance, 2)*4*pi); // pre_factor_double;
        //double E_magnitude = normIPPLVector((*E)(i), false);
        //forces(i) = -relative_r * E_magnitude * (*q)(i) / e0; // pre_factor_double;
        //forces(i) = (*q)(i)*E_average_foc;
    });
    Kokkos::fence();

    //std::cout << forces(0) << " - " << (*q)(0) << std::endl;

    /*
    // Get the center of the total simulation domain
    ippl::Vector<double, 3> bunch_center = 0.5 * (rmin + rmax);

    // Now iterate through particles to determine if they are inside/outsize the sphere!
    // R.extent(0) gives the number of components in the 0th dimension (so index for every particle...).
    for (size_t i = 0; i < R->extent(0); ++i) {
        
        //Calculate current radial force acting on the particle using the electric field that was 
        //calculated in the previous timestep.
        
        ippl::Vector<double, 3> relative_r = (*R)[i] - bunch_center; // position minus center of particle bunch (as given by the initialization, which is middle of the grid)
        double distance                    = normIPPLVector(relative_r, false); // distance from the center of the bunch
        relative_r                        /= distance; // unit vector pointing from the center of the bunch to the particle^
         
        forces[i] *= relative_r*(*q)[i];
        forces[i] *= (distance >= bunch_R) ? 1/std::pow(distance, 2) : distance/std::pow(bunch_R, 3); // finally calculate force value
        forces(i) = -relative_r * dotIPPLVector(relative_r, (*E)[i]) * (*q)[i] / e0; // "-" is definitely necessary!
    }*/
    return forces;
}

void applyReflectingBoundaryConditions(vector_view_type* R, vector_view_type* v, const double radius) {
    ippl::Vector<double, 3> bunch_center = 506.84 / 2;
    // Iterate through velocities using Kokkos for
    Kokkos::parallel_for("ReflectingBoundary", R->extent(0), KOKKOS_LAMBDA(const int i) {
        // Calculate distance from the center of the sphere
        ippl::Vector<double, 3> relative_r = (*R)(i) - bunch_center; // position minus center of particle bunch (as given by the initialization, which is middle of the grid)
        double distance                    = normIPPLVector(relative_r, false);
        
        // Check if the particle is outside the sphere
        if (distance > radius) {
            // Normalize the position vector
            ippl::Vector<double, 3> norm_r = relative_r / distance; // (*R)(i) / distance;

            // Reflect the position
            (*R)(i) = radius * norm_r + bunch_center; 

            // Reflect the velocity
            (*v)(i) = -(*v)(i); // Reverse the components of velocity
        }
    });
}

std::tuple<double, double, double, double> genPhiThetaTakizukaAbe(const double& dt, const double& u, const size_t& i1, const size_t& i2, 
                                                                  const scalar_view_type* masses, const scalar_view_type* charges,
                                                                  const double& rho, 
                                                                  std::mt19937& gen, std::uniform_real_distribution<>& dis,
                                                                  double& temperature, const std::string& test_case) {
    // Calculate relative mass
    double m1  = (*masses)(i1), m2 = (*masses)(i2);
    double m12 = m1*m2/(m1 + m2);

    // get charges
    double q1 = (*charges)(i1), q2 = (*charges)(i2);

    // Calculate "collision temperature" (correct close to equilibrium, suggested by paper...)
    // double T      = 9.333e-3; // for convergence testcase // m12*std::pow(u, 2)/(3*kB);
    // double lmbd   = std::sqrt(kB*T/(4*pi*rho/2*std::pow(q1, 2))); // coulombLog(); // 
    // double lmbd   = std::sqrt(9.3333e-3 / (4*pi*rho)); // better approximation for known temperature
    if (test_case == "convergence") temperature = 0.008/3 + 2*0.01/3;
    
    double b0 = std::abs(q1*q2) / (2*pi*e0*m12 * temperature*3);
    double lmbd = coulombLog(b0); // temperature, rho, 
    // double lmbd    = std::sqrt(temperature / (4*pi*rho)); // coulombLog(temperature, rho); // std::sqrt(T_total / (4*pi*rho)); // 23 - std::log(std::sqrt(rho*1e12)*std::pow(T_total, -1.5)); // 10.0; //  // 
    double sgm_sq  = std::pow(q1*q2, 2)*lmbd*rho / (8*pi*std::pow(e0*m12, 2)*std::pow(u, 3)) * dt;

    // Get the normal distributionand sample numbers
    std::normal_distribution<double> n_distr(0.0, std::sqrt(sgm_sq));
    double delta = n_distr(gen);
    double phi   = 2*pi*dis(gen);

    // Transform into collision angles
    double sinTh = 2*delta / (1 + std::pow(delta, 2)); 
    double cosTh = 1 - 2*std::pow(delta, 2) / (1 + std::pow(delta, 2));

    return std::make_tuple(std::sin(phi), std::cos(phi),
                           sinTh, cosTh);
}

// The following custom struct is used to find the Nd minimum and maximum simultaneously in a Kokkos:view
template<int Dim>
struct MinMaxReducer {
    Vector_t<double, Dim> min_val;
    Vector_t<double, Dim> max_val;

    KOKKOS_INLINE_FUNCTION
    MinMaxReducer() {
        for (int d = 0; d < Dim; ++d) {
            min_val[d] = std::numeric_limits<double>::max();
            max_val[d] = std::numeric_limits<double>::lowest();
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, MinMaxReducer& update) const {
        for (int d = 0; d < Dim; ++d) {
            update.min_val[d] = std::min(update.min_val[d], min_val[d]);
            update.max_val[d] = std::max(update.max_val[d], max_val[d]);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void join(const MinMaxReducer& update) {
        for (int d = 0; d < Dim; ++d) {
            min_val[d] = std::min(min_val[d], update.min_val[d]);
            max_val[d] = std::max(max_val[d], update.max_val[d]);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const MinMaxReducer& update) { // Need this to compile with openmp
        join(update);
    }
};


template<typename ViewType, int Dim>
void findMinMax(const ViewType* R, MinMaxReducer<Dim>& minR) { 
    Kokkos::parallel_reduce("FindMinMax", (*R).extent(0), KOKKOS_LAMBDA(const int i, MinMaxReducer<Dim>& reducer) {
        auto value = (*R)(i); 
        for (int d = 0; d < Dim; ++d) {
            reducer.min_val[d] = std::min(reducer.min_val[d], value[d]);
            reducer.max_val[d] = std::max(reducer.max_val[d], value[d]);
        }
    }, minR);
}


long long timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string formattedTimestamp(long long milliseconds) {
    // Get current time in milliseconds
    //auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
    //    std::chrono::system_clock::now().time_since_epoch()).count();

    // Convert milliseconds to time_point
    auto timePoint = std::chrono::system_clock::time_point(std::chrono::milliseconds(milliseconds));
    
    // Convert to time_t
    std::time_t time = std::chrono::system_clock::to_time_t(timePoint);
    
    // Get milliseconds part
    int ms = milliseconds % 1000;

    // Format the time
    std::tm* timeInfo = std::localtime(&time);
    
    std::ostringstream oss;
    oss << std::put_time(timeInfo, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms;

    return oss.str();
}

#endif
