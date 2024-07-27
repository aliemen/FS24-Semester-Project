file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (Nanbu Nanbu.cpp)
target_link_libraries (Nanbu ${IPPL_LIBS})

message (STATUS "IPPL_LIBS = ${IPPL_LIBS}" )


# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
