find_package(MPI REQUIRED)

add_library(ginkgo_mpi $<TARGET_OBJECTS:ginkgo_mpi_device> "")
target_sources(ginkgo_mpi
    PRIVATE
        base/exception.cpp
        base/executor.cpp
        base/version.cpp
        components/precision_conversion.cpp
        components/prefix_sum.cpp
        factorization/ilu_kernels.cpp
        factorization/factorization_kernels.cpp
        factorization/par_ilu_kernels.cpp
        matrix/coo_kernels.cpp
        matrix/csr_kernels.cpp
        matrix/dense_kernels.cpp
        matrix/ell_kernels.cpp
        matrix/hybrid_kernels.cpp
        matrix/sellp_kernels.cpp
        matrix/sparsity_csr_kernels.cpp
        preconditioner/isai_kernels.cpp
        preconditioner/jacobi_kernels.cpp
        reorder/rcm_kernels.cpp
        solver/bicg_kernels.cpp
        solver/bicgstab_kernels.cpp
        solver/cg_kernels.cpp
        solver/cgs_kernels.cpp
        solver/fcg_kernels.cpp
        solver/gmres_kernels.cpp
        solver/ir_kernels.cpp
        solver/lower_trs_kernels.cpp
        solver/upper_trs_kernels.cpp
        stop/criterion_kernels.cpp
        stop/residual_norm_reduction_kernels.cpp)

ginkgo_compile_features(ginkgo_mpi)
target_include_directories(ginkgo_mpi
  SYSTEM PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(ginkgo_mpi PRIVATE "${MPI_C_LIBRARIES}" "${MPI_CXX_LIBRARIES}")
target_compile_options(ginkgo_mpi PRIVATE "${GINKGO_COMPILER_FLAGS}")

# Need to link to allow mpi kernels to call other kernels.
target_link_libraries(ginkgo_mpi PUBLIC ginkgo_cuda ginkgo_omp ginkgo_reference ginkgo_hip)

ginkgo_default_includes(ginkgo_mpi)
ginkgo_install_library(ginkgo_mpi mpi)

if (GINKGO_HAVE_HWLOC)
  target_link_libraries(ginkgo_mpi PRIVATE ${HWLOC_hwloc_LIBRARY})
  target_include_directories(ginkgo_mpi PRIVATE ${HWLOC_hwloc.h_DIRS})
endif()

if(GINKGO_BUILD_TESTS)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_subdirectory(test)
endif()

# Propagate some useful information
set(MPI_C_VERSION ${MPI_C_VERSION} PARENT_SCOPE)
set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
