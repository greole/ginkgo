cmake_minimum_required(VERSION 3.9)
project(heat-equation)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()
find_package(OpenCV REQUIRED)

add_executable(heat-equation heat-equation.cpp)
target_link_libraries(heat-equation Ginkgo::ginkgo ${OpenCV_LIBS})

# Copy the data files to the execution directory
configure_file(data/initial.mtx data/initial.mtx COPYONLY)
configure_file(data/source.mtx data/source.mtx COPYONLY)
