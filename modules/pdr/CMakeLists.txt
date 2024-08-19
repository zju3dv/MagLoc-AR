cmake_minimum_required(VERSION 3.0)
project(pdr)

set(EIGEN3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../../3rd_party/eigen/)

# find_package(Eigen3)
# if(NOT EIGEN3_FOUND)
#     set(EIGEN3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../../3rd_party/eigen/)
# else()
#     set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
# endif()

# always generate PIC code, no matter static or shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-DHAVE_CONFIG_H")

set(PATH_INCLUDE ${PROJECT_SOURCE_DIR}/include)

include_directories(
        ${PATH_INCLUDE}/
        ${EIGEN3_INCLUDE_DIRS}
)

add_library(
        step_detection STATIC
        src/util.cc
        src/pdr.cc
)

