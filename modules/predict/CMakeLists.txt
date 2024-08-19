cmake_minimum_required(VERSION 3.1)
project(Predict)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") #RelWithDebInfo
endif ()
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "-std=c++14")

include_directories(${PROJECT_SOURCE_DIR}/3rd_party/eigen)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/filter_predict)

set(BUILD_TESTING OFF)
set(GFLAGS_BUILD_TESTING OFF)
set(GFLAGS_BUILD_STATIC_LIBS ON)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rd_party/glog)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rd_party/gflags)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/3rd_party/glog)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/3rd_party/gflags/include)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(predict STATIC
        sophus/so3.cpp
        sophus/se3.cpp
        filter_predict/PredictFilter.cc
        IMU/imu_predict.cc
        IMU/IMUPreIntegration.cc
        filter_predict/butter.cpp
        )
target_link_libraries(predict glog gflags)
