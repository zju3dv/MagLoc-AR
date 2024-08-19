cmake_minimum_required(VERSION 3.0.2)

# set the project name and version
project(state_estimation VERSION 1.0)

# output the configuration header
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/state_estimation)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/eigen)
include_directories(${PROJECT_SOURCE_DIR}/modules/pdr/include)
include_directories(${PROJECT_SOURCE_DIR}/modules/predict)

# set the cpp standard requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_BUILD_TYPE RELEASE)
#set(CMAKE_BUILD_TYPE DEBUG)

add_subdirectory(state_estimation)
add_subdirectory(modules/pdr)
add_subdirectory(modules/predict)

# set buildtool flags
set(CMAKE_CXX_FLAGS "-O3 -g -DRUNTIME_PROFILE -DPARTICLE_FILTER_VERBOSE -DRANDOMNESS_OFF")

# set global building variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(PATH_EXP ${PROJECT_SOURCE_DIR}/example)

add_executable(run_wireless_positioning ${PATH_EXP}/run_wireless_positioning.cc ${PATH_EXP}/WirelessPositioningMultiplexingAgentWoCovMapPrior.cc ${PATH_EXP}/compound_state_wireless_positioning_agent.cc)
target_link_libraries(run_wireless_positioning StateEstimation step_detection predict -pthread)
