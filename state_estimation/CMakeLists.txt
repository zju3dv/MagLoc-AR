set(YAMLCPP_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rd_party/yaml-cpp/include)

message("StateEstimation build type: " ${CMAKE_BUILD_TYPE})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../3rd_party/yaml-cpp ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)

add_library(StateEstimation STATIC
    ${CMAKE_CURRENT_LIST_DIR}/util/misc.cc
#    ${CMAKE_CURRENT_LIST_DIR}/util/lie_algebra.cc
    ${CMAKE_CURRENT_LIST_DIR}/util/result_format.cc
    ${CMAKE_CURRENT_LIST_DIR}/configuration/configuration.cc
    ${CMAKE_CURRENT_LIST_DIR}/variable/position.cc
    ${CMAKE_CURRENT_LIST_DIR}/variable/orientation.cc
    ${CMAKE_CURRENT_LIST_DIR}/offline/client_data_reader.cc
#    ${CMAKE_CURRENT_LIST_DIR}/offline/trajectory_reader.cc
    ${CMAKE_CURRENT_LIST_DIR}/offline/results_reader.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_constant_velocity.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_2d_position_2nd_order_continuity.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_cvcr.cc
#    #${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_cvcr_yaw.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_cvcr_yaw_1st_continuity.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_2d_velocity_diff_yaw.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_2d_displacement.cc
    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_yaw_differential.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_3d_orientation.cc
    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/parameter_model_random_walk.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/kalman_filter_model.cc
    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/compound_prediction_model.cc
    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_2d_local_velocity_1d_rotation.cc
#    ${CMAKE_CURRENT_LIST_DIR}/prediction_model/motion_model_2d_local_velocity_3d_rotation.cc
    ${CMAKE_CURRENT_LIST_DIR}/distribution/distribution_map.cc
#    ${CMAKE_CURRENT_LIST_DIR}/distribution/distribution_map_multivariate_gaussian.cc
    ${CMAKE_CURRENT_LIST_DIR}/distribution/probability_mapper_2d.cc
#    ${CMAKE_CURRENT_LIST_DIR}/distribution/probability_mapper_cov.cc
    ${CMAKE_CURRENT_LIST_DIR}/observation_model/bluetooth_observation_model.cc
    ${CMAKE_CURRENT_LIST_DIR}/observation_model/wifi_observation_model.cc
    ${CMAKE_CURRENT_LIST_DIR}/observation_model/orientation_observation_model.cc
    ${CMAKE_CURRENT_LIST_DIR}/observation_model/geomagnetism_observation_model.cc
    ${CMAKE_CURRENT_LIST_DIR}/observation_model/fusion_observation_model.cc
#    ${CMAKE_CURRENT_LIST_DIR}/observation_model/imu_observation_model.cc
    #${CMAKE_CURRENT_LIST_DIR}/visualization/visualizer.cc
)

#target_include_directories(StateEstimation PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${YAMLCPP_INCLUDE_DIR} ${PANGOLIN_INCLUDE_DIR})
#target_link_libraries(StateEstimation yaml-cpp pangolin)
target_include_directories(StateEstimation PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${YAMLCPP_INCLUDE_DIR})
target_link_libraries(StateEstimation yaml-cpp)
