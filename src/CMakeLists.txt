# Get all source files
set(SOURCES
    bindings.cpp
    solver.cpp
    hybrid_pose_estimator.cpp
    hybrid_pose_shared_focal_estimator.cpp
    hybrid_pose_two_focal_estimator.cpp
)

# Create the pybind11 module
pybind11_add_module(madpose ${SOURCES})

# Add include directory for header files
target_include_directories(madpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(madpose PRIVATE 
    Eigen3::Eigen 
    PoseLib::PoseLib 
    Ceres::ceres 
    ${OpenCV_LIBS}
)

add_custom_command(TARGET madpose POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:madpose> ${CMAKE_SOURCE_DIR}/madpose
    COMMENT "Copying madpose library to ${CMAKE_SOURCE_DIR}/madpose"
)
