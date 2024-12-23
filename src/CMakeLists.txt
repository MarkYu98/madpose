# Get all source files
set(SOURCES
    bindings.cpp
    solver.cpp
    hybrid_pose_estimator.cpp
    hybrid_pose_shared_focal_estimator.cpp
    hybrid_pose_two_focal_estimator.cpp
)

# Create the pybind11 module
pybind11_add_module(acmpose ${SOURCES})

# Add include directory for header files
target_include_directories(acmpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(acmpose PRIVATE 
    Eigen3::Eigen 
    PoseLib::PoseLib 
    Ceres::ceres 
    colmap::colmap 
    ${OpenCV_LIBS}
)