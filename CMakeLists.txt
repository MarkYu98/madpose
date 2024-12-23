cmake_minimum_required(VERSION 3.17)
project(madpose)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version needs to be at least 9.1")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++17 -std=gnu++17 -fPIC") # brute-force approach to make sure we are using C++17

find_package(Eigen3 3.3 REQUIRED)
find_package(COLMAP REQUIRED)
message(STATUS "Found COLMAP: ${COLMAP_DIR}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(PoseLib REQUIRED)
find_package(OpenCV REQUIRED)
if(${CERES_VERSION} VERSION_LESS "2.2.0")
    # ceres 2.2.0 changes the interface of local parameterization
    add_definitions("-DCERES_PARAMETERIZATION_ENABLED")
endif()

if(EIGEN3_FOUND)
  message("-- Found Eigen version ${EIGEN3_VERSION_STRING}: ${EIGEN3_INCLUDE_DIRS}")
endif()

if(COLMAP_FOUND)
  message("-- Found COLMAP version ${COLMAP_VERSION}: ${COLMAP_INCLUDE_DIRS}")
endif()

add_subdirectory(ext/pybind11)

include_directories(BEFORE
    ${pybind11_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    # ${COLMAP_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ext/RansacLib
)
include_directories(
    ${COLMAP_DIR}/../../include
    ${COLMAP_INCLUDE_DIRS}
)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

# message(STATUS "COLMAP_INCLUDE_DIRS:${COLMAP_INCLUDE_DIRS}")
# message(STATUS "COLMAP_LIBRARIES:${COLMAP_LIBRARIES}")
list(REMOVE_ITEM COLMAP_LIBRARIES colmap_cuda sift_gpu)
message("COLMAP_LIBRARIES:${COLMAP_LIBRARIES}")
message("COLMAP_EXPORT_LIBS:${COLMAP_EXPORT_LIBS}")
link_directories(${COLMAP_LINK_DIRS})

add_subdirectory(src)
