cmake_minimum_required(VERSION 3.11)
project(semantic_icp_slam CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
message("Build type: " ${CMAKE_BUILD_TYPE})


# find dependencies
find_package(CUDA 9.2 REQUIRED)
find_package(Eigen3 3.2 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(OpenGL 2.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(g2o REQUIRED)
message(WARNING ${OpenCV_INCLUDE_DIRS})
# set python to the one installed in the virtual environment
set(PYTHON_LIBRARY /home/lk18493/anaconda3/envs/maskrcnn_benchmark/lib/)
set(PYTHON_INCLUDE_DIR /home/lk18493/anaconda3/envs/maskrcnn_benchmark/include/python3.7m/)
find_package(PythonLibs 3.7 REQUIRED)
message(WARNING ${PYTHON_INCLUDE_DIRS})

message(WARNING ${OpenCV_LIBRARIES})
message(WARNING ${PYTHON_LIBRARIES})

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/Core
${CMAKE_CURRENT_SOURCE_DIR}/Detecting
${CMAKE_CURRENT_SOURCE_DIR}/GUI
${CMAKE_CURRENT_SOURCE_DIR}/Mapping
${CMAKE_CURRENT_SOURCE_DIR}/Optimization
${CMAKE_CURRENT_SOURCE_DIR}/Tracking
${CMAKE_CURRENT_SOURCE_DIR}/Utility
${CUDA_INCLUDE_DIRS}
# ${OpenCV_INCLUDE_DIRS}
/usr/local/include/opencv2/
/usr/include/openni2
/usr/include/suitesparse
${PYTHON_INCLUDE_DIRS}
# /home/lk18493/anaconda3/envs/maskrcnn_benchmark/lib/python3.7/site-packages/numpy/core/include/
)

add_library(${PROJECT_NAME} SHARED
GUI/Viewer.cc

Mapping/DeviceMap.cu
Mapping/FuseMap.cu
Mapping/Mapping.cc
Mapping/MeshScene.cu
Mapping/RenderScene.cu

Core/Frame.cc
Core/Camera.cc
Core/KeyFrame.cc
Core/System.cc

Optimization/Optimizer.cc
Optimization/Solver.cc

Tracking/KeyPoints.cu
Tracking/Pyrdown.cu
Tracking/Reduction.cu
Tracking/Tracking.cc

Detecting/MaskRCNN.cpp
)

target_link_libraries(${PROJECT_NAME}
Eigen3::Eigen
${OpenCV_LIBRARIES}
${OpenGL_LIBRARIES}
${CUDA_LIBRARIES}
${Pangolin_LIBRARIES}
g2o_core
g2o_stuff
g2o_types_sba
g2o_solver_dense
g2o_solver_cholmod
${PYTHON_LIBRARIES}/libpython3.7m.so.1.0
)


set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# build examples
add_executable(test_output_tum MainTum.cc)
target_link_libraries(test_output_tum ${PROJECT_NAME})

add_executable(test_output_cam Main.cc)
target_link_libraries(test_output_cam ${PROJECT_NAME})
