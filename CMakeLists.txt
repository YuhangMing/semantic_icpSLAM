cmake_minimum_required(VERSION 2.8)
project(fusion LANGUAGES CXX CUDA)

# defaults build to release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("-- Build type: " ${CMAKE_BUILD_TYPE})
endif()

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(fusion SHARED "")

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
include(cmake/FindOpenNI2.cmake)

find_package(Python3 COMPONENTS Development)
message("-- Python Dir: " ${Python3_INCLUDE_DIRS})
message("-- Python Lib: " ${Python3_LIBRARIES})
# message("-- OpenCV Lib: " ${OpenCV_LIBRARIES})

target_include_directories(fusion
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

target_compile_features(fusion
PUBLIC
    cxx_std_11
)

set_target_properties(fusion
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_link_libraries(fusion
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${Python3_LIBRARIES}
)

target_sources(fusion
PRIVATE
    src/system.cpp
    src/map_manager.cpp
    
    src/data_struct/rgbd_frame.cpp
    src/data_struct/map_struct.cu

    src/tracking/icp_tracker.cpp
    src/tracking/rgbd_odometry.cpp
    src/tracking/device_image.cpp
    src/tracking/cuda_imgproc.cu
    src/tracking/m_estimator.cu
    src/tracking/pose_estimator.cu
    src/tracking/build_pyramid.cpp

    src/features/matcher.cpp
    src/features/extractor.cpp

    src/keyframe_graph/keyframe_graph.cpp

    src/relocalization/ransac_ao.cpp
    src/relocalization/relocalizer.cpp

    src/visualization/main_window.cpp

    src/voxel_hashing/voxel_hashing.cpp
    src/voxel_hashing/depth_fusion.cu
    src/voxel_hashing/ray_tracing.cu
    src/voxel_hashing/scene_mesh.cu

    src/input/tum_loader.cpp
    # src/input/oni_camera.cpp

    src/detection/MaskRCNN.cpp


)

add_subdirectory(example)
include(cmake/InstallGLSL.cmake)