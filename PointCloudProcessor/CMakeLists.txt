# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(PointCloudProcessor)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release for optimized performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenGL package
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Find Open3D package
find_package(Open3D REQUIRED)

# Find PCL package
find_package(PCL REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find RealSense2 package
find_package(realsense2 REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include directories
include_directories(
    ${Open3D_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# Link directories if necessary
link_directories(
    ${PCL_LIBRARY_DIRS}
    ${Open3D_LIBRARY_DIRS}
    ${realsense2_LIBRARY_DIRS}
    /usr/local/lib 
    /usr/lib
)

# Add executable
add_executable(PointCloudProcessor 
    src/main.cpp 
    src/helper_functions.cpp 
    src/pcd_saving.cpp 
    src/detect_color_bbox.cpp
    src/pcd_processing.cpp
    #src/pcl_to_open3d.cpp # Ensure this file exists and is needed
)

# # Add test executable
# add_executable(Test_PCDSaving 
#     tests/test_pcd_saving.cpp
#     src/detect_color_bbox.cpp
# )

# Link libraries
target_link_libraries(PointCloudProcessor 
    PRIVATE 
    Open3D::Open3D
    OpenGL::GL
    glfw
    GLU
    realsense2
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    flann
)

# # Link libraries for test executable
# target_link_libraries(Test_PCDSaving 
#     PRIVATE 
#     GTest::GTest
#     GTest::Main
#     Open3D::Open3D
#     OpenGL::GL
#     glfw
#     GLU
#     realsense2
#     ${OpenCV_LIBS}
#     ${PCL_LIBRARIES}
#     flann
# )
# # Enable CTest
# enable_testing()
# add_test(NAME PCDSavingTest COMMAND Test_PCDSaving)
# Add definitions if necessary
add_definitions(${PCL_DEFINITIONS})
