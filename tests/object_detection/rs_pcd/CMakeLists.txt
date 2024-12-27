# Minimum required CMake version
cmake_minimum_required(VERSION 3.1.0)

# Project name
project(RealsensePointcloudExample)

# Enable compile commands export
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Include OpenGL configuration
include(cmake/opengl_config.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include directories
include_directories(src)
include_directories(include)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Open3D REQUIRED)

# Add the executable
add_executable(rs-pointcloud 
src/rs-pointcloud.cpp 
src/object_detection.cpp     
src/cad_pcd.cpp 
)

# Link required libraries
target_link_libraries(rs-pointcloud 
    OpenGL::GL
    glfw
    GLU
    realsense2
    Open3D::Open3D

)

# Installation rule (optional)
install(TARGETS rs-pointcloud RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
