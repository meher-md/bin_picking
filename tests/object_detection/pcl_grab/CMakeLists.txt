cmake_minimum_required(VERSION 3.10)
project(RealSensePointCloudViewer)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(viewer main.cpp)
target_link_libraries(viewer ${PCL_LIBRARIES})
