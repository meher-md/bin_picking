cmake_minimum_required(VERSION 3.10)
project(PointCloudRegistration)

set(CMAKE_CXX_STANDARD 17)

# Include Open3D
find_package(Open3D REQUIRED)

# Add executable
add_executable(point_cloud_registration point_cloud_registration.cpp)

# Link Open3D
target_link_libraries(point_cloud_registration PRIVATE Open3D::Open3D)
