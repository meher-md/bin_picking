# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(PointCloudProcessor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CMake policies to suppress deprecation warnings
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# Set build type to Release for optimized performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set CMAKE_PREFIX_PATH to prioritize Conda's libraries
set(CMAKE_PREFIX_PATH "/home/dhanuzch/miniconda3/envs/open3d2_env/")

# Find ZeroMQ
find_package(ZeroMQ REQUIRED)
if(ZeroMQ_FOUND)
    message(STATUS "ZeroMQ found: ${ZeroMQ_LIBRARIES}")
    include_directories(${ZeroMQ_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "ZeroMQ not found")
endif()

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)
if(nlohmann_json_FOUND)
    message(STATUS "nlohmann_json found: ${nlohmann_json_INCLUDE_DIRS}")
    include_directories(${nlohmann_json_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "nlohmann_json not found")
endif()

# Find cppzmq (if installed via package manager)
find_path(CPPZMQ_INCLUDE_DIR NAMES zmq.hpp PATHS /usr/local/include /usr/include)
if(CPPZMQ_INCLUDE_DIR)
    include_directories(${CPPZMQ_INCLUDE_DIR})
else()
    message(FATAL_ERROR "cppzmq not found")
endif()

# Find OpenGL package
find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
    message(STATUS "OpenGL found")
else()
    message(FATAL_ERROR "OpenGL not found")
endif()

# Find GLFW
find_package(glfw3 REQUIRED)
if(glfw3_FOUND)
    message(STATUS "glfw3 found")
else()
    message(FATAL_ERROR "glfw3 not found")
endif()

# Find Open3D package
find_package(Open3D REQUIRED)
if(Open3D_FOUND)
    message(STATUS "Open3D found")
    include_directories(${Open3D_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Open3D not found")
endif()

# Find PCL package
find_package(PCL REQUIRED)
if(PCL_FOUND)
    message(STATUS "PCL found: ${PCL_LIBRARIES}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else()
    message(FATAL_ERROR "PCL not found")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_LIBS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Find RealSense2 package
find_package(realsense2 REQUIRED)
if(realsense2_FOUND)
    message(STATUS "RealSense2 found: ${realsense2_LIBRARIES}")
    include_directories(${realsense2_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "RealSense2 not found")
endif()

# Find Google Test
find_package(GTest REQUIRED)
if(GTest_FOUND)
    message(STATUS "GTest found: ${GTEST_INCLUDE_DIRS}")
    include_directories(${GTEST_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "GTest not found")
endif()

# Find Threads
find_package(Threads REQUIRED)
if(Threads_FOUND)
    message(STATUS "Threads library found")
else()
    message(FATAL_ERROR "Threads library not found")
endif()

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

# Add executables
add_executable(PointCloudProcessor 
    src/main.cpp 
    src/helper_functions.cpp 
    src/pcd_saving.cpp 
    src/detect_color_bbox.cpp
    src/pcd_processing.cpp
    #src/pcl_to_open3d.cpp # Ensure this file exists and is needed
)

add_executable(TestBoundingBoxClient tests/test_bounding_box.cpp)

# Link libraries for PointCloudProcessor
target_link_libraries(PointCloudProcessor 
    PRIVATE 
    Open3D::Open3D
    OpenGL::GL
    glfw
    GLU
    realsense2
    zmq
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    flann
    Threads::Threads
)

# Link libraries for TestBoundingBoxClient
target_link_libraries(TestBoundingBoxClient 
    PRIVATE 
    zmq    
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

# Add definitions if necessary
add_definitions(${PCL_DEFINITIONS})
