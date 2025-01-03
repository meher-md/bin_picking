cmake_minimum_required(VERSION 3.10)

# Project name
project(rs_server)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set build type to Release for optimized performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set CMAKE_PREFIX_PATH to prioritize Conda's libraries
set(CMAKE_PREFIX_PATH "/home/dhanuzch/miniconda3/envs/open3d2_env/")


# Find required packages
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

# Find Threads
find_package(Threads REQUIRED)
if(Threads_FOUND)
    message(STATUS "Threads library found")
else()
    message(FATAL_ERROR "Threads library not found")
endif()

# Find RealSense2 package
find_package(realsense2 REQUIRED)
if(realsense2_FOUND)
    message(STATUS "RealSense2 found: ${realsense2_LIBRARIES}")
    include_directories(${realsense2_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "RealSense2 not found")
endif()


# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_LIBS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Link directories if necessary
link_directories(
    ${realsense2_LIBRARY_DIRS}
    /usr/local/lib 
    /usr/lib
)

# Add the server executable
add_executable(rs_server rs_server.cpp)

# Link libraries
target_link_libraries(rs_server
    PRIVATE
    zmq
    nlohmann_json::nlohmann_json
    realsense2::realsense2
    ${OpenCV_LIBS}
)


# Include directories for headers
target_include_directories(rs_server 
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIRS}
    )

# Add the test executable
add_executable(rs_client tests/rs_client.cpp)

# Link libraries for the test executable
target_link_libraries(rs_client
    PRIVATE
    zmq
    nlohmann_json::nlohmann_json
    ${CMAKE_THREAD_LIBS_INIT}
)

# Include directories for the test executable
target_include_directories(rs_client
    PRIVATE
    ${CPPZMQ_INCLUDE_DIR}
    ${nlohmann_json_INCLUDE_DIRS}
)