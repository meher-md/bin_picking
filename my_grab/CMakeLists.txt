cmake_minimum_required(VERSION 3.1.0)
project(RealsensePointcloudExample)

# Basic setup
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 11)

# Add definitions
add_definitions(-DFLANN_SKIP_SERIALIZATION)
add_definitions(-DPCL_NO_PRECOMPILE)  # Helps with PCL compilation issues

# Include dependencies
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Open3D REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.11.1 REQUIRED)  # Ensure PCL version 1.11.1

# Include directories
include_directories(/usr/local/include /usr/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)  # Local include directory
include_directories(src)      # Include the src directory for headers
include_directories(src/functional_tests)  # Include functional tests headers if needed

# Link directories
link_directories(/usr/local/lib /usr/lib)

# Add executable for main application
add_executable(realsense_app
    src/pcd_frm_depth.cpp
    src/detect_color_bbox.cpp
    src/main.cpp
    src/pcd_capture.cpp
    src/object_detection.cpp
    src/cad_pcd.cpp
    src/visualization.cpp
)

target_link_libraries(realsense_app
    OpenGL::GL
    glfw
    GLU
    realsense2
    Open3D::Open3D
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    flann
)

# Add executable for functional tests
add_executable(functional_tests

    src/functional_tests/test_pcd_from_bbox.cpp
    src/functional_tests/test_pcd_from_bbox_opengl.cpp
    src/functional_tests/test_pcd_segmentation.cpp
    src/functional_tests/test_cad_overlay.cpp
    src/pcd_capture.cpp
    src/object_detection.cpp
    src/cad_pcd.cpp
    src/visualization.cpp
    src/pcd_frm_depth.cpp
    src/detect_color_bbox.cpp
)

target_link_libraries(functional_tests
    GTest::GTest
    GTest::Main
    Open3D::Open3D
    OpenGL::GL
    glfw
    GLU
    realsense2
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    flann
)

# Include Google Test integration
include(GoogleTest)
gtest_discover_tests(functional_tests)

# Installation rules
install(TARGETS realsense_app RUNTIME DESTINATION bin)
