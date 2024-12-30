#include <GL/glew.h>    // GLEW must be included before any OpenGL headers
#include <GLFW/glfw3.h> // GLFW for window and context management
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "example.hpp"
#include "pcd_frm_depth.hpp"
#include "detect_color_bbox.hpp"
#include <gtest/gtest.h>


TEST(RealTimeBoundingBoxAndPointCloud_OpenGL, RealSenseStream) {
    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    auto profile = pipe.start();

    // Get camera intrinsics
    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

    // Define the HSV range for the orange color
    cv::Scalar lower_orange(0, 100, 100);
    cv::Scalar upper_orange(25, 255, 255);

    // Initialize GLFW
    if (!glfwInit()) {
        FAIL() << "Failed to initialize GLFW";
    }

    // Create a GLFW window
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Point Cloud Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        FAIL() << "Failed to create GLFW window";
    }

    glfwMakeContextCurrent(window);
    glfw_state app_state = {}; // Create a GLFW state object

    try {
        while (!glfwWindowShouldClose(window)) {
            // Wait for a new set of frames
            auto frames = pipe.wait_for_frames();
            auto color_frame = frames.get_color_frame();
            auto depth_frame = frames.get_depth_frame();

            // Convert RealSense color frame to OpenCV Mat
            cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3,
                                (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

            // Ensure the image is valid
            if (color_image.empty()) {
                throw std::runtime_error("Captured frame is empty.");
            }
            cv::Mat color_image_copy = color_image.clone();
            // Convert RGB to BGR for proper color representation
            cv::cvtColor(color_image_copy, color_image_copy, cv::COLOR_RGB2BGR);

            // Detect bounding box for the orange object
            Eigen::Vector2d min_bound_2d, max_bound_2d;
            bool bbox_detected = false;

            try {
                auto [min_bound, max_bound] = detect_color_bbox(color_image_copy, lower_orange, upper_orange);
                min_bound_2d = min_bound;
                max_bound_2d = max_bound;
                bbox_detected = true;

                // Draw the bounding box on the color image
                cv::rectangle(color_image_copy,
                              cv::Point(static_cast<int>(min_bound.x()), static_cast<int>(min_bound.y())),
                              cv::Point(static_cast<int>(max_bound.x()), static_cast<int>(max_bound.y())),
                              cv::Scalar(0, 255, 0), 2);
            } catch (const std::exception&) {
                // If no object is detected, skip the bounding box creation
            }

            // Declare pointcloud object, for calculating pointclouds and texture mappings
            rs2::pointcloud pc;
            rs2::points points;
            // Ensure pointcloud is mapped to the color frame
            pc.map_to(color_frame);

            points = pc.calculate(depth_frame);

            app_state.tex.upload(color_frame);


            // Clear the OpenGL window
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Draw the point cloud
            draw_pointcloud(800, 600, app_state, points);

            // Display the result
            cv::imshow("Bounding Box Detection", color_image_copy);

            // Swap buffers and poll events
            glfwSwapBuffers(window);
            glfwPollEvents();

            // Break the loop if 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        FAIL() << "Error during RealSense stream, bounding box detection, or point cloud creation: " << e.what();
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
}
