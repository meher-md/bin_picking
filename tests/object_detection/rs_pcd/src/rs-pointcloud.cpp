// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#include <cad_pcd.hpp>
#include "object_detection.hpp"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <algorithm>            // std::min, std::max
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>


// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

std::string get_cad_path_from_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + config_file);
    }

    nlohmann::json config;
    file >> config;
    return config["cad_file_path"].get<std::string>();
}


int main(int argc, char * argv[]) try
{
    // Get CAD file path from config
    std::string config_file = "../config/paths.json";
    std::string cad_file_path = get_cad_path_from_config(config_file);
    auto cad_pcd = cad::get_cad_pcd(cad_file_path);

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Pointcloud Example");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    while (app) // Application still alive?
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        auto color = frames.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frames.get_infrared_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);

        // Upload the color frame to OpenGL
        app_state.tex.upload(color);

        // Draw the pointcloud
        draw_pointcloud(app.width(), app.height(), app_state, points);

        // declare a o3d point cloud variable
        auto o3_pcd = std::make_shared<open3d::geometry::PointCloud>();
        // Get the vertices from the point cloud
        const rs2::vertex* vertices = points.get_vertices();
        // Loop through each vertex and populate Open3D's point cloud
        for (size_t i = 0; i < points.size(); ++i) {
            const auto& v = vertices[i];
            // Check for valid depth data (z > 0)
            if (v.z > 0) {
                o3_pcd->points_.emplace_back(v.x, v.y, v.z);
            }
        }

        // Check if the point cloud is empty and throw an error if so
        if (o3_pcd->points_.empty()) {
            throw std::runtime_error("Generated o3d point cloud is empty. Depth data may be invalid.");
        }

        auto detection_result = detect::detect_object(o3_pcd, cad_pcd);

        // Print transformation matrix
        std::cout << "Transformation Matrix:\n" << detection_result.transformation << std::endl;

    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
