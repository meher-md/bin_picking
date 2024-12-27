#include "detect_color_bbox.hpp"
#include "pcd_frm_depth.hpp"
#include "object_detection.hpp"
#include "cad_pcd.hpp"
#include "visualization.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <open3d/Open3D.h>
#include <librealsense2/rs.hpp>

// Function to get the CAD path from the config
std::string get_cad_path_from_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + config_file);
    }

    nlohmann::json config;
    file >> config;
    return config["cad_file_path"].get<std::string>();
}

int main() {
    rs2::pipeline pipe;
    auto profile = pipe.start();

    // Get camera intrinsics
    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

    std::string config_file = "../config/paths.json";
    std::string cad_file_path = get_cad_path_from_config(config_file);
    auto cad_pcd = cad::get_cad_pcd(cad_file_path);

    // Define HSV range for orange color
    cv::Scalar lower_orange(0, 100, 100);
    cv::Scalar upper_orange(25, 255, 255);

    while (true) {
        auto frames = pipe.wait_for_frames();
        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();

        // Convert RealSense color frame to OpenCV Mat
        cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, 
                            (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        try {
            // Detect bounding box of the orange object
            auto [min_bound_2d, max_bound_2d] = detect_color_bbox(color_image, lower_orange, upper_orange);

            // Generate a point cloud from the depth image within the bounding box
            auto captured_pcd = pcd_frm_depth(depth_frame, min_bound_2d, max_bound_2d, intrinsics);

            auto detection_result = detect::detect_object(captured_pcd, cad_pcd);

            // Print transformation matrix
            std::cout << "Transformation Matrix:\n" << detection_result.transformation << std::endl;

            // Visualize the filtered point cloud and CAD alignment
            visualize::show_combined_point_cloud(detection_result.aligned_pcd);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}
