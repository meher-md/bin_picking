#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <open3d/Open3D.h>
#include "pcd_frm_depth.hpp"
#include "detect_color_bbox.hpp"

TEST(RealTimeBoundingBoxAndPointCloud, RealSenseStream) {
    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    auto profile = pipe.start();

    // Get camera intrinsics
    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

    // Define the HSV range for the orange color
    cv::Scalar lower_orange(0, 100, 100);
    cv::Scalar upper_orange(25, 255, 255);

    // Create Open3D Visualizer
    auto visualizer = std::make_shared<open3d::visualization::Visualizer>();
    visualizer->CreateVisualizerWindow("Point Cloud Viewer", 800, 600);

    // Add a placeholder geometry (empty point cloud)
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
    visualizer->AddGeometry(point_cloud);

    try {
        while (true) {
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

            // Convert RGB to BGR for proper color representation
            cv::cvtColor(color_image, color_image, cv::COLOR_RGB2BGR);

            // Detect bounding box for the orange object
            Eigen::Vector2d min_bound_2d, max_bound_2d;
            bool bbox_detected = false;

            try {
                auto [min_bound, max_bound] = detect_color_bbox(color_image, lower_orange, upper_orange);
                min_bound_2d = min_bound;
                max_bound_2d = max_bound;
                bbox_detected = true;

                std::cout << "Bounding Box Min Bound: (" << min_bound.x() << ", " << min_bound.y() << ")\n";
                std::cout << "Bounding Box Max Bound: (" << max_bound.x() << ", " << max_bound.y() << ")\n";

                // Draw the bounding box on the color image
                cv::rectangle(color_image,
                              cv::Point(static_cast<int>(min_bound.x()), static_cast<int>(min_bound.y())),
                              cv::Point(static_cast<int>(max_bound.x()), static_cast<int>(max_bound.y())),
                              cv::Scalar(0, 255, 0), 2);
            } catch (const std::exception&) {
                // If no object is detected, skip the bounding box creation
            }

            // If a bounding box is detected, generate the point cloud
            if (bbox_detected) {
                auto new_point_cloud = pcd_frm_depth(depth_frame, min_bound_2d, max_bound_2d, intrinsics);

                // Remove the previous geometry and add the new one
                visualizer->RemoveGeometry(point_cloud);
                point_cloud = new_point_cloud;  // Update the point cloud reference
                visualizer->AddGeometry(point_cloud);

                // Update the visualizer
                visualizer->PollEvents();
                visualizer->UpdateRender();
            }

            // Display the result
            cv::imshow("Bounding Box Detection", color_image);

            // Break the loop if 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        FAIL() << "Error during RealSense stream, bounding box detection, or point cloud creation: " << e.what();
    }

    // Destroy the Open3D Visualizer window when the test is done
    visualizer->DestroyVisualizerWindow();
}
