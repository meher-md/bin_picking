// src/pcd_saving.cpp

#include "pcd_saving.hpp"
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "detect_color_bbox.hpp" // Ensure this header is correctly implemented
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <thread>
#include <algorithm>

// Structure to hold application state if needed (placeholder)
struct AppState {
    // Add members if necessary
};

// Function to isolate the colored point cloud and return PCL PCD
pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolate_colored_pointcloud(
    const rs2::points& points, 
    const rs2::video_frame& color_frame, 
    unsigned char target_blue, 
    unsigned char target_green, 
    unsigned char target_red)
{
    if (!points)
        return nullptr;

    // Create a PCL point cloud to store the isolated points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd(new pcl::PointCloud<pcl::PointXYZRGB>());
    auto vertices = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();
    const unsigned char* color_data = static_cast<const unsigned char*>(color_frame.get_data());
    int stride = color_frame.get_stride_in_bytes();

    for (int i = 0; i < points.size(); i++)
    {
        if (vertices[i].z) // Only consider valid depth points
        {
            // Map texture coordinates to image coordinates
            int x = static_cast<int>(tex_coords[i].u * color_frame.get_width());
            int y = static_cast<int>(tex_coords[i].v * color_frame.get_height());

            if (x >= 0 && y >= 0 && x < color_frame.get_width() && y < color_frame.get_height())
            {
                // Get the color at the mapped texture coordinates
                int index = y * stride + x * 3; // 3 channels (BGR)
                unsigned char blue = color_data[index];
                unsigned char green = color_data[index + 1];
                unsigned char red = color_data[index + 2];

                // Check if the color matches the target color
                if (blue == target_blue && green == target_green && red == target_red)
                {
                    // Add the point to the PCL point cloud
                    pcl::PointXYZRGB pcl_point;
                    pcl_point.x = vertices[i].x;
                    pcl_point.y = vertices[i].y;
                    pcl_point.z = vertices[i].z;
                    pcl_point.r = red;
                    pcl_point.g = green;
                    pcl_point.b = blue;
                    isolated_pcd->points.push_back(pcl_point);
                }
            }
        }
    }

    // Set the point cloud properties
    isolated_pcd->width = isolated_pcd->points.size();
    isolated_pcd->height = 1;
    isolated_pcd->is_dense = false;

    return isolated_pcd;
}

void GenerateAndSaveObjectPCD(const std::string& save_pcd_path) {
    try {
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);

        // Allow the camera to stabilize
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Capture frames
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        rs2::video_frame color_frame = frames.get_color_frame();

        // Convert color frame to OpenCV Mat (if needed for other processing)
        cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3,
                            (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        if (color_image.empty()) {
            throw std::runtime_error("Captured color frame is empty.");
        }

        // Define bounding box color range (e.g., orange in HSV)
        // Convert BGR to HSV for better color segmentation
        cv::Mat hsv_image;
        cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);

        // Define lower and upper bounds for the target color (orange)
        cv::Scalar lower_orange(10, 100, 100); // Adjust HSV values as needed
        cv::Scalar upper_orange(25, 255, 255);

        // Detect bounding box based on color
        Eigen::Vector2d min_bound_2d, max_bound_2d;
        std::tie(min_bound_2d, max_bound_2d) = detect_color_bbox(hsv_image, lower_orange, upper_orange);

        // Handle cases where no bounding box is detected
        if (min_bound_2d.x() == 0 && min_bound_2d.y() == 0 && 
            max_bound_2d.x() == 0 && max_bound_2d.y() == 0) {
            throw std::runtime_error("No bounding box detected based on the target color.");
        }

        // Extract bounding box coordinates
        int x_min = static_cast<int>(min_bound_2d.x());
        int y_min = static_cast<int>(min_bound_2d.y());
        int x_max = static_cast<int>(max_bound_2d.x());
        int y_max = static_cast<int>(max_bound_2d.y());

        // Ensure the coordinates are within the frame bounds
        x_min = std::max(0, x_min);
        y_min = std::max(0, y_min);
        x_max = std::min(static_cast<int>(color_frame.get_width()) - 1, x_max);
        y_max = std::min(static_cast<int>(color_frame.get_height()) - 1, y_max);

        // Optional: Modify pixel values in the bounding box to a distinct color
        // (Based on your segmentation approach, this might not be necessary)
        /*
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                color_image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); // Blue channel
            }
        }
        */

        // Generate point cloud from depth frame
        rs2::pointcloud pc;
        rs2::points points = pc.calculate(depth_frame);
        pc.map_to(color_frame);

        // Define target color for segmentation (blue)
        unsigned char target_blue = 255; // Blue channel
        unsigned char target_green = 0;  // Green channel
        unsigned char target_red = 0;    // Red channel

        // Isolate ROI based on color
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd = isolate_colored_pointcloud(
            points, color_frame, target_blue, target_green, target_red
        );

        // Save the isolated point cloud to a PCD file
        if (isolated_pcd && !isolated_pcd->points.empty()) {
            if (pcl::io::savePCDFileASCII(save_pcd_path, *isolated_pcd) == -1) {
                throw std::runtime_error("Failed to save the isolated point cloud to PCD file.");
            }
            std::cout << "Successfully saved isolated point cloud with " 
                      << isolated_pcd->points.size() << " points to " 
                      << save_pcd_path << "\n";
        } else {
            throw std::runtime_error("Isolated point cloud is empty. Nothing to save.");
        }

        // Stop the RealSense pipeline
        pipe.stop();
  } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }

}
