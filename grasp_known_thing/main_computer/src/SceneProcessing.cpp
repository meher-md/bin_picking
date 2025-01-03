// SceneProcessing.cpp
#include "SceneProcessing.hpp"
#include <open3d/Open3D.h>
#include "PclUtils.hpp"  // for createPointCloudFromDepth, base64, etc.
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace SceneProcessing {

std::shared_ptr<open3d::geometry::PointCloud> generateCadPointCloud(
    const std::string& cad_file_path,
    int number_of_points)
{
    // Create mesh from .obj
    auto mesh = open3d::io::CreateMeshFromFile(cad_file_path);
    if (!mesh) {
        throw std::runtime_error("Failed to load CAD mesh from " + cad_file_path);
    }

    // Sample uniformly
    auto pcd = mesh->SamplePointsUniformly(number_of_points);
    if (pcd->points_.empty()) {
        throw std::runtime_error("Generated point cloud is empty.");
    }

    // Downsample
    auto downsampled_pcd = pcd->VoxelDownSample(0.001);
    if (downsampled_pcd->points_.empty()) {
        throw std::runtime_error("Downsampled CAD point cloud is empty.");
    }

    return downsampled_pcd;
}



void runPointCloudProcessing(SensorReceiver &sensorReceiver,
                             const std::string &pcd_output_path)
{
    // We'll store the final isolated cloud here:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd;

    // We assume a loop that might break once we get a "good" frame 
    // or you can customize to run once, or multiple frames, etc.
    for (int iteration = 0; iteration < 10; ++iteration) {
        cv::Mat color_image, depth_image;
        nlohmann::json bbox_json;
        bool ok = sensorReceiver.requestFrame(color_image, depth_image, bbox_json);
        if (!ok) {
            std::cerr << "[SceneProcessing] Failed to get valid frame/bbox from servers.\n";
            continue;
        }

        // Make a copy of the color image to draw bounding boxes
        cv::Mat color_image_copy = color_image.clone();

        // 1. Draw bounding box and fill with blue
        if (bbox_json.contains("x_min")) {
            int x_min = bbox_json["x_min"];
            int y_min = bbox_json["y_min"];
            int x_max = bbox_json["x_max"];
            int y_max = bbox_json["y_max"];

            // Clamp coordinates to image boundaries
            x_min = std::max(0, std::min(x_min, color_image_copy.cols - 1));
            x_max = std::max(0, std::min(x_max, color_image_copy.cols - 1));
            y_min = std::max(0, std::min(y_min, color_image_copy.rows - 1));
            y_max = std::max(0, std::min(y_max, color_image_copy.rows - 1));

            // Draw bounding box rectangle
            cv::rectangle(color_image_copy,
                          cv::Point(x_min, y_min),
                          cv::Point(x_max, y_max),
                          cv::Scalar(0, 255, 0), 2);

            // Fill bounding box region with blue color
            for (int row = y_min; row <= y_max; ++row) {
                for (int col = x_min; col <= x_max; ++col) {
                    color_image_copy.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 0, 0);
                }
            }
        }
        else if (bbox_json.contains("error")) {
            std::cerr << "[SceneProcessing] Error from YOLO server: "
                      << bbox_json["error"] << std::endl;
            continue;
        }
        else {
            std::cerr << "[SceneProcessing] Bounding box not found in YOLO reply.\n";
            continue;
        }

        // 2. Create a 3D cloud from the modified color image and depth image
        auto full_cloud = PclUtils::createPointCloudFromDepth(
            depth_image, 
            color_image_copy,  // Use the modified color image
            sensorReceiver.getIntrinsicsJson(),
            0.001f // depth_scale
        );

        // 3. Isolate points that have been colored blue within the bounding box
        unsigned char target_blue  = 255;
        unsigned char target_green = 0;
        unsigned char target_red   = 0;
        isolated_pcd = PclUtils::isolateColoredPointCloud(
            full_cloud,
            target_blue,
            target_green,
            target_red
        );

        if (isolated_pcd->points.empty()) {
            std::cerr << "[SceneProcessing] Isolated cloud is empty in iteration " 
                      << iteration << ".\n";
            continue;
        }

        // 4. Save the isolated point cloud and exit the loop
        pcl::io::savePCDFileASCII(pcd_output_path, *isolated_pcd);
        std::cout << "[SceneProcessing] Saved " << isolated_pcd->points.size()
                  << " points to " << pcd_output_path << std::endl;
        break; // Exit after successful isolation and saving
    }

    // If no valid isolated cloud was found after all iterations
    if (!isolated_pcd || isolated_pcd->points.empty()) {
        std::cerr << "[SceneProcessing] Could not isolate any valid object cloud.\n";
    }
}

} // namespace SceneProcessing
