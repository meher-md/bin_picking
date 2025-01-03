// PclUtils.hpp
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

namespace PclUtils {

    // Base64
    std::string base64Encode(const std::vector<uchar>& data);
    std::vector<uchar> base64Decode(const std::string &encoded);

    // Decode color & depth from JSON into cv::Mat
    bool decodeFramesFromJson(const nlohmann::json &frame_json,
                              cv::Mat &color_image,
                              cv::Mat &depth_image);

    // Create a PCL XYZRGB cloud from color+depth
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr createPointCloudFromDepth(
        const cv::Mat& depth_image,
        const cv::Mat& color_image,
        const nlohmann::json& intrinsics_json,
        float depth_scale
    );

    // Example: isolate points with a specific color
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolateColoredPointCloud(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud,
        unsigned char target_blue,
        unsigned char target_green,
        unsigned char target_red
    );

} // namespace PclUtils
