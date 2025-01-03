// PclUtils.cpp
#include "PclUtils.hpp"
#include <iostream>
#include <limits>

namespace PclUtils {

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64Encode(const std::vector<uchar>& data)
{
    std::string encoded;
    int val = 0, valb = -6;
    for (uchar c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6)
        encoded.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (encoded.size() % 4)
        encoded.push_back('=');
    return encoded;
}

std::vector<uchar> base64Decode(const std::string &encoded)
{
    std::vector<uchar> decoded;
    decoded.reserve(encoded.size() * 3 / 4);

    int val = 0;
    int valb = -8;

    for (unsigned char c : encoded) {
        if (c == '=') break;
        int pos = base64_chars.find(c);
        if (pos == (int)std::string::npos) {
            continue;
        }
        val = (val << 6) + pos;
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(static_cast<uchar>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }

    return decoded;
}

bool decodeFramesFromJson(const nlohmann::json &frame_json,
                          cv::Mat &color_image,
                          cv::Mat &depth_image)
{
    // Extract base64-encoded data
    std::string encoded_color = frame_json.value("color_encoded", "");
    int color_width = frame_json.value("color_width", 0);
    int color_height = frame_json.value("color_height", 0);

    std::string encoded_depth = frame_json.value("depth_encoded", "");
    int depth_width = frame_json.value("depth_width", 0);
    int depth_height = frame_json.value("depth_height", 0);

    if (encoded_color.empty() || encoded_depth.empty())
        return false;

    // Decode color
    std::vector<uchar> color_data = base64Decode(encoded_color);
    color_image = cv::imdecode(color_data, cv::IMREAD_COLOR);
    if (color_image.empty()) {
        std::cerr << "[PclUtils] Failed to decode color image.\n";
        return false;
    }
    if (color_image.cols != color_width || color_image.rows != color_height) {
        std::cerr << "[PclUtils] Warning: Color image size mismatch.\n";
    }

    // Decode depth
    std::vector<uchar> depth_data = base64Decode(encoded_depth);
    if (depth_data.size() != (size_t)depth_width * depth_height * 2) {
        std::cerr << "[PclUtils] Depth size mismatch or decoding error.\n";
        return false;
    }
    depth_image = cv::Mat(depth_height, depth_width, CV_16UC1, depth_data.data()).clone();
    if (depth_image.empty()) {
        std::cerr << "[PclUtils] Failed to decode depth image.\n";
        return false;
    }

    return true;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr createPointCloudFromDepth(
    const cv::Mat& depth_image,
    const cv::Mat& color_image,
    const nlohmann::json& intrinsics_json,
    float depth_scale)
{
    // Parse intrinsics
    int width  = intrinsics_json["width"];
    int height = intrinsics_json["height"];
    float ppx  = intrinsics_json["ppx"];
    float ppy  = intrinsics_json["ppy"];
    float fx   = intrinsics_json["fx"];
    float fy   = intrinsics_json["fy"];

    auto cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZRGB>);

    cloud->width    = static_cast<uint32_t>(width);
    cloud->height   = static_cast<uint32_t>(height);
    cloud->is_dense = false;
    cloud->points.resize(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            pcl::PointXYZRGB& pt = cloud->points[idx];

            uint16_t depth_val = depth_image.at<uint16_t>(y, x);
            if (depth_val == 0) {
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
                pt.r = pt.g = pt.b = 0;
                continue;
            }
            float z = depth_val * depth_scale;
            pt.z = z;
            pt.x = (static_cast<float>(x) - ppx) * z / fx;
            pt.y = (static_cast<float>(y) - ppy) * z / fy;

            // color_image is [rows, cols], BGR
            cv::Vec3b rgb = color_image.at<cv::Vec3b>(y, x);
            pt.b = rgb[0];
            pt.g = rgb[1];
            pt.r = rgb[2];
        }
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolateColoredPointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud,
    unsigned char target_blue,
    unsigned char target_green,
    unsigned char target_red)
{
    auto isolated = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZRGB>);

    for (auto &pt : input_cloud->points) {
        if (std::isfinite(pt.z) && 
            pt.b == target_blue &&
            pt.g == target_green &&
            pt.r == target_red)
        {
            isolated->points.push_back(pt);
        }
    }
    isolated->width  = static_cast<uint32_t>(isolated->points.size());
    isolated->height = 1;
    isolated->is_dense = false;
    return isolated;
}

} // namespace PclUtils
