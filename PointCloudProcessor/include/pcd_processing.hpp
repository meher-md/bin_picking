// pcd_processing.hpp

#ifndef POINTCLOUD_MODULE_HPP
#define POINTCLOUD_MODULE_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Dense>
#include <string>

// Include custom headers
#include "example.hpp"             // Should define the Texture class
#include "detect_color_bbox.hpp"   // Should declare detect_color_bbox function

// // Structure to hold GLFW and application state
// struct glfw_state {
//     Texture tex;      // Texture object for uploading color frames
//     float offset_y;   // Offset for Y-axis translation
//     double pitch;     // Rotation angle around X-axis
//     double yaw;       // Rotation angle around Y-axis

//     glfw_state() : offset_y(0.0f), pitch(0.0), yaw(0.0) {}
// };

// Initializes GLFW and creates a window
// Throws std::runtime_error on failure
GLFWwindow* initializeGLFW();

// Isolates points in the point cloud that match the target RGB color
// Returns a pointer to the isolated PCL point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolate_colored_pointcloud(
    float width, 
    float height, 
    glfw_state& app_state, 
    rs2::points& points, 
    const rs2::video_frame& color_frame, 
    unsigned char target_blue, 
    unsigned char target_green, 
    unsigned char target_red
);

// Runs the entire point cloud processing pipeline and saves the isolated PCD file
// `pcd_file_path` specifies where to save the PCD file
void run_pointcloud_processing(const std::string& pcd_file_path);

#endif // POINTCLOUD_MODULE_HPP
