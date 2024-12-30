#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <gtest/gtest.h>
#include "example.hpp"
#include "pcd_frm_depth.hpp"
#include "cad_pcd.hpp"

// Initialize GLFW
inline GLFWwindow* initializeGLFW() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Point Cloud Viewer with CAD Overlay", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    return window;
}

// Generate CAD Point Cloud
inline pcl::PointCloud<pcl::PointXYZ>::Ptr generateCADPCD(const std::string& cad_file) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cad_object(new pcl::PointCloud<pcl::PointXYZ>());
    auto cad_pcd = cad::get_cad_pcd(cad_file, 100000); // Generate CAD PCD
    for (const auto& point : cad_pcd->points_) {
        cad_object->points.emplace_back(point(0), point(1), point(2));
    }
    return cad_object;
}

// Visualize a Point Cloud
inline void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    glBegin(GL_POINTS);
    for (const auto& point : cloud->points) {
        glColor3f(point.r / 255.0, point.g / 255.0, point.b / 255.0); // Use RGB colors
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

// Isolate Colored Point Cloud
inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolate_colored_pointcloud(
    float width, float height, glfw_state& app_state, rs2::points& points,
    const rs2::video_frame& color_frame, unsigned char target_blue,
    unsigned char target_green, unsigned char target_red) {
    
    if (!points) return nullptr;

    auto vertices = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();
    const unsigned char* color_data = static_cast<const unsigned char*>(color_frame.get_data());
    int stride = color_frame.get_stride_in_bytes();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i = 0; i < points.size(); i++) {
        if (vertices[i].z) {
            int x = static_cast<int>(tex_coords[i].u * color_frame.get_width());
            int y = static_cast<int>(tex_coords[i].v * color_frame.get_height());
            if (x >= 0 && y >= 0 && x < color_frame.get_width() && y < color_frame.get_height()) {
                int index = y * stride + x * 3; // BGR format
                unsigned char blue = color_data[index];
                unsigned char green = color_data[index + 1];
                unsigned char red = color_data[index + 2];
                if (blue == target_blue && green == target_green && red == target_red) {
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
    isolated_pcd->width = isolated_pcd->points.size();
    isolated_pcd->height = 1;
    isolated_pcd->is_dense = true;

    return isolated_pcd;
}

// Convert XYZRGB to XYZ
inline pcl::PointCloud<pcl::PointXYZ>::Ptr convertToXYZ(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_rgb) {
    if (!cloud_rgb || cloud_rgb->empty()) {
        throw std::runtime_error("Input XYZRGB cloud is empty!");
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : cloud_rgb->points) {
        pcl::PointXYZ xyz_point;
        xyz_point.x = point.x;
        xyz_point.y = point.y;
        xyz_point.z = point.z;
        cloud_xyz->points.push_back(xyz_point);
    }
    cloud_xyz->width = cloud_rgb->width;
    cloud_xyz->height = cloud_rgb->height;
    cloud_xyz->is_dense = cloud_rgb->is_dense;
    return cloud_xyz;
}

// Test case
TEST(NewSegmentationOfPCD_OpenGL, RealSenseStreamWithCADOverlay) {
    try {
        rs2::pipeline pipe;
        auto profile = pipe.start();
        auto depth_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = depth_profile.get_intrinsics();

        GLFWwindow* window = initializeGLFW();
        glfw_state app_state = {};

        std::string cad_file = "../assets/VN_1400.obj";
        auto cad_object = generateCADPCD(cad_file);

        while (!glfwWindowShouldClose(window)) {
            auto frames = pipe.wait_for_frames();
            auto color_frame = frames.get_color_frame();
            auto depth_frame = frames.get_depth_frame();

            rs2::pointcloud pc;
            rs2::points points = pc.calculate(depth_frame);
            pc.map_to(color_frame);

            unsigned char target_blue = 255, target_green = 0, target_red = 0;
            auto isolated_pcd = isolate_colored_pointcloud(800, 600, app_state, points, color_frame, target_blue, target_green, target_red);
            if (!isolated_pcd || isolated_pcd->empty()) {
                std::cerr << "Isolated point cloud is empty!" << std::endl;
                continue; // Skip processing if the point cloud is empty
            }

            auto isolated_xyz = convertToXYZ(isolated_pcd);

            pcl::PointCloud<pcl::PointXYZ>::Ptr object_pcd_filtered(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(isolated_xyz);
            sor.setMeanK(20);
            sor.setStddevMulThresh(2.0);
            sor.filter(*object_pcd_filtered);

            if (object_pcd_filtered->empty()) {
                std::cerr << "Filtered point cloud is empty!" << std::endl;
                continue;
            }

            visualizePointCloud(isolated_pcd);
            glfwSwapBuffers(window);
            glfwPollEvents();

            if (cv::waitKey(1) == 'q') break;
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
}
