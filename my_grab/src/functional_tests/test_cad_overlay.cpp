#include <GL/glew.h>    // GLEW must be included before any OpenGL headers
#include <GLFW/glfw3.h> // GLFW for window and context management
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "example.hpp"
#include "pcd_frm_depth.hpp"
#include "detect_color_bbox.hpp"
#include "cad_pcd.hpp"
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/icp.h>
#include <gtest/gtest.h>

// Function to initialize GLFW
GLFWwindow* initializeGLFW() {
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

// Function to generate CAD PCD
pcl::PointCloud<pcl::PointXYZ>::Ptr generateCADPCD(const std::string& cad_file) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cad_object(new pcl::PointCloud<pcl::PointXYZ>());
    try {
        auto cad_pcd = cad::get_cad_pcd(cad_file, 100000); // Generate CAD PCD
        for (const auto& point : cad_pcd->points_) {
            cad_object->points.emplace_back(point(0), point(1), point(2));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to generate CAD PCD: ") + e.what());
    }
    return cad_object;
}

// Function to extract ROI from the scene point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr extractROI(
    const rs2::points& points, const Eigen::Vector2d& min_bound, const Eigen::Vector2d& max_bound) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_pcl(new pcl::PointCloud<pcl::PointXYZ>());
    for (int i = 0; i < points.size(); ++i) {
        auto v = points.get_vertices()[i];
        if (v.z) scene_pcl->push_back(pcl::PointXYZ(v.x, v.y, v.z));
    }

    pcl::CropBox<pcl::PointXYZ> crop_box;
    crop_box.setInputCloud(scene_pcl);
    crop_box.setMin(Eigen::Vector4f(min_bound.x(), min_bound.y(), 0.0f, 1.0f));
    crop_box.setMax(Eigen::Vector4f(max_bound.x(), max_bound.y(), 1.0f, 1.0f));

    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_scene(new pcl::PointCloud<pcl::PointXYZ>());
    crop_box.filter(*roi_scene);
    return roi_scene;
}

// Function to perform ICP alignment
pcl::PointCloud<pcl::PointXYZ>::Ptr alignCADToScene(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cad_object, pcl::PointCloud<pcl::PointXYZ>::Ptr roi_scene) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cad(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cad_object);
    icp.setInputTarget(roi_scene);
    icp.align(*aligned_cad);

    if (!icp.hasConverged()) {
        throw std::runtime_error("ICP alignment failed!");
    }
    return aligned_cad;
}

// Test case
TEST(OverlayCADOntoScene_OpenGL, RealSenseStreamWithCADOverlay) {
    try {
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        auto profile = pipe.start();

        // Define bounding box color range
        cv::Scalar lower_orange(0, 100, 100);
        cv::Scalar upper_orange(25, 255, 255);

        // Generate CAD PCD
        std::string cad_file = "../assets/VN_1400.obj"; // Replace with your CAD file path
        auto cad_object = generateCADPCD(cad_file);

        // Initialize GLFW
        GLFWwindow* window = initializeGLFW();
        glfw_state app_state = {}; // Create a GLFW state object

        while (!glfwWindowShouldClose(window)) {
            // Capture frames from RealSense
            auto frames = pipe.wait_for_frames();
            auto color_frame = frames.get_color_frame();
            auto depth_frame = frames.get_depth_frame();

            // Convert color frame to OpenCV Mat
            cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3,
                                (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            if (color_image.empty()) {
                throw std::runtime_error("Captured frame is empty.");
            }

            cv::Mat color_image_copy = color_image.clone();
            cv::cvtColor(color_image_copy, color_image_copy, cv::COLOR_RGB2BGR);

            // Detect bounding box
            Eigen::Vector2d min_bound_2d, max_bound_2d;
            auto [min_bound, max_bound] = detect_color_bbox(color_image_copy, lower_orange, upper_orange);
            min_bound_2d = min_bound;
            max_bound_2d = max_bound;

            cv::rectangle(color_image_copy,
                          cv::Point(static_cast<int>(min_bound.x()), static_cast<int>(min_bound.y())),
                          cv::Point(static_cast<int>(max_bound.x()), static_cast<int>(max_bound.y())),
                          cv::Scalar(0, 255, 0), 2);

            // Generate point cloud from depth frame
            rs2::pointcloud pc;
            rs2::points points;
            pc.map_to(color_frame);
            points = pc.calculate(depth_frame);

            // Extract ROI and align CAD
            auto roi_scene = extractROI(points, min_bound_2d, max_bound_2d);
            auto aligned_cad = alignCADToScene(cad_object, roi_scene);

            // Render point clouds
            app_state.tex.upload(color_frame);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            draw_pointcloud(800, 600, app_state, points);

            // Overlay the aligned CAD point cloud
            for (const auto& point : aligned_cad->points) {
                glBegin(GL_POINTS);
                glVertex3f(point.x, point.y, point.z);
                glColor3f(1.0, 0.0, 0.0); // Red for CAD overlay
                glEnd();
            }

            cv::imshow("Bounding Box Detection", color_image_copy);
            glfwSwapBuffers(window);
            glfwPollEvents();

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
}
