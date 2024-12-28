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

// Function to visualize bounding box in 3D
void drawBoundingBox(const Eigen::Vector2d& min_2d, const Eigen::Vector2d& max_2d, float depth_min, float depth_max) {
    glColor3f(0.0, 1.0, 0.0); // Green for bounding box
    glBegin(GL_LINES);

    // Bottom face
    glVertex3f(min_2d.x(), min_2d.y(), depth_min);
    glVertex3f(max_2d.x(), min_2d.y(), depth_min);

    glVertex3f(max_2d.x(), min_2d.y(), depth_min);
    glVertex3f(max_2d.x(), max_2d.y(), depth_min);

    glVertex3f(max_2d.x(), max_2d.y(), depth_min);
    glVertex3f(min_2d.x(), max_2d.y(), depth_min);

    glVertex3f(min_2d.x(), max_2d.y(), depth_min);
    glVertex3f(min_2d.x(), min_2d.y(), depth_min);

    // Top face
    glVertex3f(min_2d.x(), min_2d.y(), depth_max);
    glVertex3f(max_2d.x(), min_2d.y(), depth_max);

    glVertex3f(max_2d.x(), min_2d.y(), depth_max);
    glVertex3f(max_2d.x(), max_2d.y(), depth_max);

    glVertex3f(max_2d.x(), max_2d.y(), depth_max);
    glVertex3f(min_2d.x(), max_2d.y(), depth_max);

    glVertex3f(min_2d.x(), max_2d.y(), depth_max);
    glVertex3f(min_2d.x(), min_2d.y(), depth_max);

    // Vertical edges
    glVertex3f(min_2d.x(), min_2d.y(), depth_min);
    glVertex3f(min_2d.x(), min_2d.y(), depth_max);

    glVertex3f(max_2d.x(), min_2d.y(), depth_min);
    glVertex3f(max_2d.x(), min_2d.y(), depth_max);

    glVertex3f(max_2d.x(), max_2d.y(), depth_min);
    glVertex3f(max_2d.x(), max_2d.y(), depth_max);

    glVertex3f(min_2d.x(), max_2d.y(), depth_min);
    glVertex3f(min_2d.x(), max_2d.y(), depth_max);

    glEnd();
}

// Updated function to visualize a colored point cloud
void visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    glBegin(GL_POINTS);
    for (const auto& point : cloud->points) {
        glColor3f(point.r / 255.0, point.g / 255.0, point.b / 255.0); // Use RGB colors
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

// Function to highlight ROI in the scene point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr highlightROI(
    const rs2::points& points, const Eigen::Vector3f& min_bound_3d, const Eigen::Vector3f& max_bound_3d) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_pcl(new pcl::PointCloud<pcl::PointXYZRGB>());
    auto vertices = points.get_vertices();

    for (int i = 0; i < points.size(); ++i) {
        auto& v = vertices[i];
        if (v.z) {
            pcl::PointXYZRGB point;
            point.x = v.x;
            point.y = v.y;
            point.z = v.z;

            // Check if the point lies within the ROI bounds
            if (v.x >= min_bound_3d.x() && v.x <= max_bound_3d.x() &&
                v.y >= min_bound_3d.y() && v.y <= max_bound_3d.y() &&
                v.z >= min_bound_3d.z() && v.z <= max_bound_3d.z()) {
                // Highlight ROI points with red color
                point.r = 255;
                point.g = 0;
                point.b = 0;
            } else {
                // Default color for non-ROI points (white)
                point.r = 255;
                point.g = 255;
                point.b = 255;
            }
            scene_pcl->push_back(point);
        }
    }

    return scene_pcl;
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


// Function to convert 2D bounding box to 3D bounding box
std::tuple<Eigen::Vector3f, Eigen::Vector3f> calculate3DBoundingBox(
    const Eigen::Vector2d& min_bound_2d, const Eigen::Vector2d& max_bound_2d,
    const rs2::depth_frame& depth_frame, const rs2_intrinsics& intrinsics) {
    auto projectTo3D = [&depth_frame, &intrinsics](float x, float y) -> Eigen::Vector3f {
        float depth = depth_frame.get_distance(static_cast<int>(x), static_cast<int>(y));
        float point_x = (x - intrinsics.ppx) / intrinsics.fx * depth;
        float point_y = (y - intrinsics.ppy) / intrinsics.fy * depth;
        float point_z = depth;
        return Eigen::Vector3f(point_x, point_y, point_z);
    };

    Eigen::Vector3f min_3d = projectTo3D(min_bound_2d.x(), min_bound_2d.y());
    Eigen::Vector3f max_3d = projectTo3D(max_bound_2d.x(), max_bound_2d.y());
    return std::make_tuple(min_3d, max_3d);
}


// Test case
TEST(OverlayCADOntoScene_OpenGL, RealSenseStreamWithCADOverlay) {
    try {
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        auto profile = pipe.start();

        // Get depth stream intrinsics
        auto depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = depth_stream_profile.get_intrinsics();

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

            if (!points || points.size() == 0) {
                std::cerr << "Generated point cloud is empty!" << std::endl;
                continue;
            }

            // Convert 2D bounding box to 3D bounding box
            Eigen::Vector3f min_bound_3d, max_bound_3d;
            std::tie(min_bound_3d, max_bound_3d) = calculate3DBoundingBox(min_bound_2d, max_bound_2d, depth_frame, intrinsics);

            // Highlight ROI in the point cloud
            auto highlighted_scene = highlightROI(points, min_bound_3d, max_bound_3d);

            // Visualize the full scene with highlighted ROI
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            visualizePointCloud(highlighted_scene);

            // Print 2D bounding box values
            std::cout << "2D Bounding Box: Min(" << min_bound_2d.x() << ", " << min_bound_2d.y() << "), "
                    << "Max(" << max_bound_2d.x() << ", " << max_bound_2d.y() << ")" << std::endl;

            // Print 3D bounding box values
            std::cout << "3D Bounding Box: Min(" << min_bound_3d.x() << ", " << min_bound_3d.y() << ", " << min_bound_3d.z() << "), "
                    << "Max(" << max_bound_3d.x() << ", " << max_bound_3d.y() << ", " << max_bound_3d.z() << ")" << std::endl;


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