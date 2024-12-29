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

// Function to isolate the colored point cloud and return PCL PCD
inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolate_colored_pointcloud(
    float width, float height, glfw_state& app_state, rs2::points& points, 
    const rs2::video_frame& color_frame, unsigned char target_blue, 
    unsigned char target_green, unsigned char target_red)
{
    if (!points)
        return nullptr;

    // Create a PCL point cloud to store the isolated points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd(new pcl::PointCloud<pcl::PointXYZRGB>());

    // OpenGL commands that prep screen for the point cloud
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glPointSize(width / 640);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());
    float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE

    glBegin(GL_POINTS);

    // Render and collect points with the specified color
    auto vertices = points.get_vertices();              // Get vertices
    auto tex_coords = points.get_texture_coordinates(); // Get texture coordinates
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
                    glVertex3fv(vertices[i]);  // Render the point
                    glTexCoord2f(tex_coords[i].u, tex_coords[i].v); // Upload texture coordinate

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

    glEnd();

    // OpenGL cleanup
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();

    // Set the point cloud properties
    isolated_pcd->width = isolated_pcd->points.size();
    isolated_pcd->height = 1;
    isolated_pcd->is_dense = false;

    return isolated_pcd;
}


// Test case
TEST(SegmentaionOfPCD_OpenGL, RealSenseStreamWithCADOverlay) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd;
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


            // Get color frame dimensions and data pointer
            int width = color_frame.get_width();
            int height = color_frame.get_height();
            int stride = color_frame.get_stride_in_bytes();
            unsigned char* data = (unsigned char*)color_frame.get_data();
            // Extract bounding box coordinates
            int x_min = static_cast<int>(min_bound_2d.x());
            int y_min = static_cast<int>(min_bound_2d.y());
            int x_max = static_cast<int>(max_bound_2d.x());
            int y_max = static_cast<int>(max_bound_2d.y());

            // Ensure the coordinates are within the frame bounds
            x_min = std::max(0, x_min);
            y_min = std::max(0, y_min);
            x_max = std::min(width - 1, x_max);
            y_max = std::min(height - 1, y_max);

            // Modify pixel values in the bounding box to blue
            for (int y = y_min; y <= y_max; ++y) {
                for (int x = x_min; x <= x_max; ++x) {
                    int index = y * stride + x * 3; // Assuming 3 bytes per pixel (RGB)
                    data[index] = 255; // Blue channel
                    data[index + 1] = 0; // Green channel
                    data[index + 2] = 0; // Red channel
                }
            }

            // Generate point cloud from depth frame
            rs2::pointcloud pc;
            rs2::points points;
            pc.map_to(color_frame);
            points = pc.calculate(depth_frame);

            // // Convert 2D bounding box to 3D bounding box
            // Eigen::Vector3f min_bound_3d, max_bound_3d;
            // std::tie(min_bound_3d, max_bound_3d) = calculate3DBoundingBox(min_bound_2d, max_bound_2d, depth_frame, intrinsics);

            // // Extract ROI and align CAD
            // auto roi_scene = highlightROI(points, min_bound_3d, max_bound_3d);
            // visualizePointCloud(roi_scene, 0.0, 0.0, 1.0); // Blue for ROI

            // auto aligned_cad = alignCADToScene(cad_object, roi_scene);
            // visualizePointCloud(aligned_cad, 1.0, 0.0, 0.0); // Red for CAD overlay

            // Render point clouds
            app_state.tex.upload(color_frame);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            unsigned char target_blue = 255; // Blue channel
            unsigned char target_green = 0;  // Green channel
            unsigned char target_red = 0;    // Red channel

            isolated_pcd  = isolate_colored_pointcloud(800, 600, app_state, points, color_frame, target_blue, target_green, target_red);
            // draw_colored_pointcloud(800, 600, app_state, points, color_frame, target_blue, target_green, target_red);

            glfwSwapBuffers(window);
            glfwPollEvents();   

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        if (isolated_pcd && !isolated_pcd->points.empty()) {
            pcl::io::savePCDFileASCII("../assets/isolated_pcd.pcd", *isolated_pcd);
        } else {
            std::cerr << "Isolated point cloud is empty. Nothing to save." << std::endl;
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
}
