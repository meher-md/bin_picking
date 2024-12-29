#include "cad_pcd.hpp"
#include <gtest/gtest.h>
#include <GL/glew.h>    // GLEW must be included before any OpenGL headers
#include <GLFW/glfw3.h> // GLFW for window and context management
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "example.hpp"
#include "pcd_frm_depth.hpp"
#include "detect_color_bbox.hpp"
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include <stdexcept>

// Declare a global pointer for the merged point cloud
std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> global_merged_pcd;

// Signal handler function
void signalHandler(int signum) {
    if (global_merged_pcd && !global_merged_pcd->points.empty()) {
        std::cout << "Saving merged point cloud before exiting..." << std::endl;
        pcl::io::savePCDFileASCII("../assets/merged_pcd.pcd", *global_merged_pcd);
        std::cout << "Merged point cloud saved to ../assets/merged_pcd.pcd" << std::endl;
    }
    std::cout << "Exiting gracefully..." << std::endl;
    exit(signum); // Exit the program
}

// Function to initialize GLFW
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

// Function to generate CAD PCD
inline pcl::PointCloud<pcl::PointXYZ>::Ptr generateCADPCD(const std::string& cad_file) {
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

// Function to convert rs2::points to PCL PointCloud<pcl::PointXYZRGB>
pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertRS2PointsToPCL(
    const rs2::points& points, 
    const rs2::video_frame& color_frame) 
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    auto vertices = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();
    const unsigned char* color_data = static_cast<const unsigned char*>(color_frame.get_data());
    int color_stride = color_frame.get_stride_in_bytes();

    for (int i = 0; i < points.size(); ++i) {
        if (vertices[i].z) { // Valid point
            pcl::PointXYZRGB point;
            point.x = vertices[i].x;
            point.y = vertices[i].y;
            point.z = vertices[i].z;

            // Map texture coordinates to image coordinates
            int x = static_cast<int>(tex_coords[i].u * color_frame.get_width());
            int y = static_cast<int>(tex_coords[i].v * color_frame.get_height());

            if (x >= 0 && y >= 0 && x < color_frame.get_width() && y < color_frame.get_height()) {
                int idx = y * color_stride + x * 3; // BGR format
                point.b = color_data[idx];
                point.g = color_data[idx + 1];
                point.r = color_data[idx + 2];
            } else {
                // Assign default color if out of bounds
                point.r = point.g = point.b = 255;
            }

            pcl_cloud->points.push_back(point);
        }
    }

    pcl_cloud->width = pcl_cloud->points.size();
    pcl_cloud->height = 1;
    pcl_cloud->is_dense = false;

    return pcl_cloud;
}

// Function to downsample a point cloud using VoxelGrid filter
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
    float leaf_size = 0.005f) 
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxel_filter.filter(*cloud_filtered);
    return cloud_filtered;
}

// Function to remove statistical outliers
pcl::PointCloud<pcl::PointXYZ>::Ptr removeOutliers(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
    int mean_k = 50, 
    double std_dev = 1.0) 
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_dev);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    sor.filter(*cloud_filtered);
    return cloud_filtered;
}

// Function to compute normals
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.setKSearch(30);
    ne.compute(*normals);
    return normals;
}

// Function to compute FPFH features
pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFHFeatures(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals) 
{
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    fpfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh.setRadiusSearch(0.05); // Adjust radius as needed
    fpfh.compute(*features);
    return features;
}

// Function to perform alignment using SampleConsensusPrerejective and ICP
Eigen::Matrix4f alignPointClouds(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target) 
{
    // Compute normals
    pcl::PointCloud<pcl::Normal>::Ptr source_normals = computeNormals(source);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals = computeNormals(target);

    // Compute FPFH features
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features = computeFPFHFeatures(source, source_normals);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features = computeFPFHFeatures(target, target_normals);

    // Initial Alignment using SampleConsensusPrerejective
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_prerejective;
    sac_prerejective.setInputSource(source);
    sac_prerejective.setInputTarget(target);
    sac_prerejective.setSourceFeatures(source_features);
    sac_prerejective.setTargetFeatures(target_features);
    sac_prerejective.setMaximumIterations(500); // Number of RANSAC iterations
    sac_prerejective.setNumberOfSamples(3);      // Number of points to sample for generating/prerejecting a pose
    sac_prerejective.setCorrespondenceRandomness(5); // Number of nearest features to use
    sac_prerejective.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
    sac_prerejective.setMaxCorrespondenceDistance(2.5f * 0.05f); // Inlier threshold
    sac_prerejective.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis

    pcl::PointCloud<pcl::PointXYZ> sac_prerejective_aligned;
    sac_prerejective.align(sac_prerejective_aligned);

    if (!sac_prerejective.hasConverged()) {
        throw std::runtime_error("SampleConsensusPrerejective alignment failed!");
    }

    Eigen::Matrix4f sac_transformation = sac_prerejective.getFinalTransformation();

    // Refine alignment using ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr sac_aligned_ptr(new pcl::PointCloud<pcl::PointXYZ>(sac_prerejective_aligned));
    icp.setInputSource(sac_aligned_ptr);
    icp.setInputTarget(target);
    icp.setMaximumIterations(100);
    icp.setMaxCorrespondenceDistance(0.05);
    pcl::PointCloud<pcl::PointXYZ> icp_aligned;
    icp.align(icp_aligned);

    if (!icp.hasConverged()) {
        throw std::runtime_error("ICP alignment failed!");
    }

    Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();

    // Combine transformations
    Eigen::Matrix4f final_transformation = icp_transformation * sac_transformation;

    return final_transformation;
}
// Function to visualize PCL PointCloud using OpenGL
void draw_pcl_pointcloud(
    int width, 
    int height, 
    glfw_state& app_state, 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) 
{
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (const auto& point : cloud->points) {
        glColor3ub(point.r, point.g, point.b);
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
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

TEST(CADPCDTest, LoadAndSaveCADFile) {
    std::string test_file = "../assets/VN_1400.obj"; // Replace with a valid STL file path
    ASSERT_NO_THROW({
        auto pcd = cad::get_cad_pcd(test_file, 100000);
        ASSERT_NE(pcd, nullptr);
        ASSERT_GT(pcd->points_.size(), 0);
    });

    // Verify the output PCD file
    std::string output_file = "../assets/generated_pcd.pcd";
    auto loaded_pcd = open3d::io::CreatePointCloudFromFile(output_file);
    ASSERT_NE(loaded_pcd, nullptr);
    ASSERT_GT(loaded_pcd->points_.size(), 0);
}

TEST(OverlayCADOntoScene_OpenGL, RealSenseStreamWithCADOverlay) {
    // In the TEST function
    global_merged_pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(); 
    try {
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        auto profile = pipe.start();

        // Get depth stream intrinsics
        auto depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = depth_stream_profile.get_intrinsics();

        // Define target color (blue in BGR format)
        unsigned char target_blue = 255;  // Blue channel
        unsigned char target_green = 0;   // Green channel
        unsigned char target_red = 0;     // Red channel
        // Define bounding box color range
        cv::Scalar lower_orange(0, 100, 100);
        cv::Scalar upper_orange(25, 255, 255);
        // Generate CAD PCD
        std::string cad_file = "../assets/VN_1400.obj"; // Replace with your CAD file path
        pcl::PointCloud<pcl::PointXYZ>::Ptr cad_object = generateCADPCD(cad_file);

        // Downsample CAD PCD for faster processing (optional but recommended)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cad_downsampled = downsamplePointCloud(cad_object, 0.005f);

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
            std::cout << "Generate point cloud from depth frame..." << std::endl;

            rs2::pointcloud pc;
            rs2::points points;
            pc.map_to(color_frame);
            points = pc.calculate(depth_frame);

            // Convert rs2::points to PCL PointCloud<pcl::PointXYZRGB>
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_pcl = convertRS2PointsToPCL(points, color_frame);

            std::cout << "Convert rs2::points to PCL Point..." << std::endl;

            // Use the modified draw_colored_pointcloud to isolate and draw the object
            // Render point clouds
            app_state.tex.upload(color_frame);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd = isolate_colored_pointcloud(
                800, 
                600, 
                app_state, 
                points, 
                color_frame, 
                target_blue, 
                target_green, 
                target_red
            );
            
            std::cout << "draw_colored_pointcloud to isolate..." << std::endl;
            if (!isolated_pcd || isolated_pcd->points.empty()) {
                std::cout << "No object detected..." << std::endl;
                glfwSwapBuffers(window);
                glfwPollEvents();
                continue;
            }
            // Convert isolated_pcd to pcl::PointCloud<pcl::PointXYZ>::Ptr for alignment
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_pcd(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& point : isolated_pcd->points) {
                pcl::PointXYZ p;
                p.x = point.x;
                p.y = point.y;
                p.z = point.z;
                object_pcd->points.push_back(p);
            }

            object_pcd->width = object_pcd->points.size();
            object_pcd->height = 1;
            object_pcd->is_dense = false;
            std::cout << "Convert isolated_pcd to pcl..." << std::endl;

            if (object_pcd->points.empty()) {
                // No valid object points after conversion, continue
                std::cout << "No valid object points after conversion..." << std::endl;
                glfwSwapBuffers(window);
                glfwPollEvents();
                continue;
            }

            // // Downsample and remove outliers from object PCD
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_pcd_filtered = downsamplePointCloud(object_pcd, 0.005f);
            object_pcd_filtered = removeOutliers(object_pcd_filtered, 50, 1.0);
            std::cout << "Downsample and remove outliers from object PCD..." << std::endl;

            // merge and save
            *global_merged_pcd = *object_pcd_filtered + *cad_downsampled;
            // Perform alignment between CAD PCD and isolated object PCD
            std::cout << "Perform alignment between CAD PCD and isolated object PCD..." << std::endl;

            Eigen::Matrix4f transformation;
            try {
                transformation = alignPointClouds(cad_downsampled, object_pcd_filtered);
            } catch (const std::runtime_error& e) {
                std::cerr << "Alignment error: " << e.what() << std::endl;
                // Handle alignment failure (e.g., skip replacement)
                glfwSwapBuffers(window);
                glfwPollEvents();
                continue;
            }

            // Apply transformation to the original CAD PCD
            std::cout << "Apply transformation to the original CAD PCD.." << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cad(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*cad_object, *transformed_cad, transformation);

            // Convert transformed CAD to XYZRGB (assigning a default color, e.g., white)
            std::cout << "Convert transformed CAD to XYZRGB.." << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cad_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
            for (const auto& point : transformed_cad->points) {
                pcl::PointXYZRGB point_rgb;
                point_rgb.x = point.x;
                point_rgb.y = point.y;
                point_rgb.z = point.z;
                point_rgb.r = 255;
                point_rgb.g = 255;
                point_rgb.b = 255;
                transformed_cad_rgb->points.push_back(point_rgb);
            }

            // Remove original object points from the scene
            std::cout << "Remove original object points from the scene.." << std::endl;
            pcl::PointIndices::Ptr object_indices(new pcl::PointIndices());
            for (size_t i = 0; i < object_pcd->points.size(); ++i) {
                // Find corresponding index in scene_pcl
                for (size_t j = 0; j < scene_pcl->points.size(); ++j) {
                    if (std::abs(scene_pcl->points[j].x - object_pcd->points[i].x) < 1e-5 &&
                        std::abs(scene_pcl->points[j].y - object_pcd->points[i].y) < 1e-5 &&
                        std::abs(scene_pcl->points[j].z - object_pcd->points[i].z) < 1e-5) {
                        object_indices->indices.push_back(j);
                        break;
                    }
                }
            }

            // Extract the scene without the original object points
            std::cout << "extract the scene without the original object points.." << std::endl;
            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(scene_pcl);
            extract.setIndices(object_indices);
            extract.setNegative(true);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_without_object(new pcl::PointCloud<pcl::PointXYZRGB>());
            extract.filter(*scene_without_object);

            // Combine the scene without the object with the transformed CAD PCD
            std::cout << "Combine the scene without the object.." << std::endl;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr updated_scene(new pcl::PointCloud<pcl::PointXYZRGB>());
            *updated_scene = *scene_without_object + *transformed_cad_rgb;

            // Visualize the updated scene using OpenGL
            std::cout << "Visualize the updated scene.." << std::endl;
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            draw_pcl_pointcloud(800, 600, app_state, updated_scene);

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
