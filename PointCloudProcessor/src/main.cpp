// src/main.cpp

#include <open3d/Open3D.h>
#include "helper_functions.hpp"
#include "pcd_saving.hpp"         // Include the PCD saving module
#include "pcl_to_open3d.hpp"      // Include the PCL to Open3D converter
#include "pcd_processing.hpp"
#include <filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;

// Function to covert point cloud from pcl to Open3D
std::shared_ptr<open3d::geometry::PointCloud> pclToOpen3D(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
    auto open3d_cloud = std::make_shared<open3d::geometry::PointCloud>();

    // Iterate through PCL points and add them to Open3D point cloud
    for (const auto& point : pcl_cloud->points) {
        open3d_cloud->points_.emplace_back(point.x, point.y, point.z);
    }

    return open3d_cloud;
}

// Function to downsample a point cloud using VoxelGrid filter

std::shared_ptr<open3d::geometry::PointCloud> DownsamplePointCloud(
    const std::shared_ptr<open3d::geometry::PointCloud> &pcd, 
    double voxel_size) 
{
    if (!pcd) {
        std::cerr << "Error: Invalid point cloud for downsampling.\n";
        return nullptr;
    }

    std::cout << "Downsampling point cloud with voxel size: " << voxel_size << "...\n";
    auto downsampled_pcd = pcd->VoxelDownSample(voxel_size);
    if (downsampled_pcd->IsEmpty()) {
        std::cerr << "Error: Downsampled point cloud is empty.\n";
        return nullptr;
    }

    std::cout << "Downsampled point cloud has " << downsampled_pcd->points_.size() << " points.\n";
    return downsampled_pcd;
}

std::shared_ptr<open3d::geometry::PointCloud> get_cad_pcd(const std::string& file_path, int number_of_points) {
    auto mesh = open3d::io::CreateMeshFromFile(file_path);

    auto pcd = mesh->SamplePointsUniformly(number_of_points);
    if (pcd->points_.empty()) {
        throw std::runtime_error("Generated point cloud is empty.");
    }

    // Apply voxel downsampling
    
    auto downsampled_pcd = pcd->VoxelDownSample(0.001);
    if (downsampled_pcd->points_.empty()) {
        throw std::runtime_error("Downsampled point cloud is empty.");
    }

    // Save the generated PCD to a file
    std::string output_path = "../assets/generated_pcd.pcd";
    if (!open3d::io::WritePointCloud(output_path, *downsampled_pcd)) {
        throw std::runtime_error("Failed to save PCD to file: " + output_path);
    }

    std::cout << "Saved PCD to: " << output_path << std::endl;
    return pcd;
}
// Function to remove statistical outliers
std::shared_ptr<open3d::geometry::PointCloud> RemoveStatisticalOutliers(
    const std::shared_ptr<open3d::geometry::PointCloud>& pcd,
    int nb_neighbors = 20,
    double std_ratio = 2.0) {
    
    auto [pcd_clean, inliers] = pcd->RemoveStatisticalOutliers(nb_neighbors, std_ratio);
    return pcd_clean;
}

// Function to keep the largest cluster using DBSCAN
std::shared_ptr<open3d::geometry::PointCloud> KeepLargestCluster(
    const std::shared_ptr<open3d::geometry::PointCloud>& pcd,
    double eps = 0.02,
    int min_points = 10) {
    
    auto labels = pcd->ClusterDBSCAN(eps, min_points, false);
    std::unordered_map<int, int> label_count;
    for (const auto& label : labels) {
        if (label >= 0) {
            label_count[label]++;
        }
    }

    if (label_count.empty()) {
        std::cout << "No valid clusters found!\n";
        return pcd;
    }

    // Find the label with the maximum count
    int largest_label = -1;
    int max_count = 0;
    for (const auto& [label, count] : label_count) {
        if (count > max_count) {
            max_count = count;
            largest_label = label;
        }
    }

    // Extract points belonging to the largest cluster
    std::vector<size_t> indices;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == largest_label) {
            indices.push_back(i);
        }
    }

    return pcd->SelectByIndex(indices);
}

void ProcessAndVisualizePointClouds() {
    // Path to the data folder containing .pcd files
    std::string data_folder = "../assets";

    // Ensure the data folder exists
    if (!fs::exists(data_folder)) {
        std::cerr << "Error: The folder '" << data_folder << "' does not exist.\n";
        return;
    }

    // Load the visible point cloud file
    std::string visible_pcd_path = data_folder + "/object_pcd.pcd";
    if (!fs::exists(visible_pcd_path)) {
        std::cerr << "Error: File '" << visible_pcd_path << "' not found.\n";
        return;
    }

    auto raw_pcd = open3d::io::CreatePointCloudFromFile(visible_pcd_path);
    if (!raw_pcd) {
        std::cerr << "Error: Failed to read point cloud from '" << visible_pcd_path << "'.\n";
        return;
    }

    // Downsample the point cloud
    double voxel_size = 0.001; // Adjust voxel size as needed
    auto downsampled_pcd = DownsamplePointCloud(raw_pcd, voxel_size);
    if (!downsampled_pcd) {
        std::cerr << "Error: Failed to down sample the pcd '" << visible_pcd_path << "'.\n";
        return;
    }

    // Keep the largest cluster after outlier removal
    auto visible_pcd = KeepLargestCluster(downsampled_pcd);

    // Preprocess the object point cloud
    visible_pcd = RemoveStatisticalOutliers(visible_pcd);
    
    // Load the reference point cloud file
    std::string reference_pcd_path = data_folder + "/generated_pcd.pcd";
    if (!fs::exists(reference_pcd_path)) {
        std::cerr << "Error: File '" << reference_pcd_path << "' not found.\n";
        return;
    }

    auto reference_pcd = open3d::io::CreatePointCloudFromFile(reference_pcd_path);
    if (!reference_pcd) {
        std::cerr << "Error: Failed to read point cloud from '" << reference_pcd_path << "'.\n";
        return;
    }

    // Visualize individual point clouds
    std::cout << "Displaying the visible point cloud...\n";
    open3d::visualization::DrawGeometries({visible_pcd}, "Visible Point Cloud");

    std::cout << "Displaying the reference point cloud...\n";
    open3d::visualization::DrawGeometries({reference_pcd}, "Reference Point Cloud");

    // Print bounding boxes
    std::cout << "Original Bounding Boxes:\n";
    auto visible_bbox = visible_pcd->GetAxisAlignedBoundingBox();
    auto reference_bbox = reference_pcd->GetAxisAlignedBoundingBox();
    std::cout << "Visible Point Cloud Bounds: " << visible_bbox.GetMinBound().transpose() 
              << " to " << visible_bbox.GetMaxBound().transpose() << "\n";
    std::cout << "Reference Point Cloud Bounds: " << reference_bbox.GetMinBound().transpose() 
              << " to " << reference_bbox.GetMaxBound().transpose() << "\n";

    // Normalize the scale of the reference point cloud
    double visible_size = (visible_bbox.GetMaxBound() - visible_bbox.GetMinBound()).maxCoeff();
    double reference_size = (reference_bbox.GetMaxBound() - reference_bbox.GetMinBound()).maxCoeff();
    double scale_factor = visible_size / reference_size;
    std::cout << "Scaling reference point cloud by factor: " << scale_factor << "\n";
    reference_pcd->Scale(scale_factor, reference_pcd->GetCenter());

    // Align the centers of the two point clouds
    Eigen::Vector3d visible_center = visible_pcd->GetCenter();
    Eigen::Vector3d reference_center = reference_pcd->GetCenter();
    Eigen::Vector3d translation_vector = visible_center - reference_center;
    std::cout << "Translating reference point cloud by vector: " 
              << translation_vector.transpose() << "\n";
    reference_pcd->Translate(translation_vector);

    // Visualize merged point clouds
    std::cout << "Displaying the merged point clouds...\n";
    open3d::visualization::DrawGeometries(
        {visible_pcd, reference_pcd}, "Merged Point Clouds (Aligned)");

    // Perform registration
    std::cout << "Performing registration...\n";
    auto [transformation, fitness] = register_point_cloud_to_reference(visible_pcd, reference_pcd);
    std::cout << "Transformation Matrix:\n" << transformation << "\n";
    std::cout << "Fitness Score: " << fitness << "\n";

    // Apply the transformation and visualize the aligned point clouds
    visible_pcd->Transform(transformation);
    std::cout << "Displaying the aligned point clouds...\n";
    open3d::visualization::DrawGeometries(
        {visible_pcd, reference_pcd}, "Aligned Point Clouds");
}

int main(int argc, char* argv[]) {
    try {


        // Generate CAD point cloud for reference
        std::string cad_file_path = "../assets/VB_1400.obj"; // Path to the CAD file
        std::shared_ptr<open3d::geometry::PointCloud> reference_pcd;

        try {
            // Generate the CAD point cloud directly as an Open3D point cloud
            reference_pcd = get_cad_pcd(cad_file_path, 100000); // Provide appropriate file path and number of points
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return EXIT_FAILURE;
        }

        // Save the generated CAD PCD to a file
        std::string output_pcd_path = "../assets/generated_pcd.pcd";
        if (!open3d::io::WritePointCloud(output_pcd_path, *reference_pcd)) {
            std::cerr << "Error: Failed to save the generated point cloud to '" << output_pcd_path << "'.\n";
            return EXIT_FAILURE;
        }
        std::cout << "Saved the generated CAD point cloud to: " << output_pcd_path << "\n";


        // Specify the path where the PCD file will be saved
        std::string pcd_file_path = "../assets/object_pcd.pcd";
        
        // Run the point cloud processing pipeline
        run_pointcloud_processing(pcd_file_path);

        // Now call the ProcessAndVisualizePointClouds function
        ProcessAndVisualizePointClouds();
        
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
