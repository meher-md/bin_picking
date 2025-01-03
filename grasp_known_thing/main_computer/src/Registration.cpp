// Registration.cpp
#include "Registration.hpp"
#include <open3d/Open3D.h>
#include <filesystem>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace {

std::tuple<Eigen::Matrix4d, double> registerPointCloudToReference(
    std::shared_ptr<open3d::geometry::PointCloud> source,
    std::shared_ptr<open3d::geometry::PointCloud> target)
{
    // Your ICP or other alignment logic goes here
    // For example, using Open3D’s registration
    auto result = open3d::pipelines::registration::RegistrationICP(
        *source, *target, 0.005, 
        Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPlane()
    );

    double fitness = result.fitness_;
    return {result.transformation_, fitness};
}

std::shared_ptr<open3d::geometry::PointCloud> removeStatisticalOutliers(
    std::shared_ptr<open3d::geometry::PointCloud> pcd, 
    int nb_neighbors = 20, 
    double std_ratio = 2.0)
{
    auto [pcd_clean, inliers] = pcd->RemoveStatisticalOutliers(nb_neighbors, std_ratio);
    return pcd_clean;
}

std::shared_ptr<open3d::geometry::PointCloud> keepLargestCluster(
    std::shared_ptr<open3d::geometry::PointCloud> pcd,
    double eps = 0.02,
    int min_points = 10)
{
    auto labels = pcd->ClusterDBSCAN(eps, min_points, false);
    std::unordered_map<int, int> label_count;
    for (const auto& label : labels) {
        if (label >= 0) label_count[label]++;
    }
    if (label_count.empty()) {
        std::cout << "[Registration] No valid clusters found.\n";
        return pcd;
    }

    int largest_label = -1;
    int max_count = 0;
    for (const auto& [label, count] : label_count) {
        if (count > max_count) {
            max_count = count;
            largest_label = label;
        }
    }

    std::vector<size_t> indices;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == largest_label) {
            indices.push_back(i);
        }
    }
    return pcd->SelectByIndex(indices);
}

std::shared_ptr<open3d::geometry::PointCloud> downsamplePointCloud(
    std::shared_ptr<open3d::geometry::PointCloud> pcd,
    double voxel_size)
{
    if (!pcd) return nullptr;
    auto downsampled = pcd->VoxelDownSample(voxel_size);
    if (downsampled->IsEmpty()) return nullptr;
    return downsampled;
}

} // anonymous namespace

namespace Registration {

void processAndVisualizePointClouds(const std::string &object_pcd_path,
                                    const std::string &reference_pcd_path)
{
    // 1. Check files
    if (!fs::exists(object_pcd_path) || !fs::exists(reference_pcd_path)) {
        std::cerr << "[Registration] PCD file(s) not found.\n";
        return;
    }

    // 2. Load object PCD
    auto raw_object_pcd = open3d::io::CreatePointCloudFromFile(object_pcd_path);
    if (!raw_object_pcd || raw_object_pcd->IsEmpty()) {
        std::cerr << "[Registration] Failed to load or empty: " << object_pcd_path << "\n";
        return;
    }

    // 3. Downsample, cluster, outlier removal
    auto object_pcd = downsamplePointCloud(raw_object_pcd, 0.001);
    if (!object_pcd) {
        std::cerr << "[Registration] Downsample failed.\n";
        return;
    }
    object_pcd = keepLargestCluster(object_pcd);
    object_pcd = removeStatisticalOutliers(object_pcd, 20, 2.0);

    // 4. Load reference (CAD) PCD
    auto reference_pcd = open3d::io::CreatePointCloudFromFile(reference_pcd_path);
    if (!reference_pcd || reference_pcd->IsEmpty()) {
        std::cerr << "[Registration] Failed to load or empty: " << reference_pcd_path << "\n";
        return;
    }

    // Optional: visualize them separately
    open3d::visualization::DrawGeometries({object_pcd}, "Object PCD");
    open3d::visualization::DrawGeometries({reference_pcd}, "Reference PCD");

    // 5. Align bounding boxes roughly by scaling & center
    auto object_bbox = object_pcd->GetAxisAlignedBoundingBox();
    auto ref_bbox = reference_pcd->GetAxisAlignedBoundingBox();
    double object_size = (object_bbox.GetMaxBound() - object_bbox.GetMinBound()).maxCoeff();
    double ref_size = (ref_bbox.GetMaxBound() - ref_bbox.GetMinBound()).maxCoeff();
    double scale_factor = object_size / ref_size;

    reference_pcd->Scale(scale_factor, reference_pcd->GetCenter());
    Eigen::Vector3d trans = object_pcd->GetCenter() - reference_pcd->GetCenter();
    reference_pcd->Translate(trans);

    // 6. Merge them for a quick look
    open3d::visualization::DrawGeometries({object_pcd, reference_pcd}, "Merged (Pre-ICP)");

    // 7. Registration (ICP or similar)
    auto [transformation, fitness] = registerPointCloudToReference(object_pcd, reference_pcd);
    std::cout << "[Registration] Transformation:\n" << transformation << "\n";
    std::cout << "[Registration] Fitness: " << fitness << "\n";

    // 8. Transform object & visualize final
    object_pcd->Transform(transformation);
    open3d::visualization::DrawGeometries({object_pcd, reference_pcd}, "Aligned");
}

} // namespace Registration
