#include <iostream>
#include <Eigen/Dense>
#include <Open3D/Open3D.h>

// Function to create synthetic point clouds
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>,
           std::shared_ptr<open3d::geometry::PointCloud>,
           Eigen::Matrix4d>
create_test_point_clouds() {
    // Reference point cloud (a cube)
    auto ref_pcd = std::make_shared<open3d::geometry::PointCloud>();
    std::vector<Eigen::Vector3d> ref_points = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},  // Bottom face
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}   // Top face
    };
    ref_pcd->points_ = ref_points;

    // Transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 3) = 0.5;  // Translation in X
    transform(1, 3) = 0.2;  // Translation in Y
    transform(2, 3) = 0.3;  // Translation in Z

    // Transformed point cloud
    auto trans_pcd = std::make_shared<open3d::geometry::PointCloud>();
    for (const auto& point : ref_points) {
        Eigen::Vector4d homogenous_point(point(0), point(1), point(2), 1.0);
        Eigen::Vector4d transformed_point = transform * homogenous_point;
        trans_pcd->points_.emplace_back(transformed_point.head<3>());
    }

    return {ref_pcd, trans_pcd, transform};
}

// Function to perform ICP registration
std::tuple<Eigen::Matrix4d, double> register_point_cloud_to_reference(
    const std::shared_ptr<open3d::geometry::PointCloud>& source,
    const std::shared_ptr<open3d::geometry::PointCloud>& target,
    double threshold) {
    auto result = open3d::pipelines::registration::RegistrationICP(
        *source, *target, threshold,
        Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPoint());
    return {result.transformation_, result.fitness_};
}

// Test registration function
void test_registration() {
    auto [ref_pcd, visible_pcd, ground_truth_transform] = create_test_point_clouds();

    // Visualize the original clouds (optional)
    std::cout << "Visualizing reference and visible point clouds..." << std::endl;
    ref_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));  // Red
    visible_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));  // Green
    open3d::visualization::DrawGeometries({ref_pcd, visible_pcd}, "Original Point Clouds");

    // Run ICP registration
    std::cout << "Running registration..." << std::endl;
    double threshold = 0.05;
    auto [transformation, fitness] = register_point_cloud_to_reference(visible_pcd, ref_pcd, threshold);

    std::cout << "Computed Transformation Matrix:\n" << transformation << std::endl;
    std::cout << "Fitness: " << fitness << std::endl;
    std::cout << "Ground Truth Transformation Matrix:\n" << ground_truth_transform << std::endl;

    // Apply the transformation to align the visible point cloud
    auto aligned_pcd = std::make_shared<open3d::geometry::PointCloud>();
    *aligned_pcd = *visible_pcd;
    aligned_pcd->Transform(transformation);

    // Visualize the aligned clouds
    std::cout << "Visualizing aligned point clouds..." << std::endl;
    ref_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));  // Red
    aligned_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));  // Green
    open3d::visualization::DrawGeometries({ref_pcd, aligned_pcd}, "Aligned Point Clouds");
}

int main() {
    test_registration();
    return 0;
}
