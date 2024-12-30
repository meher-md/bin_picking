// src/helper_functions.cpp

#include "helper_functions.hpp"

std::pair<Eigen::Matrix4d, double> register_point_cloud_to_reference(
    const std::shared_ptr<open3d::geometry::PointCloud>& source,
    const std::shared_ptr<open3d::geometry::PointCloud>& target) {
    
    // Initialize the registration result
    open3d::pipelines::registration::RegistrationResult result;

    // Create downsampled copies of the source and target point clouds using the copy constructor
    auto source_down = std::make_shared<open3d::geometry::PointCloud>(*source);
    auto target_down = std::make_shared<open3d::geometry::PointCloud>(*target);
    
    // Estimate normals on downsampled point clouds
    source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    
    // Compute FPFH features
    auto source_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
        *source_down, open3d::geometry::KDTreeSearchParamKNN(100));
    auto target_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
        *target_down, open3d::geometry::KDTreeSearchParamKNN(100));

    // Define the transformation estimation method (Point to Point)
    open3d::pipelines::registration::TransformationEstimationPointToPoint estimation;

    // Define the RANSAC convergence criteria
    open3d::pipelines::registration::RANSACConvergenceCriteria criteria(4000000, 500);

    // Define correspondence checkers if needed (empty in this case)
    std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> correspondence_checkers;

    // Perform RANSAC registration with feature matching
    result = open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
        *source_down, *target_down, *source_fpfh, *target_fpfh,
        true, 0.05, estimation,
        0, // Assuming '0' is a placeholder for the required 'int' parameter
        correspondence_checkers,
        criteria
    );

    // **Important: Ensure that both original source and target have normals**
    
    // Estimate normals on the original target point cloud if not already done
    if (!target->HasNormals()) {
        target->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    }

    // Optionally, estimate normals on the original source point cloud if desired
    if (!source->HasNormals()) {
        source->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    }

    // Perform ICP for fine alignment using Point to Plane estimation
    open3d::pipelines::registration::TransformationEstimationPointToPlane icp_estimation;
    result = open3d::pipelines::registration::RegistrationICP(
        *source, *target, 0.02, result.transformation_,
        icp_estimation
    );

    // Return the transformation and fitness
    return {result.transformation_, result.fitness_};
}
