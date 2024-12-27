#include "object_detection.hpp"

namespace detect {
ObjectDetectionResult detect_object(
    const std::shared_ptr<open3d::geometry::PointCloud>& captured_pcd,
    const std::shared_ptr<open3d::geometry::PointCloud>& cad_pcd) {

    auto downsampled_captured = captured_pcd->VoxelDownSample(0.01);
    downsampled_captured->EstimateNormals();

    auto downsampled_cad = cad_pcd->VoxelDownSample(0.01);
    downsampled_cad->EstimateNormals();

    auto result = open3d::pipelines::registration::RegistrationICP(
        *downsampled_captured, *downsampled_cad, 0.05, Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPoint());

    return { result.transformation_, downsampled_cad };
}
}
