#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <open3d/Open3D.h>
#include <string>
#include <memory>

namespace detect {
struct ObjectDetectionResult {
    Eigen::Matrix4d transformation;
    std::shared_ptr<open3d::geometry::PointCloud> aligned_pcd;
};

ObjectDetectionResult detect_object(
    const std::shared_ptr<open3d::geometry::PointCloud>& captured_pcd,
    const std::shared_ptr<open3d::geometry::PointCloud>& cad_pcd);
}

#endif