#ifndef PCD_FRM_DEPTH_HPP
#define PCD_FRM_DEPTH_HPP

#include <librealsense2/rs.hpp>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <memory>

// Function to create a point cloud from depth image and bounding box
std::shared_ptr<open3d::geometry::PointCloud> pcd_frm_depth(
    const rs2::depth_frame& depth_frame,
    const Eigen::Vector2d& min_bound_2d,
    const Eigen::Vector2d& max_bound_2d,
    const rs2_intrinsics& intrinsics);

#endif // CREATE_POINT_CLOUD_FROM_DEPTH_BBOX_HPP
