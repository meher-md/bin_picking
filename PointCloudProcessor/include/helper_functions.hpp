// include/helper_functions.hpp

#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <open3d/Open3D.h>    // Corrected case to 'open3d'
#include <utility>            // For std::pair

// Function to register a point cloud to a reference point cloud.
// Returns a pair containing the transformation matrix and fitness score.
std::pair<Eigen::Matrix4d, double> register_point_cloud_to_reference(
    const std::shared_ptr<open3d::geometry::PointCloud>& source,
    const std::shared_ptr<open3d::geometry::PointCloud>& target);

#endif // HELPER_FUNCTIONS_HPP
