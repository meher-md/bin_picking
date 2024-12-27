#include "pcd_frm_depth.hpp"
#include <stdexcept>

std::shared_ptr<open3d::geometry::PointCloud> pcd_frm_depth(
    const rs2::depth_frame& depth_frame,
    const Eigen::Vector2d& min_bound_2d,
    const Eigen::Vector2d& max_bound_2d,
    const rs2_intrinsics& intrinsics) {
    
    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();

    for (int y = static_cast<int>(min_bound_2d.y()); y < static_cast<int>(max_bound_2d.y()); ++y) {
        for (int x = static_cast<int>(min_bound_2d.x()); x < static_cast<int>(max_bound_2d.x()); ++x) {
            float depth = depth_frame.get_distance(x, y);
            if (depth <= 0) continue;

            // Explicitly create the pixel array to avoid temporary array issues
            float pixel[2] = {static_cast<float>(x), static_cast<float>(y)};

            // Deproject pixel to 3D point
            float point[3];
            rs2_deproject_pixel_to_point(point, &intrinsics, pixel, depth);

            point_cloud->points_.emplace_back(point[0], point[1], point[2]);
        }
    }

    return point_cloud;
}
