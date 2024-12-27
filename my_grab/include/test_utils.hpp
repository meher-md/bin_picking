#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <open3d/Open3D.h>

namespace test_utils {
inline std::shared_ptr<open3d::geometry::PointCloud> create_dummy_pcd(size_t num_points) {
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    for (size_t i = 0; i < num_points; ++i) {
        pcd->points_.emplace_back(static_cast<double>(i), 0.0, 0.0);
    }
    return pcd;
}
}

#endif