#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <open3d/Open3D.h>

namespace visualize {
    void show_combined_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud>& combined_pcd);
}

#endif