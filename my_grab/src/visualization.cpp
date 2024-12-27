#include "visualization.hpp"

namespace visualize {
    void show_combined_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud>& combined_pcd) {
        open3d::visualization::DrawGeometries({ combined_pcd }, "Combined Point Cloud", 1280, 720);
    }
}
