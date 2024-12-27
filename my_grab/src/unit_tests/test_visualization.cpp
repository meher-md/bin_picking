#include "visualization.hpp"
#include <gtest/gtest.h>

TEST(VisualizationTest, DisplayPointClouds) {
    auto combined_pcd = std::make_shared<open3d::geometry::PointCloud>();

    // Add dummy points for testing
    combined_pcd->points_.emplace_back(0, 0, 0);
    combined_pcd->points_.emplace_back(1, 1, 1);

    ASSERT_NO_THROW(visualize::show_combined_point_cloud(combined_pcd));
}