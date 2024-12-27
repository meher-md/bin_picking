#include "object_detection.hpp"
#include <gtest/gtest.h>

TEST(ObjectDetectionTest, DetectObjectAlignment) {
    auto captured_pcd = std::make_shared<open3d::geometry::PointCloud>();
    auto cad_pcd = std::make_shared<open3d::geometry::PointCloud>();

    // Add dummy points for testing
    captured_pcd->points_.emplace_back(0, 0, 0);
    cad_pcd->points_.emplace_back(0, 0, 0);

    auto result = detect::detect_object(captured_pcd, cad_pcd);
    ASSERT_FALSE(result.transformation.isZero(0));
}