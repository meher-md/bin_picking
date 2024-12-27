#include <gtest/gtest.h>
#include <librealsense2/rs.hpp>
#include "pcd_frm_depth.hpp"

TEST(PointCloudFromDepth, ValidInput) {
    rs2::pipeline pipe;
    auto profile = pipe.start();

    auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics();

    auto frames = pipe.wait_for_frames();
    auto depth_frame = frames.get_depth_frame();

    Eigen::Vector2d min_bound_2d(100, 100);
    Eigen::Vector2d max_bound_2d(200, 200);

    EXPECT_NO_THROW({
        auto point_cloud = pcd_frm_depth(depth_frame, min_bound_2d, max_bound_2d, intrinsics);
        ASSERT_FALSE(point_cloud->points_.empty());
    });
}
