#include "pcd_capture.hpp"
#include <gtest/gtest.h>

TEST(PCDCaptureTest, CaptureNonNull) {
    rs2::pipeline pipe;

    // Configure the pipeline with default settings
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    ASSERT_NO_THROW({
        auto pcd = capture::capture_pcd(pipe);
        ASSERT_NE(pcd, nullptr);
        ASSERT_GT(pcd->points_.size(), 0);
    });
}
