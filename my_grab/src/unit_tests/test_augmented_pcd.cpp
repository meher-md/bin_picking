#include "pcd_capture.hpp"
#include "visualization.hpp"
#include <gtest/gtest.h>
#include <librealsense2/rs.hpp> // RealSense API

TEST(PCDCaptureTest, AugmentedPointCloudWithRealColors) {
    rs2::pipeline pipe;
    pipe.start();

    auto augmented_pcd = std::make_shared<open3d::geometry::PointCloud>();
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(5);

    while (std::chrono::steady_clock::now() < end_time) {
        auto frames = pipe.wait_for_frames();
        auto color = frames.get_color_frame();
        auto depth = frames.get_depth_frame();

        rs2::pointcloud pc;
        rs2::points points = pc.calculate(depth);
        pc.map_to(color);

        auto vertices = points.get_vertices();
        auto tex_coords = points.get_texture_coordinates();

        for (size_t i = 0; i < points.size(); ++i) {
            const auto& v = vertices[i];
            if (v.z) { // Avoid points with zero depth
                augmented_pcd->points_.emplace_back(v.x, v.y, v.z);

                // Generate real color data
                const auto& uv = tex_coords[i];
                int x = static_cast<int>(uv.u * color.get_width());
                int y = static_cast<int>(uv.v * color.get_height());
                uint8_t r = 0, g = 0, b = 0;

                if (x >= 0 && x < color.get_width() && y >= 0 && y < color.get_height()) {
                    auto pixels = reinterpret_cast<const uint8_t*>(color.get_data());
                    size_t idx = (y * color.get_width() + x) * 3;
                    r = pixels[idx];
                    g = pixels[idx + 1];
                    b = pixels[idx + 2];
                }

                augmented_pcd->colors_.emplace_back(
                    static_cast<double>(r) / 255.0, 
                    static_cast<double>(g) / 255.0, 
                    static_cast<double>(b) / 255.0
                );
            }
        }
    }

    ASSERT_GT(augmented_pcd->points_.size(), 0);
    ASSERT_GT(augmented_pcd->colors_.size(), 0);

    // Display the augmented PCD with real colors
    ASSERT_NO_THROW(visualize::show_combined_point_cloud(augmented_pcd));
}
