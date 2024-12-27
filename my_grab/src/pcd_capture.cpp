#include "pcd_capture.hpp"

namespace capture {

std::shared_ptr<open3d::geometry::PointCloud> capture_pcd(rs2::pipeline& pipe) {
    // Wait for frames from the RealSense pipeline
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();

    // Declare a point cloud object from RealSense
    rs2::pointcloud pc;
    rs2::points points = pc.calculate(depth);

    // Get the vertices from the point cloud
    const rs2::vertex* vertices = points.get_vertices();
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();

    // Loop through each vertex and populate Open3D's point cloud
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& v = vertices[i];
        // Check for valid depth data (z > 0)
        if (v.z > 0) {
            pcd->points_.emplace_back(v.x, v.y, v.z);
        }
    }

    // Check if the point cloud is empty and throw an error if so
    if (pcd->points_.empty()) {
        throw std::runtime_error("Generated point cloud is empty. Depth data may be invalid.");
    }

    return pcd;
}

}

