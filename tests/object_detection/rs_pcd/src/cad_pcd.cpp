#include "cad_pcd.hpp"
#include <open3d/Open3D.h>
#include <iostream>
#include <stdexcept>

namespace cad {

std::shared_ptr<open3d::geometry::PointCloud> get_cad_pcd(const std::string& file_path, int number_of_points) {
    auto mesh = open3d::io::CreateMeshFromFile(file_path);

    auto pcd = mesh->SamplePointsUniformly(number_of_points);
    if (pcd->points_.empty()) {
        throw std::runtime_error("Generated point cloud is empty.");
    }

    // Save the generated PCD to a file
    std::string output_path = "../assets/generated_pcd.pcd";
    if (!open3d::io::WritePointCloud(output_path, *pcd)) {
        throw std::runtime_error("Failed to save PCD to file: " + output_path);
    }

    std::cout << "Saved PCD to: " << output_path << std::endl;
    return pcd;
}

} // namespace cad
