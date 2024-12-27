#ifndef CAD_PCD_HPP
#define CAD_PCD_HPP

#include <open3d/Open3D.h>
#include <string>
#include <memory>

namespace cad {
std::shared_ptr<open3d::geometry::PointCloud> get_cad_pcd(const std::string& file_path, int number_of_points = 100000);
}

#endif