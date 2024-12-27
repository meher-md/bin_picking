#ifndef PCD_CAPTURE_HPP
#define PCD_CAPTURE_HPP

#include <librealsense2/rs.hpp>
#include <open3d/Open3D.h>
#include <memory>

namespace capture {
std::shared_ptr<open3d::geometry::PointCloud> capture_pcd(rs2::pipeline& pipe);
}

#endif
