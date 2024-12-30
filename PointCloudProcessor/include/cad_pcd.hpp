// include/cad_pcd.hpp

#ifndef CAD_PCD_HPP
#define CAD_PCD_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>

namespace cad {
    // Function to get CAD point cloud from a CAD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr get_cad_pcd(const std::string& cad_file, int resolution);
}

#endif // CAD_PCD_HPP
