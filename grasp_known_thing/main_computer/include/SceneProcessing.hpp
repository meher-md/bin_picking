// SceneProcessing.hpp
#pragma once

#include <memory>
#include <string>
#include <open3d/Open3D.h>
#include "SensorReceiver.hpp"

namespace SceneProcessing {

    // Generate CAD -> PCD from a .obj file, downsample, and return as Open3D point cloud
    std::shared_ptr<open3d::geometry::PointCloud> generateCadPointCloud(
        const std::string& cad_file_path,
        int number_of_points);

    // Runs the pipeline: 
    //  - Repeatedly request frames from SensorReceiver
    //  - Build PCL from color+depth
    //  - Crop object using bounding box
    //  - Save final object PCD to disk
    void runPointCloudProcessing(SensorReceiver &sensorReceiver,
                                 const std::string &pcd_output_path);
}
