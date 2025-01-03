#include <open3d/Open3D.h>
#include <filesystem>
#include <iostream>

#include "SensorReceiver.hpp"
#include "SceneProcessing.hpp"
#include "Registration.hpp"
#include "PclUtils.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    try {
        // 1. Connect to RealSense server & YOLO server
        SensorReceiver sensorReceiver("tcp://localhost:6000",  // RealSense server
                                      "tcp://localhost:5555"); // YOLO server
        sensorReceiver.requestIntrinsics();  // Get camera intrinsics from the Pi server

        // 2. Convert CAD to PCD (one-time) and save to disk
        std::string cad_file_path = "../assets/VB_1400.obj";  // CAD file
        std::string generated_pcd_path = "../assets/generated_pcd.pcd";

        auto reference_pcd = SceneProcessing::generateCadPointCloud(cad_file_path, 100000);
        open3d::io::WritePointCloud(generated_pcd_path, *reference_pcd);

        std::cout << "Saved generated CAD point cloud to: " << generated_pcd_path << "\n";

        // 3. Build scene point cloud from Pi + YOLO bounding box
        std::string object_pcd_path = "../assets/object_pcd.pcd";
        SceneProcessing::runPointCloudProcessing(sensorReceiver, object_pcd_path);

        // 4. Perform final registration & visualization
        Registration::processAndVisualizePointClouds(
            "../assets/object_pcd.pcd",    // The object PCD from the scene
            "../assets/generated_pcd.pcd"  // The reference CAD PCD
        );

        // 5. (Future) Grasp planning & motion planning can go here
        //    e.g., GraspPlanning::computeGraspPoints(...);
        //    RobotPlanner::executeGrasp(...);

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
