#include "cad_pcd.hpp"
#include <gtest/gtest.h>

TEST(CADPCDTest, LoadAndSaveCADFile) {
    std::string test_file = "../assets/VN_1400.obj"; // Replace with a valid STL file path
    ASSERT_NO_THROW({
        auto pcd = cad::get_cad_pcd(test_file, 100000);
        ASSERT_NE(pcd, nullptr);
        ASSERT_GT(pcd->points_.size(), 0);
    });

    // Verify the output PCD file
    std::string output_file = "../assets/generated_pcd.pcd";
    auto loaded_pcd = open3d::io::CreatePointCloudFromFile(output_file);
    ASSERT_NE(loaded_pcd, nullptr);
    ASSERT_GT(loaded_pcd->points_.size(), 0);
}
