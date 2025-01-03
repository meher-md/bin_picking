// Registration.hpp
#pragma once

#include <string>

namespace Registration {
    // Load the object and reference PCDs, do scaling, alignment, ICP, etc.
    void processAndVisualizePointClouds(const std::string &object_pcd_path,
                                        const std::string &reference_pcd_path);
}
