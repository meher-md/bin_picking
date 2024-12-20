/*
 * File: capture_pointcloud.cpp
 * Description: 
 * This program captures depth and color frames using an Intel RealSense camera,
 * processes the frames to align depth data with color data, and generates a 
 * point cloud. The resulting point cloud is saved in PLY format, suitable for 
 * 3D visualization and analysis. Key features include:
 * 
 * - Captures high-resolution depth and color frames (1280x720, 30 FPS).
 * - Aligns depth data to color stream for accurate point cloud generation.
 * - Applies filters to improve depth data quality:
 *   - Decimation filter to reduce resolution.
 *   - Spatial filter to smooth depth values.
 *   - Temporal filter to minimize temporal noise.
 * - Maps color data to the point cloud for better visualization.
 * - Exports the processed point cloud to a PLY file.
 * 
 * Usage:
 *   ./capture_pointcloud <output_ply_file>
 * 
 * Dependencies:
 *   - Intel RealSense SDK (librealsense2)
 *   - Standard C++ libraries
 * 
 * Note:
 *   Ensure a compatible Intel RealSense device is connected before running 
 *   this program. The output PLY file can be viewed in any 3D viewer supporting 
 *   the PLY format.
 */
// Issue #001: Sparse point clouds.   
#include <librealsense2/rs.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <output_ply_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const char *output_file = argv[1];

    try {
        // Declare RealSense pipeline
        rs2::pipeline pipe;
        rs2::config config;

        // Configure streams with specific settings
        config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
        config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);

        // Start the pipeline with the configuration
        rs2::pipeline_profile profile = pipe.start(config);

        // Warm-up delay
        std::cout << "Warming up the camera..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Skip initial frames
        for (int i = 0; i < 30; ++i) {
            pipe.wait_for_frames();
        }

        // Capture frames
        std::cout << "Capturing frames... Please wait!" << std::endl;
        auto frames = pipe.wait_for_frames();

        // Align depth to color
        rs2::align align_to(RS2_STREAM_COLOR);
        auto aligned_frames = align_to.process(frames);

        auto depth = aligned_frames.get_depth_frame();
        auto color = aligned_frames.get_color_frame();

        if (!depth || !color) {
            throw std::runtime_error("Aligned frames are missing!");
        }

        std::cout << "Depth frame resolution: " << depth.get_width() << "x" << depth.get_height() << std::endl;
        std::cout << "Color frame resolution: " << color.get_width() << "x" << color.get_height() << std::endl;

        // Validate frame data
        if (depth.get_data_size() == 0 || color.get_data_size() == 0) {
            throw std::runtime_error("Captured frames are empty!");
        }

        // Apply filters to improve depth data
        rs2::decimation_filter decimation; // Reduce resolution
        rs2::spatial_filter spatial;       // Smooth depth
        rs2::temporal_filter temporal;     // Reduce temporal noise

        depth = decimation.process(depth);
        depth = spatial.process(depth);
        depth = temporal.process(depth);

        // Create point cloud
        rs2::pointcloud pc;
        rs2::points points = pc.calculate(depth);

        // Map color to point cloud
        pc.map_to(color);

        // Validate point cloud
        if (points.size() == 0) {
            throw std::runtime_error("Generated point cloud is empty!");
        }
        std::cout << "Number of points: " << points.size() << std::endl;

        // Save to PLY
        std::cout << "Saving point cloud to " << output_file << "..." << std::endl;
        points.export_to_ply(output_file, color);
        std::cout << "Point cloud saved successfully!" << std::endl;

    } catch (const rs2::error &e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "("
                  << e.get_failed_args() << "): " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
