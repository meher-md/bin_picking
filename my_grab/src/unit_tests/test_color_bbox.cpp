#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "detect_color_bbox.hpp"

TEST(BoundingBoxDetection, RealSenseStream) {
    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    auto config = pipe.start();

    // Define the HSV range for the orange color
    cv::Scalar lower_orange(0, 100, 100);
    cv::Scalar upper_orange(25, 255, 255);

    try {
        while (true) {
            // Wait for a new frame
            auto frames = pipe.wait_for_frames();
            auto color_frame = frames.get_color_frame();

            // Convert RealSense color frame to OpenCV Mat
            cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3,
                                (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

            // Ensure the image is valid
            if (color_image.empty()) {
                throw std::runtime_error("Captured frame is empty.");
            }

            // Convert RGB to BGR for proper color representation
            cv::cvtColor(color_image, color_image, cv::COLOR_RGB2BGR);

            // Detect bounding box for the orange object
            try {
                auto [min_bound, max_bound] = detect_color_bbox(color_image, lower_orange, upper_orange);

                std::cout << "Bounding Box Min Bound: (" << min_bound.x() << ", " << min_bound.y() << ")\n";
                std::cout << "Bounding Box Max Bound: (" << max_bound.x() << ", " << max_bound.y() << ")\n";


                // Draw the bounding box on the color image
                cv::rectangle(color_image,
                              cv::Point(static_cast<int>(min_bound.x()), static_cast<int>(min_bound.y())),
                              cv::Point(static_cast<int>(max_bound.x()), static_cast<int>(max_bound.y())),
                              cv::Scalar(0, 255, 0), 2);
            } catch (const std::exception&) {
                // If no object is detected, skip drawing the bounding box
            }

            // Display the result
            cv::imshow("Bounding Box Detection", color_image);

            // Break the loop if 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        FAIL() << "Error during RealSense stream or bounding box detection: " << e.what();
    }
}
