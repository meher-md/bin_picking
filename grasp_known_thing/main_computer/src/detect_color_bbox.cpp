#include "detect_color_bbox.hpp"
#include <stdexcept>

std::pair<Eigen::Vector2d, Eigen::Vector2d> detect_color_bbox(const cv::Mat& color_image, 
                                                              const cv::Scalar& lower_bound, 
                                                              const cv::Scalar& upper_bound) {
    if (color_image.empty()) {
        throw std::runtime_error("Empty color image provided.");
    }

    // Convert the image to HSV for better color segmentation
    cv::Mat hsv_image;
    cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);

    // Create a mask for the specified color
    cv::Mat mask;
    cv::inRange(hsv_image, lower_bound, upper_bound, mask);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        throw std::runtime_error("No object detected in the specified color range.");
    }

    // Get the largest contour as the detected object
    double max_area = 0;
    std::vector<cv::Point> largest_contour;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > max_area) {
            max_area = area;
            largest_contour = contour;
        }
    }

    // Compute the bounding box of the largest contour
    cv::Rect bbox = cv::boundingRect(largest_contour);

    // Convert to Eigen format
    Eigen::Vector2d min_bound(bbox.x, bbox.y);
    Eigen::Vector2d max_bound(bbox.x + bbox.width, bbox.y + bbox.height);

    return {min_bound, max_bound};
}
