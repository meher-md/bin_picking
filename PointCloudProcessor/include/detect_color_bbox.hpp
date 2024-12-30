#ifndef DETECT_COLOR_BBOX_HPP
#define DETECT_COLOR_BBOX_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <utility>

// Function to detect a bounding box for a specified color in the image
std::pair<Eigen::Vector2d, Eigen::Vector2d> detect_color_bbox(const cv::Mat& color_image, 
                                                              const cv::Scalar& lower_bound, 
                                                              const cv::Scalar& upper_bound);

#endif // DETECT_COLOR_BBOX_HPP
