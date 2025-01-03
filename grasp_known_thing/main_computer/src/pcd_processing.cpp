// pcd_processing.cpp

#include "pcd_processing.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>   // for decoding base64 into cv::Mat, etc.
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// include ZeroMQ headers
#include <zmq.hpp>
#include <nlohmann/json.hpp>


// OpenGL for display
#include <pcl/point_cloud.h>
#include <GLFW/glfw3.h>    // or your chosen GL headers
#include <GL/glu.h>        // for gluPerspective, gluLookAt, etc.
#include <cmath>

// Namespace aliases
using json = nlohmann::json;

// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

std::vector<uchar> base64_decode(const std::string &encoded) 
{
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    // Prepare an output buffer (we'll push_back into a std::vector<uchar>)
    std::vector<uchar> decoded;
    decoded.reserve(encoded.size() * 3 / 4); // Rough estimate

    // Temporary variables to hold decoding state
    int val = 0;
    int valb = -8;

    for (unsigned char c : encoded) {
        if (c == '=') {
            // '=' padding indicates end
            break;
        }

        // Use the base64_chars index to find out where c is in that string
        int pos = base64_chars.find(c);
        if (pos == (int)std::string::npos) {
            // Skip non-base64 chars or throw an error if you prefer
            continue;
        }

        // Update val with new 6 bits
        val = (val << 6) + pos;
        valb += 6;

        // If we have a byte or more, extract
        if (valb >= 0) {
            decoded.push_back(static_cast<uchar>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }

    return decoded;
}

// Base64 encoding function
std::string base64_encode(const std::vector<uchar>& data) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string encoded;
    int val = 0, valb = -6;
    for (uchar c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) encoded.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (encoded.size() % 4) encoded.push_back('=');
    return encoded;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr createPointCloudFromDepth(
    const cv::Mat& depth_image,
    const cv::Mat& color_image,
    const nlohmann::json& intrinsics_json,
    float depth_scale
)
{
    // Parse intrinsics from JSON
    int width       = intrinsics_json["width"];
    int height      = intrinsics_json["height"];
    float ppx       = intrinsics_json["ppx"];
    float ppy       = intrinsics_json["ppy"];
    float fx        = intrinsics_json["fx"];
    float fy        = intrinsics_json["fy"];
    // Distortion model is intrinsics_json["model"], and
    // intrinsics_json["coeffs"] is a vector<float>, if needed.
    // For a simple example, we ignore distortion or assume minimal.

    // Create a new PCL point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->width    = static_cast<uint32_t>(width);
    cloud->height   = static_cast<uint32_t>(height);
    cloud->is_dense = false;
    cloud->points.resize(width * height);

    // Safety checks in case actual depth_image size differs from intrinsics
    if (depth_image.cols != width || depth_image.rows != height) {
        std::cerr << "[Client] Warning: depth_image size mismatch with intrinsics.\n";
    }
    if (color_image.cols != width || color_image.rows != height) {
        std::cerr << "[Client] Warning: color_image size mismatch with intrinsics.\n";
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Index in the cloud array
            int idx = y * width + x;
            pcl::PointXYZRGB& pt = cloud->points[idx];

            // Retrieve raw depth
            uint16_t depth_val = depth_image.at<uint16_t>(y, x);
            if (depth_val == 0) {
                // No depth, mark as invalid
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
                pt.r = pt.g = pt.b = 0;
                continue;
            }
            float z = depth_val * depth_scale; // Convert to meters if needed

            // Project pixel (x, y) into 3D (X, Y, Z) using intrinsics
            pt.z = z;
            pt.x = (static_cast<float>(x) - ppx) * z / fx;
            pt.y = (static_cast<float>(y) - ppy) * z / fy;

            // Get color from color_image (BGR order in OpenCV)
            cv::Vec3b rgb = color_image.at<cv::Vec3b>(y, x);
            pt.b = rgb[0];
            pt.g = rgb[1];
            pt.r = rgb[2];
        }
    }

    return cloud;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolate_colored_pointcloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud,
    unsigned char target_blue,
    unsigned char target_green,
    unsigned char target_red
)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (auto &pt : input_cloud->points) {
        if ( std::isfinite(pt.z) && 
             pt.b == target_blue &&
             pt.g == target_green &&
             pt.r == target_red )
        {
            isolated->points.push_back(pt);
        }
    }
    isolated->width  = static_cast<uint32_t>(isolated->points.size());
    isolated->height = 1;
    isolated->is_dense = false;
    return isolated;
}
// Helper to decode color & depth from JSON
void decode_frames_from_json(
    const nlohmann::json &frame_json,
    cv::Mat &color_image,
    cv::Mat &depth_image
) {
    // Extract base64-encoded data
    std::string encoded_color = frame_json.at("color_encoded").get<std::string>();
    int color_width = frame_json.at("color_width").get<int>();
    int color_height = frame_json.at("color_height").get<int>();

    std::string encoded_depth = frame_json.at("depth_encoded").get<std::string>();
    int depth_width = frame_json.at("depth_width").get<int>();
    int depth_height = frame_json.at("depth_height").get<int>();

    // Decode color
    std::vector<uchar> color_data = base64_decode(encoded_color);
    color_image = cv::imdecode(color_data, cv::IMREAD_COLOR);
    if (!color_image.empty()) {
        // Ensure it matches the reported width/height (optional check)
        if (color_image.cols != color_width || color_image.rows != color_height) {
            std::cerr << "[Client] Warning: Color image size mismatch.\n";
        }
    }

    // Decode depth
    std::vector<uchar> depth_data = base64_decode(encoded_depth);
    // Depth was sent raw 16-bit. We could decode it from a PNG or treat it as a raw buffer.
    // This depends on how the server sent it. 
    // If it's raw, we can do:
    if (depth_data.size() == (size_t)depth_width * depth_height * 2) {
        depth_image = cv::Mat(depth_height, depth_width, CV_16UC1, depth_data.data()).clone();
    } else {
        std::cerr << "[Client] Depth size mismatch or decoding error.\n";
    }
}




// // Suppose glfw_state is defined somewhere else in your code
// struct glfw_state {
//     float offset_y = 0.0f;
//     float pitch    = 0.0f;
//     float yaw      = 0.0f;

//     // If you still have an OpenGL texture you want to bind, put it here:
//     // texture tex; // For example, if you need it. Otherwise omit.
// };

// Handles all the OpenGL calls needed to display a PCL point cloud
inline void display_pointcloud(
    float width,
    float height,
    glfw_state& app_state,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud
)
{
    // Safety check
    if (!cloud || cloud->empty())
        return;

    // OpenGL: prep screen for point cloud
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    // Background color
    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT);

    // Set up perspective
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60.0, width / height, 0.01, 10.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0,  // Eye/camera position
              0, 0, 1,  // Look at point
              0, -1, 0); // Up vector

    // Basic transformations for user interaction
    glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw,   0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    // Set point size
    glPointSize(width / 640.0f);

    glEnable(GL_DEPTH_TEST);

    // If you don’t need texturing, disable it:
    glDisable(GL_TEXTURE_2D);

    // Begin drawing
    glBegin(GL_POINTS);

    // Loop over PCL points
    for (const auto& pt : cloud->points)
    {
        // Check for valid/finite point
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;

        // Set color from the PCL point (pcl::PointXYZRGB stores r,g,b as uint8)
        glColor3ub(pt.r, pt.g, pt.b);

        // Position in 3D
        glVertex3f(pt.x, pt.y, pt.z);
    }

    glEnd(); // End GL_POINTS

    // Cleanup matrices
    glPopMatrix();           // MODELVIEW
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();           // PROJECTION
    glPopAttrib();           // ATTRIB_BITS
}




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

void run_pointcloud_processing(const std::string& pcd_file_path) {
    // We'll store the final isolated cloud here:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd;

    // Create an OpenGL window for rendering:
    window app(1280, 720, "Pointcloud Client (No RealSense on Client)");
    glfw_state app_state;
    register_glfw_callbacks(app, app_state);

    try {
        // --------------------------------------------------------------------
        // 1. Initialize ZeroMQ client, create TWO sockets:
        //    (A) rs_socket for RealSense server
        //    (B) yolo_socket for YOLO bounding box server
        // --------------------------------------------------------------------
        zmq::context_t context_zmq(1);

        zmq::socket_t rs_socket(context_zmq, zmq::socket_type::req);
        rs_socket.connect("tcp://localhost:6000");
        std::cout << "[Client] Connected to RS-Server on tcp://localhost:6000\n";

        zmq::socket_t yolo_socket(context_zmq, zmq::socket_type::req);
        yolo_socket.connect("tcp://localhost:5555");
        std::cout << "[Client] Connected to YOLO-Server on tcp://localhost:5555\n";

        // --------------------------------------------------------------------
        // 2. Request intrinsics from the RealSense server
        // --------------------------------------------------------------------
        nlohmann::json intrinsics_json;
        {
            std::string intr_request = "get_intrinsics";
            zmq::message_t request_msg(intr_request.size());
            memcpy(request_msg.data(), intr_request.c_str(), intr_request.size());
            rs_socket.send(request_msg, zmq::send_flags::none);

            // Receive intrinsics JSON from RS-Server
            zmq::message_t reply_msg;
            auto res = rs_socket.recv(reply_msg, zmq::recv_flags::none);
            if (!res) {
                throw std::runtime_error("[Client] Failed to receive intrinsics from RS-Server.");
            }
            std::string reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
            intrinsics_json = nlohmann::json::parse(reply_str);

            std::cout << "[Client] Received intrinsics: "
                      << intrinsics_json["width"] << "x" << intrinsics_json["height"]
                      << " fx=" << intrinsics_json["fx"]
                      << " fy=" << intrinsics_json["fy"] << std::endl;
        }

        // If the server sends depth in millimeters, use 0.001f to convert to meters:
        float depth_scale = 0.001f;

        // --------------------------------------------------------------------
        // 3. Main loop: request frames from RS-Server, process them, display
        // --------------------------------------------------------------------
        while (app) {
            // 3A) Ask RealSense server for the next frame
            {
                std::string frame_request = "get_frame";
                zmq::message_t request_msg(frame_request.size());
                memcpy(request_msg.data(), frame_request.c_str(), frame_request.size());
                rs_socket.send(request_msg, zmq::send_flags::none);
            }

            // 3B) Receive color+depth JSON from RS-Server
            cv::Mat color_image, depth_image;
            {
                zmq::message_t reply_msg;
                auto res = rs_socket.recv(reply_msg, zmq::recv_flags::none);
                if (!res) {
                    std::cerr << "[Client] Failed to receive frames from RS-Server.\n";
                    continue;
                }
                std::string reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
                auto frames_json = nlohmann::json::parse(reply_str);

                // Convert from Base64 → cv::Mat
                decode_frames_from_json(frames_json, color_image, depth_image);
                if (color_image.empty() || depth_image.empty()) {
                    std::cerr << "[Client] Invalid frames received.\n";
                    continue;
                }
                cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);

            }

            // 3C) Send color image to the YOLO server for bounding-box detection
            cv::Mat color_image_copy = color_image.clone();
            {
                // Encode color_image_copy as JPEG
                std::vector<uchar> buf;
                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
                if (!cv::imencode(".jpg", color_image_copy, buf, params)) {
                    std::cerr << "[Client] Failed to encode image for YOLO.\n";
                    continue;
                }
                // Base64-encode
                std::string encoded_image = base64_encode(buf);

                // Build JSON with "image_data"
                nlohmann::json request_json;
                request_json["image_data"] = encoded_image;
                std::string request_str = request_json.dump();

                // Send to YOLO server
                zmq::message_t request_msg(request_str.size());
                memcpy(request_msg.data(), request_str.c_str(), request_str.size());
                yolo_socket.send(request_msg, zmq::send_flags::none);
            }

            // 3D) Receive bounding box from YOLO
            nlohmann::json reply_json;
            {
                zmq::message_t reply_msg;
                auto res = yolo_socket.recv(reply_msg, zmq::recv_flags::none);
                if (!res) {
                    std::cerr << "[Client] Failed to receive reply from YOLO server.\n";
                    continue;
                }
                std::string py_reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
                reply_json = nlohmann::json::parse(py_reply_str);
            }

            // 3E) Draw bounding box on color_image_copy
            if (reply_json.contains("bbox")) {
                auto bbox_json = reply_json["bbox"];
                int x_min = bbox_json["x_min"];
                int y_min = bbox_json["y_min"];
                int x_max = bbox_json["x_max"];
                int y_max = bbox_json["y_max"];

                // Clamp coords to image boundaries
                x_min = std::max(0, std::min(x_min, color_image_copy.cols - 1));
                x_max = std::max(0, std::min(x_max, color_image_copy.cols - 1));
                y_min = std::max(0, std::min(y_min, color_image_copy.rows - 1));
                y_max = std::max(0, std::min(y_max, color_image_copy.rows - 1));

                // Draw bounding box
                cv::rectangle(color_image_copy,
                              cv::Point(x_min, y_min),
                              cv::Point(x_max, y_max),
                              cv::Scalar(0, 255, 0), 2);

                // (Optional) fill bounding box with blue
                for (int row = y_min; row <= y_max; ++row) {
                    for (int col = x_min; col <= x_max; ++col) {
                        color_image_copy.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 0, 0);
                    }
                }
            }
            else if (reply_json.contains("error")) {
                std::cerr << "[Client] Error from YOLO server: "
                          << reply_json["error"] << std::endl;
            }
            else {
                std::cerr << "[Client] BBox not found in YOLO reply.\n";
            }

            // 3F) Create a 3D cloud from color+depth using intrinsics
            auto full_cloud = createPointCloudFromDepth(
                depth_image, 
                color_image, 
                intrinsics_json,
                depth_scale 
            );

            // (Optional) Filter points that turned "blue" (B=255,G=0,R=0).
            // You can skip or adapt to your logic.
            unsigned char target_blue  = 255;
            unsigned char target_green = 0;
            unsigned char target_red   = 0;
            isolated_pcd = isolate_colored_pointcloud(
                full_cloud,
                target_blue,
                target_green,
                target_red
            );

            // Show bounding box in a 2D window (OpenCV)
            cv::imshow("Color w/ BBox", color_image_copy);

            // Display the 3D cloud in OpenGL
            display_pointcloud(app.width(), app.height(), app_state, full_cloud);

            // Press 'q' in the OpenCV window to exit
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        // --------------------------------------------------------------------
        // 4. Save the final isolated cloud to PCD
        // --------------------------------------------------------------------
        if (isolated_pcd && !isolated_pcd->points.empty()) {
            pcl::io::savePCDFileASCII(pcd_file_path, *isolated_pcd);
            std::cout << "Saved " << isolated_pcd->points.size()
                      << " points to " << pcd_file_path << std::endl;
        } else {
            std::cerr << "Isolated point cloud is empty. Nothing to save." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in run_pointcloud_processing: " << e.what() << std::endl;
    }
}
