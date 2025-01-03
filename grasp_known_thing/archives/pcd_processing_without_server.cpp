// pcd_processing.cpp

#include "pcd_processing.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "example.hpp"          // Include short list of convenience functions for rendering

// include ZeroMQ headers
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <tuple>
#include <algorithm>

// Namespace aliases
using json = nlohmann::json;


// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

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


// Function to isolate the colored point cloud and return PCL PCD
pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolate_colored_pointcloud(
    float width, 
    float height, 
    rs2::points& points, 
    const rs2::video_frame& color_frame, 
    unsigned char target_blue, 
    unsigned char target_green, 
    unsigned char target_red
) {
    if (!points) {
        std::cerr << "No points to process." << std::endl;
        return nullptr;
    }

    // Create a PCL point cloud to store the isolated points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd(new pcl::PointCloud<pcl::PointXYZRGB>());


    // Render and collect points with the specified color
    auto vertices = points.get_vertices();              // Get vertices
    auto tex_coords = points.get_texture_coordinates(); // Get texture coordinates
    const unsigned char* color_data = static_cast<const unsigned char*>(color_frame.get_data());
    int stride = color_frame.get_stride_in_bytes();

    for (int i = 0; i < points.size(); i++) {
        if (vertices[i].z) { // Only consider valid depth points
            // Map texture coordinates to image coordinates
            int x = static_cast<int>(tex_coords[i].u * color_frame.get_width());
            int y = static_cast<int>(tex_coords[i].v * color_frame.get_height());

            if (x >= 0 && y >= 0 && x < color_frame.get_width() && y < color_frame.get_height()) {
                // Get the color at the mapped texture coordinates
                int index = y * stride + x * 3; // 3 channels (BGR)
                unsigned char blue = color_data[index];
                unsigned char green = color_data[index + 1];
                unsigned char red = color_data[index + 2];

                // Check if the color matches the target color
                if (blue == target_blue && green == target_green && red == target_red) {
                    glVertex3fv(reinterpret_cast<const GLfloat*>(&vertices[i]));  // Render the point
                    glTexCoord2f(tex_coords[i].u, tex_coords[i].v); // Upload texture coordinate

                    // Add the point to the PCL point cloud
                    pcl::PointXYZRGB pcl_point;
                    pcl_point.x = vertices[i].x;
                    pcl_point.y = vertices[i].y;
                    pcl_point.z = vertices[i].z;
                    pcl_point.r = red;
                    pcl_point.g = green;
                    pcl_point.b = blue;
                    isolated_pcd->points.push_back(pcl_point);
                }
            }
        }
    }

    // Set the point cloud properties
    isolated_pcd->width = static_cast<uint32_t>(isolated_pcd->points.size());
    isolated_pcd->height = 1;
    isolated_pcd->is_dense = false;

    return isolated_pcd;
}

// Function to run the point cloud processing and save the PCD file

void run_pointcloud_processing(const std::string& pcd_file_path) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_pcd;

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Pointcloud Example");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    try {
        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        auto profile = pipe.start();

        // Get depth stream intrinsics
        auto depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = depth_stream_profile.get_intrinsics();

        // Define bounding box color range (example: orange in HSV)
        cv::Scalar lower_orange(0, 0, 0);
        cv::Scalar upper_orange(20, 20, 20);


        // Initialize ZeroMQ context and socket
        zmq::context_t context_zmq(1);
        zmq::socket_t socket(context_zmq, zmq::socket_type::req);
        // Connect to the Python server
        socket.connect("tcp://localhost:5555");

        while (app) {
            // Capture frames from RealSense
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::video_frame color_frame = frames.get_color_frame();
            // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
            if (!color_frame)
                color_frame = frames.get_infrared_frame();
            rs2::depth_frame depth_frame = frames.get_depth_frame();

            // Convert color frame to OpenCV Mat
            cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3,
                                (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            if (color_image.empty()) {
                throw std::runtime_error("Captured frame is empty.");
            }

            // Replace the bounding box detection section with ZeroMQ IPC
            // ------------------------------------------------------------

            // Clone the image (if needed)
            cv::Mat color_image_copy = color_image.clone();
            // Uncomment if conversion is necessary
            // cv::cvtColor(color_image_copy, color_image_copy, cv::COLOR_RGB2BGR);

            // Encode the image as JPEG
            std::vector<uchar> buf;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
            if (!cv::imencode(".jpg", color_image_copy, buf, params)) {
                std::cerr << "Failed to encode image as JPEG." << std::endl;
                continue;
            }

            // Convert to Base64
            std::string encoded_image = base64_encode(buf);

            // Create JSON request
            json request_json;
            request_json["image_data"] = encoded_image;
            std::string request_str = request_json.dump();

            // Send the request
            zmq::message_t request(request_str.size());
            memcpy(request.data(), request_str.c_str(), request_str.size());
            socket.send(request, zmq::send_flags::none);

            // Receive the reply from the Python server
            zmq::message_t reply;
            auto result = socket.recv(reply, zmq::recv_flags::none);

            // Handle the receive result
            if (!result) {
                std::cerr << "Failed to receive reply from the Python server." << std::endl;
                continue; // Skip processing this frame or handle as needed
            }

            // Deserialize the reply
            std::string reply_str(static_cast<char*>(reply.data()), reply.size());
            json reply_json = json::parse(reply_str);

            // Extract bounding box
            if (reply_json.contains("bbox")) {
                json bbox_json = reply_json["bbox"];
                int x_min = bbox_json["x_min"];
                int y_min = bbox_json["y_min"];
                int x_max = bbox_json["x_max"];
                int y_max = bbox_json["y_max"];

                // Draw bounding box on the image for visualization
                cv::rectangle(color_image_copy,
                            cv::Point(x_min, y_min),
                            cv::Point(x_max, y_max),
                            cv::Scalar(0, 255, 0), 2);

                // Get color frame dimensions and data pointer
                int width = color_frame.get_width();
                int height = color_frame.get_height();
                int stride = color_frame.get_stride_in_bytes();
                unsigned char* data = (unsigned char*)color_frame.get_data();

                // Ensure the coordinates are within the frame bounds
                x_min = std::max(0, x_min);
                y_min = std::max(0, y_min);
                x_max = std::min(width - 1, x_max);
                y_max = std::min(height - 1, y_max);

                // Modify pixel values in the bounding box to blue
                for (int y = y_min; y <= y_max; ++y) {
                    for (int x = x_min; x <= x_max; ++x) {
                        int index = y * stride + x * 3; // Assuming 3 bytes per pixel (BGR)
                        data[index] = 255;   // Blue channel
                        data[index + 1] = 0; // Green channel
                        data[index + 2] = 0; // Red channel
                    }
                }
            } else if (reply_json.contains("error")) {
                std::cerr << "Error from Python server: " << reply_json["error"] << std::endl;
            } else {
                std::cerr << "Bounding box not found in the reply." << std::endl;
            }

            // ------------------------------------------------------------

            // Generate point cloud from depth frame
            rs2::pointcloud pc;
            rs2::points points;
            pc.map_to(color_frame);
            points = pc.calculate(depth_frame);

            // Render point clouds
            app_state.tex.upload(color_frame);

            unsigned char target_blue = 255; // Blue channel
            unsigned char target_green = 0;  // Green channel
            unsigned char target_red = 0;    // Red channel

            isolated_pcd = isolate_colored_pointcloud(800, 600, points, color_frame, target_blue, target_green, target_red);

            // Optionally, implement and call a function like draw_colored_pointcloud if needed
            draw_pointcloud(app.width(), app.height(), app_state, points);

            // Exit loop if 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        // Save the isolated point cloud to a PCD file
        if (isolated_pcd && !isolated_pcd->points.empty()) {
            pcl::io::savePCDFileASCII(pcd_file_path, *isolated_pcd);
            std::cout << "Saved " << isolated_pcd->points.size() << " points to " << pcd_file_path << std::endl;
        } else {
            std::cerr << "Isolated point cloud is empty. Nothing to save." << std::endl;
        }

        // Cleanup GLFW
        // glfwDestroyWindow(window);
        // glfwTerminate();
    }catch (const std::exception& e) {
        std::cerr << "Error in run_pointcloud_processing: " << e.what() << std::endl;
    }
}
