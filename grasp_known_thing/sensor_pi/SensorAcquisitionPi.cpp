// SensorAcquisitionPi.cpp

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <zmq.hpp>  
#include <nlohmann/json.hpp>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

// Base64 encoding function (unchanged)
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
    if (valb > -6) 
        encoded.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (encoded.size() % 4) 
        encoded.push_back('=');
    return encoded;
}

int SensorAcquisitionPi() {
    try {
        // Initialize ZeroMQ context and REP socket
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::rep);
        socket.bind("tcp://*:6000");
        std::cout << "[RS-Server] Bound to tcp://*:6000" << std::endl;

        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        auto profile = pipe.start();

        // Retrieve intrinsics from the depth stream
        auto depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        rs2_intrinsics depth_intrinsics = depth_stream_profile.get_intrinsics();

        // Prepare intrinsics JSON once, since intrinsics typically don't change
        nlohmann::json intrinsics_json;
        intrinsics_json["width"] = depth_intrinsics.width;
        intrinsics_json["height"] = depth_intrinsics.height;
        intrinsics_json["ppx"] = depth_intrinsics.ppx;
        intrinsics_json["ppy"] = depth_intrinsics.ppy;
        intrinsics_json["fx"]  = depth_intrinsics.fx;
        intrinsics_json["fy"]  = depth_intrinsics.fy;
        intrinsics_json["model"] = depth_intrinsics.model;
        intrinsics_json["coeffs"] = std::vector<float>(std::begin(depth_intrinsics.coeffs),
                                                       std::end(depth_intrinsics.coeffs));

        std::cout << "[RS-Server] Ready to handle requests." << std::endl;

        // Main loop to handle incoming requests
        while (true) {
            zmq::message_t request_msg;
            // Receive a request from the client
            zmq::recv_result_t received = socket.recv(request_msg, zmq::recv_flags::none);
            if (!received.has_value()) {
                std::cerr << "[RS-Server] Failed to receive message." << std::endl;
                // Optionally, send an error message back to the client
                nlohmann::json error_json;
                error_json["error"] = "Failed to receive message.";
                std::string error_str = error_json.dump();
                zmq::message_t error_reply(error_str.size());
                memcpy(error_reply.data(), error_str.c_str(), error_str.size());
                socket.send(error_reply, zmq::send_flags::none);
                continue; // Continue to wait for the next message
            }

            // Convert the received message to a string
            std::string request_str(static_cast<char*>(request_msg.data()), request_msg.size());

            // Debug Logging: Print the received request and its size
            std::cout << "[RS-Server] Received request: '" << request_str 
                      << "' with size " << request_str.size() << std::endl;

            // Determine the type of request
            if (request_str == "get_intrinsics") {
                // Handle intrinsics request
                std::string intrinsics_str = intrinsics_json.dump();
                zmq::message_t reply_intrinsics(intrinsics_str.size());
                memcpy(reply_intrinsics.data(), intrinsics_str.c_str(), intrinsics_str.size());
                socket.send(reply_intrinsics, zmq::send_flags::none);
                std::cout << "[RS-Server] Intrinsics sent." << std::endl;
            }
            else if (request_str == "get_frame") {
                // Handle frame request
                // Capture frames from RealSense
                rs2::frameset frameset = pipe.wait_for_frames();
                rs2::video_frame color_frame = frameset.get_color_frame();
                rs2::depth_frame depth_frame = frameset.get_depth_frame();

                // Convert color_frame to OpenCV Mat
                cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()),
                                    CV_8UC3,
                                    (void*)color_frame.get_data(),
                                    cv::Mat::AUTO_STEP);
                // Optional: Convert BGR to RGB if needed
                // cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);

                // Encode color image as JPEG
                std::vector<uchar> buf;
                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
                if (!cv::imencode(".jpg", color_image, buf, params)) {
                    std::cerr << "[RS-Server] Failed to encode color image." << std::endl;
                    // Send an error response to the client
                    nlohmann::json error_json;
                    error_json["error"] = "Failed to encode color image.";
                    std::string error_str = error_json.dump();
                    zmq::message_t error_reply(error_str.size());
                    memcpy(error_reply.data(), error_str.c_str(), error_str.size());
                    socket.send(error_reply, zmq::send_flags::none);
                    continue; // Skip sending frame data
                }
                std::string encoded_color = base64_encode(buf);

                // Handle depth data: Encode raw depth as base64
                int depth_width = depth_frame.get_width();
                int depth_height = depth_frame.get_height();
                int depth_bpp = 2; // 16 bits per pixel
                size_t depth_size = depth_width * depth_height * depth_bpp;
                const unsigned char* depth_data = static_cast<const unsigned char*>(depth_frame.get_data());

                // Convert depth data to vector<uchar>
                std::vector<uchar> depth_raw(depth_data, depth_data + depth_size);
                std::string encoded_depth = base64_encode(depth_raw);

                // Create JSON payload
                nlohmann::json reply_json;
                reply_json["color_width"]   = color_frame.get_width();
                reply_json["color_height"]  = color_frame.get_height();
                reply_json["color_encoded"] = encoded_color;

                reply_json["depth_width"]   = depth_width;
                reply_json["depth_height"]  = depth_height;
                reply_json["depth_encoded"] = encoded_depth;

                // Serialize JSON to string
                std::string reply_str = reply_json.dump();

                // Send the frame data back to the client
                zmq::message_t reply_msg(reply_str.size());
                memcpy(reply_msg.data(), reply_str.c_str(), reply_str.size());
                socket.send(reply_msg, zmq::send_flags::none);
                std::cout << "[RS-Server] Sent frames to client." << std::endl;

                // Optional: Sleep to control frame rate
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            else {
                // Handle unknown request types
                std::cerr << "[RS-Server] Received unknown request: " << request_str << std::endl;
                nlohmann::json error_json;
                error_json["error"] = "Unknown request type.";
                std::string error_str = error_json.dump();
                zmq::message_t error_reply(error_str.size());
                memcpy(error_reply.data(), error_str.c_str(), error_str.size());
                socket.send(error_reply, zmq::send_flags::none);
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "[RS-Server] Exception: " << e.what() << std::endl;
    }
    return 0;
}

int main() {
    return SensorAcquisitionPi();  // Run the server
}
