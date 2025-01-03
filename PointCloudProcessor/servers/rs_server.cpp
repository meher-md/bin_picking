// rs_server.cpp

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <zmq.hpp>  
#include <nlohmann/json.hpp>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>


// Base64 encoding (same as in your existing code)
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

int rs_server() {
    try {
        // Initialize ZeroMQ context and socket
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::rep);
        // Bind to a port (e.g. 6000) so that a client can connect
        socket.bind("tcp://*:6000");
        std::cout << "[RS-Server] Bound to tcp://*:6000" << std::endl;

        // Initialize RealSense pipeline
        rs2::pipeline pipe;
        auto profile = pipe.start();

        // Get intrinsics (example: from depth stream)
        auto depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        rs2_intrinsics depth_intrinsics = depth_stream_profile.get_intrinsics();

        // For demonstration: We'll send intrinsics once on startup to the client.
        // Alternatively, you can send them every loop or on demand.
        {
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

            // Wait for a client to request intrinsics
            std::cout << "[RS-Server] Waiting for intrinsics request..." << std::endl;
            zmq::message_t request_msg;
            // This call will block until a client sends a request
            socket.recv(request_msg, zmq::recv_flags::none);

            // Once received, send intrinsics in JSON form
            std::string intrinsics_str = intrinsics_json.dump();
            zmq::message_t reply_intrinsics(intrinsics_str.size());
            memcpy(reply_intrinsics.data(), intrinsics_str.c_str(), intrinsics_str.size());
            socket.send(reply_intrinsics, zmq::send_flags::none);
            std::cout << "[RS-Server] Intrinsics sent" << std::endl;
        }

        // Now, keep sending frames in a loop
        while (true) {
            // Wait for a 'frame request' from client
            zmq::message_t request_msg;
            socket.recv(request_msg, zmq::recv_flags::none);
            // We won't parse the message body here, but you could check it if needed.

            // Capture frames
            rs2::frameset frameset = pipe.wait_for_frames();
            rs2::video_frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();

            // Convert color_frame to OpenCV for potential encoding
            cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()),
                                CV_8UC3,
                                (void*)color_frame.get_data(),
                                cv::Mat::AUTO_STEP);
            // Optional BGR -> RGB flip or not, depending on your usage
            // cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);

            // Encode color image as JPEG + base64
            std::vector<uchar> buf;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
            if (!cv::imencode(".jpg", color_image, buf, params)) {
                std::cerr << "[RS-Server] Failed to encode color image." << std::endl;
                continue;
            }
            std::string encoded_color = base64_encode(buf);

            // Convert depth_frame to something encodable (16-bit PNG for instance).
            // For demonstration: just send raw data as base64
            // Depth is typically 16 bits, so we can encode as PNG or just send raw
            int depth_width = depth_frame.get_width();
            int depth_height = depth_frame.get_height();
            int depth_bpp = 2; // RS depth is 16 bits
            size_t depth_size = depth_width * depth_height * depth_bpp;
            const unsigned char* depth_data = static_cast<const unsigned char*>(depth_frame.get_data());

            // Convert to vector<uchar> for base64
            std::vector<uchar> depth_raw(depth_data, depth_data + depth_size);
            std::string encoded_depth = base64_encode(depth_raw);

            // Create JSON to send
            nlohmann::json reply_json;
            reply_json["color_width"]   = color_frame.get_width();
            reply_json["color_height"]  = color_frame.get_height();
            reply_json["color_encoded"] = encoded_color;

            reply_json["depth_width"]   = depth_width;
            reply_json["depth_height"]  = depth_height;
            reply_json["depth_encoded"] = encoded_depth;

            // Convert JSON to string
            std::string reply_str = reply_json.dump();

            // Send it
            zmq::message_t reply_msg(reply_str.size());
            memcpy(reply_msg.data(), reply_str.c_str(), reply_str.size());
            socket.send(reply_msg, zmq::send_flags::none);
            std::cout << "[RS-Server] Sent frames to client." << std::endl;

            // Sleep just to avoid spamming the loop too fast (optional)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

    } catch (const std::exception &e) {
        std::cerr << "[RS-Server] Exception: " << e.what() << std::endl;
    }
    return 0;
}

int main() {
    return rs_server();  // Run the server
}