// SensorReceiver.cpp
#include "SensorReceiver.hpp"

#include <zmq.hpp>
#include <stdexcept>
#include <iostream>
#include "PclUtils.hpp"     // For base64_encode / decode_frames_from_json
#include <nlohmann/json.hpp>

using json = nlohmann::json;

SensorReceiver::SensorReceiver(const std::string& rs_server_addr,
                               const std::string& yolo_server_addr)
    : m_rsServerAddr(rs_server_addr),
      m_yoloServerAddr(yolo_server_addr)
{
    // 1. Initialize ZeroMQ context & sockets
    auto context = new zmq::context_t(1);
    m_context = context;

    // RealSense socket
    auto rs_socket = new zmq::socket_t(*context, zmq::socket_type::req);
    rs_socket->connect(m_rsServerAddr);
    m_rsSocket = rs_socket;

    // YOLO socket
    auto yolo_socket = new zmq::socket_t(*context, zmq::socket_type::req);
    yolo_socket->connect(m_yoloServerAddr);
    m_yoloSocket = yolo_socket;
}

void SensorReceiver::requestIntrinsics() {
    // 2. Request intrinsics from the RealSense server
    std::string intr_request = "get_intrinsics";
    zmq::message_t req_msg(intr_request.size());
    memcpy(req_msg.data(), intr_request.c_str(), intr_request.size());
    static_cast<zmq::socket_t*>(m_rsSocket)->send(req_msg, zmq::send_flags::none);

    zmq::message_t reply_msg;
    auto res = static_cast<zmq::socket_t*>(m_rsSocket)->recv(reply_msg, zmq::recv_flags::none);
    if (!res.has_value()) {
        throw std::runtime_error("[SensorReceiver] Failed to receive intrinsics from RS-Server.");
    }
    std::string reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
    m_intrinsicsJson = json::parse(reply_str);

    std::cout << "[SensorReceiver] Intrinsics: "
              << m_intrinsicsJson["width"] << "x" << m_intrinsicsJson["height"]
              << ", fx=" << m_intrinsicsJson["fx"]
              << ", fy=" << m_intrinsicsJson["fy"] << std::endl;
}

bool SensorReceiver::requestFrame(cv::Mat &color_image, 
                                  cv::Mat &depth_image,
                                  json &bbox_json)
{
    // 1. Ask RealSense server for the next frame
    {
        std::string frame_request = "get_frame";
        zmq::message_t request_msg(frame_request.size());
        memcpy(request_msg.data(), frame_request.c_str(), frame_request.size());
        static_cast<zmq::socket_t*>(m_rsSocket)->send(request_msg, zmq::send_flags::none);
    }

    // 2. Receive color+depth JSON
    {
        zmq::message_t reply_msg;
        auto res = static_cast<zmq::socket_t*>(m_rsSocket)->recv(reply_msg, zmq::recv_flags::none);
        if (!res.has_value()) {
            std::cerr << "[SensorReceiver] Failed to receive frames from RS-Server.\n";
            return false;
        }

        std::string reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
        auto frames_json = json::parse(reply_str);

        // Convert from Base64 → cv::Mat
        if (!PclUtils::decodeFramesFromJson(frames_json, color_image, depth_image)) {
            std::cerr << "[SensorReceiver] Invalid frames received or decode error.\n";
            return false;
        }
    }

    // 3. Send color image to YOLO server
    {
        // Encode color_image as JPEG
        std::vector<uchar> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
        if (!cv::imencode(".jpg", color_image, buf, params)) {
            std::cerr << "[SensorReceiver] Failed to encode image for YOLO.\n";
            return false;
        }

        // Base64-encode
        std::string encoded_image = PclUtils::base64Encode(buf);

        // Build JSON with "image_data"
        json request_json;
        request_json["image_data"] = encoded_image;
        std::string request_str = request_json.dump();

        // Send to YOLO server
        zmq::message_t request_msg(request_str.size());
        memcpy(request_msg.data(), request_str.c_str(), request_str.size());
        static_cast<zmq::socket_t*>(m_yoloSocket)->send(request_msg, zmq::send_flags::none);
    }

    // 4. Receive bounding box from YOLO
    {
        zmq::message_t reply_msg;
        auto res = static_cast<zmq::socket_t*>(m_yoloSocket)->recv(reply_msg, zmq::recv_flags::none);
        if (!res.has_value()) {
            std::cerr << "[SensorReceiver] Failed to receive YOLO reply.\n";
            return false;
        }

        std::string py_reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
        auto yolo_reply_json = json::parse(py_reply_str);

        if (yolo_reply_json.contains("bbox")) {
            bbox_json = yolo_reply_json["bbox"];
        }
        else if (yolo_reply_json.contains("error")) {
            std::cerr << "[SensorReceiver] YOLO error: " << yolo_reply_json["error"] << "\n";
            return false;
        }
        else {
            std::cerr << "[SensorReceiver] BBox not found in YOLO reply.\n";
            return false;
        }
    }

    return true;
}
