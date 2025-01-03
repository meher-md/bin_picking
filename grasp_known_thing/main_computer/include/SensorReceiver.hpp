// SensorReceiver.hpp
#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class SensorReceiver {
public:
    SensorReceiver(const std::string& rs_server_addr,
                   const std::string& yolo_server_addr);

    void requestIntrinsics();
    nlohmann::json getIntrinsicsJson() const { return m_intrinsicsJson; }

    // Request a new frame from the Pi server (color & depth)
    // and also get YOLO bounding box from the YOLO server
    bool requestFrame(cv::Mat &color_image, 
                      cv::Mat &depth_image,
                      nlohmann::json &bbox_json);

private:
    std::string m_rsServerAddr;
    std::string m_yoloServerAddr;

    nlohmann::json m_intrinsicsJson;

    // ZeroMQ context, sockets, etc.
    // We'll keep them as members for repeated calls
    void* m_context;  // Or use zmq::context_t if you prefer C++ wrappers
    void* m_rsSocket;
    void* m_yoloSocket;
};

