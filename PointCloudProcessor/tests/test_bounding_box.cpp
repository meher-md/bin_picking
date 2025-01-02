#include <iostream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>

// Namespace aliases
using json = nlohmann::json;

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: TestBoundingBoxClient <image_path>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];

    try {
        // Initialize ZeroMQ context and socket
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::req);
        socket.connect("tcp://localhost:5555");
        std::cout << "Connected to Bounding Box Server at tcp://localhost:5555" << std::endl;

        // Load image using OpenCV
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image from path: " << image_path << std::endl;
            return 1;
        }
        std::cout << "Loaded image: " << image_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;

        // Encode image as JPEG
        std::vector<uchar> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
        if (!cv::imencode(".jpg", image, buf, params)) {
            std::cerr << "Failed to encode image as JPEG." << std::endl;
            return 1;
        }
        std::cout << "Encoded image as JPEG, size: " << buf.size() << " bytes" << std::endl;

        // Convert to Base64
        std::string encoded_image_str = base64_encode(buf);

        // Create JSON object
        json request_json;
        request_json["image_data"] = encoded_image_str;
        std::string request_str = request_json.dump();
        std::cout << "Sending JSON request: " << request_str.size() << " bytes" << std::endl;

        // Send the request
        zmq::message_t request_msg(request_str.size());
        memcpy(request_msg.data(), request_str.c_str(), request_str.size());
        socket.send(request_msg, zmq::send_flags::none);
        std::cout << "Sent request to server." << std::endl;

        // Receive the reply
        zmq::message_t reply_msg;
        zmq::recv_result_t result = socket.recv(reply_msg, zmq::recv_flags::none);
        if (!result) {
            std::cerr << "Failed to receive reply from server." << std::endl;
            return 1;
        }
        std::string reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
        std::cout << "Received reply (" << reply_msg.size() << " bytes): " << reply_str << std::endl;

        // Parse JSON
        json reply_json = json::parse(reply_str);

        // Extract bounding box
        if (reply_json.contains("bbox")) {
            json bbox = reply_json["bbox"];
            int x_min = bbox["x_min"];
            int y_min = bbox["y_min"];
            int x_max = bbox["x_max"];
            int y_max = bbox["y_max"];
            std::cout << "Received bounding box: x_min=" << x_min
                      << ", y_min=" << y_min
                      << ", x_max=" << x_max
                      << ", y_max=" << y_max << std::endl;

            // Optionally, draw the bounding box on the image and display it
            cv::rectangle(image, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
                          cv::Scalar(0, 255, 0), 2);
            cv::imshow("Bounding Box", image);
            cv::waitKey(0);
        } else {
            std::cerr << "No bounding box found in the reply." << std::endl;
            return 1;
        }

    } catch (const zmq::error_t& e) {
        std::cerr << "ZeroMQ error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
