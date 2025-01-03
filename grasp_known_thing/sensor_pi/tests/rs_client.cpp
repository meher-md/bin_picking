#include <iostream>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

int main() {
    try {
        // Initialize ZeroMQ context and socket
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::req);

        // Connect to the server
        socket.connect("tcp://localhost:6000");
        std::cout << "[Client] Connected to server at tcp://localhost:6000" << std::endl;

        // Send a request for intrinsics
        std::string request = "GET_INTRINSICS";
        zmq::message_t request_msg(request.size());
        memcpy(request_msg.data(), request.c_str(), request.size());
        socket.send(request_msg, zmq::send_flags::none);

        // Receive the intrinsics response
        zmq::message_t reply_msg;
        socket.recv(reply_msg, zmq::recv_flags::none);
        std::string reply_str(static_cast<char*>(reply_msg.data()), reply_msg.size());
        nlohmann::json intrinsics = nlohmann::json::parse(reply_str);
        std::cout << "[Client] Received intrinsics: " << intrinsics.dump(4) << std::endl;

        // Send a frame request
        request = "GET_FRAME";
        request_msg = zmq::message_t(request.size());
        memcpy(request_msg.data(), request.c_str(), request.size());
        socket.send(request_msg, zmq::send_flags::none);

        // Receive the frame response
        socket.recv(reply_msg, zmq::recv_flags::none);
        reply_str = std::string(static_cast<char*>(reply_msg.data()), reply_msg.size());
        nlohmann::json frame_data = nlohmann::json::parse(reply_str);
        std::cout << "[Client] Received frame data: " << frame_data.dump(4) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Client] Exception: " << e.what() << std::endl;
    }

    return 0;
}
