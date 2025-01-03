import zmq
import json
import cv2
import numpy as np
import base64

def get_bounding_boxes(image):
    """
    Replace this dummy function with your actual bounding box detection logic.
    For demonstration, this function returns a fixed bounding box.
    """
    # Example: Return the center region as the bounding box
    height, width, _ = image.shape
    x_min = width // 4
    y_min = height // 4
    x_max = 3 * width // 4
    y_max = 3 * height // 4
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

def decode_image(encoded_image):
    """
    Decodes a Base64-encoded image string into a cv2 image.
    """
    try:
        binary_image = base64.b64decode(encoded_image)
        nparr = np.frombuffer(binary_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
        return img
    except Exception as e:
        raise ValueError(f"Image decoding error: {e}")


def handle_request(request_json):
    """
    Processes a single request from the client and returns the response.
    """
    try:
        # Extract image data
        encoded_image = request_json.get("image_data", "")
        if not encoded_image:
            return {"error": "No image data received."}

        # Decode the image
        img = decode_image(encoded_image)

        # Perform bounding box detection
        bbox = get_bounding_boxes(img)

        # Prepare the response
        return {"bbox": bbox}

    except Exception as e:
        return {"error": str(e)}

def main():
    # Initialize ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")  # Listen on port 5555

    print("Bounding Box Server is running and listening on port 5555...")

    while True:
        try:
            # Receive the request from the client
            message = socket.recv()
            request_json = json.loads(message.decode())

            # Handle the request
            response = handle_request(request_json)

            # Send the response back to the client
            socket.send_string(json.dumps(response))

        except json.JSONDecodeError:
            error_response = {"error": "Invalid JSON format in request."}
            socket.send_string(json.dumps(error_response))
            print("Error: Invalid JSON format.")
        except Exception as e:
            error_response = {"error": str(e)}
            socket.send_string(json.dumps(error_response))
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
