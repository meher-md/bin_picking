import zmq
import json
import base64
import numpy as np
import cv2
from ultralytics import YOLO


# Load the YOLO model
model = YOLO("../assets/best.pt")  # Update the path to your YOLO model file


def get_bounding_boxes(image):
    """
    Perform bounding box detection using YOLO.
    """
    try:
        # Predict with YOLO
        results = model.predict(source=image, conf=0.25)  # Confidence threshold = 0.25
        
        # Extract the first result (assuming a single image)
        result = results[0]

        # Extract bounding boxes
        if result.boxes is None or len(result.boxes) == 0:
            return None  # Return None if no bounding boxes are detected

        # YOLO's result.boxes contains [x_min, y_min, x_max, y_max] coordinates
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Convert to integer values

        # Assuming we're returning the first detected bounding box
        x_min, y_min, x_max, y_max = boxes[0]
        return {"x_min": int(x_min), "y_min": int(y_min), "x_max": int(x_max), "y_max": int(y_max)}

    except Exception as e:
        raise RuntimeError(f"YOLO bounding box detection error: {e}")


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
        if bbox is None:
            return {"error": "No bounding boxes detected."}
        return {"bbox": bbox}

    except Exception as e:
        return {"error": str(e)}


def main():
    # Initialize ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")  # Listen on port 5555

    print("YOLO Bounding Box Server is running and listening on port 5555...")

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
