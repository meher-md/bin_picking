from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2


def initialize_realsense():
    """
    Initialize and configure the RealSense camera pipeline.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable RGB stream
    pipeline.start(config)
    print("RealSense camera started.")
    return pipeline


def main():
    # 1. Load the YOLO model (must be a YOLOv8 segmentation .pt file)
    model_path = "data/best.pt"  # Update to your .pt file path
    model = YOLO(model_path)
    print("Model loaded successfully.")

    # 2. Initialize the RealSense camera
    pipeline = initialize_realsense()

    try:
        print("Press 'q' to quit.")
        while True:
            # Get frames from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense color frame to a NumPy array (BGR by default)
            bgr_image = np.asanyarray(color_frame.get_data())

            # 3. Predict with confidence threshold (e.g., conf=0.25)
            results = model.predict(source=bgr_image, conf=0.25)  # Accepts np.ndarray

            # 4. Retrieve the first result (assuming a single image per inference)
            result = results[0]

            # If no masks found, just show the original frame
            if result.masks is None or len(result.masks.data) == 0:
                cv2.imshow("Segmentation Overlay", bgr_image)
            else:
                # result.masks.data is a list of per-object binary masks
                # Combine them into one overlay or handle individually
                combined_mask = np.zeros((bgr_image.shape[0], bgr_image.shape[1]), dtype=np.uint8)

                # Each mask in result.masks.data is a torch.Tensor of shape (H, W)
                for mask_tensor in result.masks.data:
                    # Convert to NumPy (0/1)
                    mask = mask_tensor.cpu().numpy().astype(np.uint8)
                    # Combine (logical OR)
                    combined_mask = np.bitwise_or(combined_mask, mask)

                # Create a color overlay from the combined mask
                overlay = np.zeros_like(bgr_image)
                overlay[combined_mask == 1] = [0, 255, 0]  # Green color for segmented regions

                # Blend
                highlighted_image = cv2.addWeighted(bgr_image, 0.7, overlay, 0.3, 0)
                cv2.imshow("Segmentation Overlay", highlighted_image)

            # Show the live BGR image (optional)
            cv2.imshow("RGB Image", bgr_image)

            # Quit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera and display windows closed.")


if __name__ == "__main__":
    main()
