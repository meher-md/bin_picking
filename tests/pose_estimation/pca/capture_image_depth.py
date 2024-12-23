import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream depth and color frames
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Output directory for saved frames
output_dir = "data/captured_frames"
os.makedirs(output_dir, exist_ok=True)

# Start streaming
pipeline.start(config)

try:
    print("Waiting for the camera to stabilize...")
    for _ in range(30):  # Discard the first 30 frames to allow stabilization
        pipeline.wait_for_frames()

    print("Camera stabilized. Capturing frames...")
    captured_frames = 0

    while captured_frames < 3:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Save RGB image
        rgb_filename = os.path.join(output_dir, f"rgb_frame_{captured_frames + 1}.png")
        cv2.imwrite(rgb_filename, color_image)

        # Save depth map
        depth_filename = os.path.join(output_dir, f"depth_frame_{captured_frames + 1}.png")
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imwrite(depth_filename, depth_colormap)

        print(f"Saved RGB image and depth map {captured_frames + 1}.")

        captured_frames += 1

finally:
    print("Stopping the pipeline.")
    pipeline.stop()
    cv2.destroyAllWindows()