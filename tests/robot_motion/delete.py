import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualize_pcd_with_realsense():
    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create a point cloud object
    pc = rs.pointcloud()
    colorizer = rs.colorizer()

    try:
        while True:
            # Get frameset of depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Generate point cloud
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)

            # Convert point cloud to numpy array
            vertices = np.asanyarray(points.get_vertices())  # (N, 3)
            colors = np.asanyarray(points.get_texture_coordinates())  # (N, 2)

            # Visualize the colorized depth image
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())
            combined_image = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense Point Cloud", combined_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_pcd_with_realsense()
