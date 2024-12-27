import pyrealsense2 as rs
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from helper_functions import keep_largest_cluster

def filter_and_keep_largest_cluster(x_coords, y_coords, z_coords):
    """
    Apply clustering to keep the largest cluster of points.

    Parameters:
    -----------
    x_coords : ndarray
        X-coordinates of the point cloud.
    y_coords : ndarray
        Y-coordinates of the point cloud.
    z_coords : ndarray
        Z-coordinates of the point cloud.

    Returns:
    --------
    x_filtered, y_filtered, z_filtered : ndarray
        Coordinates of the largest cluster.
    """
    points = np.stack((x_coords, y_coords, z_coords), axis=1)
    points = points[z_coords > 0]  # Remove points with zero depth
    
    # Convert points to an Open3D point cloud
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Keep the largest cluster
    largest_cluster_pcd = keep_largest_cluster(pcd, eps=0.02, min_points=20)

    # Extract filtered points
    filtered_points = np.asarray(largest_cluster_pcd.points)
    return filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

def main():
    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # High-resolution depth
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # High-resolution color

    # Start streaming
    pipeline.start(config)

    # Warm up
    for _ in range(30):
        pipeline.wait_for_frames()

    # For color threshold (roughly for "orange")
    orange_lower = (0, 100, 100)
    orange_upper = (10, 255, 255)

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert color image to HSV and create a mask
            hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, orange_lower, orange_upper)

            # Morphological operations to clean the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour and bounding box
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)

                if w > 10 and h > 10:
                    # Draw bounding box on the color image
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Extract the depth data within the bounding box
                    depth_crop = depth_image[y:y + h, x:x + w]

                    # Validate the depth crop
                    if depth_crop.size == 0:
                        print("Depth crop is empty. Skipping visualization.")
                        continue

                    # Replace zeros with a small constant to avoid issues
                    depth_crop[depth_crop == 0] = 1

                    # Apply an outlier filter (median filter)
                    filtered_depth_crop = median_filter(depth_crop, size=3)

                    # Normalize the filtered depth crop for visualization
                    depth_display = cv2.normalize(filtered_depth_crop, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # Apply a colormap for visualization
                    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_TURBO)

                    # Show depth within the bounding box
                    cv2.imshow("Depth Crop (Filtered)", depth_colormap)

                    # Create 3D scatter plot with downsampling
                    ax.cla()  # Clear the previous plot
                    rows, cols = filtered_depth_crop.shape

                    # Downsample the coordinates for better visualization
                    downsample_factor = 2
                    x_coords, y_coords = np.meshgrid(
                        np.arange(0, cols, downsample_factor),
                        np.arange(0, rows, downsample_factor)
                    )
                    z_coords = filtered_depth_crop[::downsample_factor, ::downsample_factor]

                    # Flatten the arrays for clustering
                    x_coords = x_coords.flatten()
                    y_coords = y_coords.flatten()
                    z_coords = z_coords.flatten()

                    # Keep the largest cluster
                    x_filtered, y_filtered, z_filtered = filter_and_keep_largest_cluster(x_coords, y_coords, z_coords)

                    # Plot the filtered data
                    ax.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='jet', s=5)
                    ax.set_title("3D Depth Map (Filtered and Clustered)")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Depth")
                    plt.pause(0.01)  # Update the plot in real-time

            # Display the color image with bounding box
            cv2.imshow("Orange Object Tracking", color_image)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
