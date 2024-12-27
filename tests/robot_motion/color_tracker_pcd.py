# orange_tracker.py

import cv2
import numpy as np
import pyrealsense2 as rs
from pcd_plotter import setup_matplotlib_visualizer, update_partial_cloud
import open3d as o3d

from helper_functions import (
    create_point_cloud_from_bbox,
    refine_point_cloud,
    keep_largest_cluster,
    get_distance_and_orientation_along_axes,
    load_reference_point_cloud,
    register_point_cloud_to_reference,
    rotation_matrix_to_euler_angles,    
    SimpleIntrinsics
)

DEBUG_PCD_CAPTURE = True
def main():
    if DEBUG_PCD_CAPTURE:
        fig,ax = setup_matplotlib_visualizer()

    # Load the reference point cloud
    reference_pcd = load_reference_point_cloud("data/output.xyz")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Warm up
    for _ in range(30):
        pipeline.wait_for_frames()

    # For color threshold (roughly for "orange")
    # Adjust as needed for your lighting conditions
    # orange_lower = (5, 100, 100)   # HSV
    # orange_upper = (15, 255, 255)
    orange_lower = (0, 100, 100)
    orange_upper = (10, 255, 255)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert to HSV and create a mask for the target color
            hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, orange_lower, orange_upper)

            # Optionally, clean the mask using morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)

                if w > 10 and h > 10:
                    # Draw bounding box on the color image
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Extract intrinsics from the depth stream
                    intrinsics_data = depth_frame.profile.as_video_stream_profile().get_intrinsics()

                    # Pass the correct RealSense intrinsics directly
                    pcd = create_point_cloud_from_bbox(
                        color_image, depth_image, (x, y, w, h),
                        intrinsics=intrinsics_data  # Use RealSense intrinsics directly
                    )

                    # Refine
                    # pcd_refined = refine_point_cloud(pcd, voxel_size=0.01,
                    #                                  nb_neighbors=20, std_ratio=2.0)
                    pcd_clustered = keep_largest_cluster(pcd, eps=0.02, min_points=20)

                    if pcd_clustered:
                        if DEBUG_PCD_CAPTURE:
                            #update_partial_cloud(ax, test_pcd)
                            update_partial_cloud(ax,pcd_clustered)
                    else:
                        print("No PCD Data")
                    # Align to reference point cloud
                    transformation, fitness = register_point_cloud_to_reference(pcd_clustered, reference_pcd)

                    # Extract orientation
                    R = transformation[:3, :3]
                    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)

                    # Extract translation
                    tx, ty, tz = transformation[:3, 3]

                    # Overlay results
                    info_str_line1 = (f"X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f} (m)")
                    info_str_line2 = (f"Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f} deg")

                    # Draw the first line a bit above the bounding box
                    cv2.putText(color_image, info_str_line1, (x, y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw the second line slightly below the first line
                    cv2.putText(color_image, info_str_line2, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Display
            cv2.imshow("Orange Object Tracking", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
