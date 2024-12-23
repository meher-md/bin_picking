# orange_tracker.py

import cv2
import numpy as np
import pyrealsense2 as rs

from helper_functions import (
    create_point_cloud_from_bbox,
    refine_point_cloud,
    keep_largest_cluster,
    get_distance_and_orientation_along_axes,
    SimpleIntrinsics
)

def main():
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
    orange_lower = (5, 100, 100)   # HSV
    orange_upper = (15, 255, 255)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert to HSV
            hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, orange_lower, orange_upper)

            # Optionally, do morphological ops to clean the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Largest contour
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)

                if w > 10 and h > 10:
                    # If bounding box is big enough, we consider it
                    # Draw bounding box on color image
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # We'll build pcd from that bounding box
                    # Extract intrinsics from depth stream
                    intrinsics_data = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                    # We'll use a simple wrapper:
                    fx = intrinsics_data.fx
                    fy = intrinsics_data.fy
                    ppx = intrinsics_data.ppx
                    ppy = intrinsics_data.ppy
                    my_intrinsics = SimpleIntrinsics(fx, fy, ppx, ppy)

                    pcd = create_point_cloud_from_bbox(
                        color_image, depth_image, (x, y, w, h),
                        intrinsics=my_intrinsics
                    )

                    # Refine
                    pcd_refined = refine_point_cloud(pcd, voxel_size=0.01,
                                                     nb_neighbors=20, std_ratio=2.0)
                    pcd_clustered = keep_largest_cluster(pcd_refined, eps=0.02, min_points=20)

            # Compute distance/orientation
            pose_data = get_distance_and_orientation_along_axes(pcd_clustered)
            if pose_data:
                # We’ll split the info into two lines
                info_str_line1 = (f"X={pose_data['distance_x']:.2f}, "
                                f"Y={pose_data['distance_y']:.2f}, "
                                f"Z={pose_data['distance_z']:.2f} (m)")
                
                info_str_line2 = (f"D={pose_data['distance_euclidean']:.2f}m, "
                                f"R={pose_data['roll']:.1f}, "
                                f"P={pose_data['pitch']:.1f}, "
                                f"Y={pose_data['yaw']:.1f} deg")

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
