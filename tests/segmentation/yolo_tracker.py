# yolo_tracker_refactored.py

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from helper_functions import (
    create_point_cloud_from_bbox,
    refine_point_cloud,
    keep_largest_cluster,
    get_distance_and_orientation_along_axes,
    SimpleIntrinsics
)

def main():
    # 1. Initialize YOLO model
    yolo_model = YOLO("data/best.pt")
    confidence_threshold = 0.25  # Adjust as needed

    # 2. Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Warm up the camera
    for _ in range(30):
        pipeline.wait_for_frames()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            try:
                # 3. Use YOLO to detect objects in the color frame
                results = yolo_model.predict(source=color_image, conf=confidence_threshold)
                detections = results[0].boxes if results else None

                if not detections or len(detections) == 0:
                    cv2.imshow("YOLO Object Tracking", color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # 4. Pick one bounding box with the highest confidence
                best_box = None
                best_conf = 0.0
                for det in detections:
                    try:
                        conf = float(det.conf[0])  # confidence
                        if conf > best_conf:
                            best_conf = conf
                            best_box = det.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue

                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    if w > 10 and h > 10:
                        # Draw bounding box
                        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        try:
                            # 5. Extract depth intrinsics
                            intrinsics_data = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                            fx, fy = intrinsics_data.fx, intrinsics_data.fy
                            ppx, ppy = intrinsics_data.ppx, intrinsics_data.ppy
                            my_intrinsics = SimpleIntrinsics(fx, fy, ppx, ppy)

                            # 6. Create point cloud from bounding box
                            pcd = create_point_cloud_from_bbox(
                                color_image, depth_image, (x, y, w, h),
                                intrinsics=my_intrinsics
                            )

                            # 7. Refine point cloud
                            pcd_refined = refine_point_cloud(pcd, voxel_size=0.01,
                                                             nb_neighbors=20, std_ratio=2.0)
                            pcd_clustered = keep_largest_cluster(pcd_refined, eps=0.02, min_points=20)

                            # 8. Compute 3D distance and orientation
                            pose_data = get_distance_and_orientation_along_axes(pcd_clustered)
                            if pose_data:
                                info_str_line1 = (
                                    f"X={pose_data['distance_x']:.2f}, "
                                    f"Y={pose_data['distance_y']:.2f}, "
                                    f"Z={pose_data['distance_z']:.2f} (m)"
                                )
                                info_str_line2 = (
                                    f"D={pose_data['distance_euclidean']:.2f}m, "
                                    f"R={pose_data['roll']:.1f}, "
                                    f"P={pose_data['pitch']:.1f}, "
                                    f"Y={pose_data['yaw']:.1f} deg"
                                )
                                cv2.putText(color_image, info_str_line1, (x, y - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                cv2.putText(color_image, info_str_line2, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        except Exception as e:
                            print(f"Error during point cloud processing: {e}")

            except Exception as e:
                print(f"Error during YOLO detection: {e}")

            # 9. Show the results
            cv2.imshow("YOLO Object Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
