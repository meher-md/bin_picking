# yolo_tracker_pytorch_detection.py

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import sys
import os

# Add YOLOv5 repository to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(script_dir, 'yolov5')
if not os.path.exists(yolov5_path):
    raise FileNotFoundError(
        f"YOLOv5 repository not found at {yolov5_path}. Please clone it using:\n"
        f"git clone https://github.com/ultralytics/yolov5.git"
    )

sys.path.insert(0, yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

from helper_functions import (
    create_point_cloud_from_bbox,
    refine_point_cloud,
    keep_largest_cluster,
    get_distance_and_orientation_along_axes,
    SimpleIntrinsics
)

def load_model(pt_path, device, img_size=640):
    """
    Load the YOLO model from a .pt file using YOLOv5's DetectMultiBackend.
    
    Args:
        pt_path (str): Path to the .pt model file.
        device (torch.device): Device to load the model on.
        img_size (int): Inference image size.
    
    Returns:
        model: Loaded YOLO model.
    """
    try:
        model = DetectMultiBackend(pt_path, device=device, dnn=False, data=None, fp16=False)
        model.warmup(imgsz=(1, 3, img_size, img_size))  # Warmup
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise e

def main():
    # 1. Initialize YOLO model
    model_path = "data/best.pt"  # Path to your .pt model
    device = select_device('')  # Automatically select device (CUDA if available)
    img_size = 640  # Adjust based on your model's expected input size
    confidence_threshold = 0.25  # Adjust as needed
    iou_threshold = 0.45  # Non-maximum suppression threshold

    # Load the YOLO model
    try:
        model = load_model(model_path, device, img_size)
    except Exception as e:
        print("Failed to load the model. Exiting.")
        sys.exit(1)

    # Verify model.names is a list or dict
    if not hasattr(model, 'names'):
        print("Error: model.names attribute is missing.")
        sys.exit(1)

    if isinstance(model.names, dict):
        # Convert dict to list sorted by keys
        sorted_keys = sorted(model.names.keys())
        model_names = [model.names[k] for k in sorted_keys]
    elif isinstance(model.names, list):
        model_names = model.names
    else:
        print("Error: model.names is neither a list nor a dict.")
        model_names = []

    if not model_names:
        print("Error: No class names found in model.names.")
        sys.exit(1)

    print(f"Number of classes: {len(model_names)}")
    # Optional: Print class names
    # print(f"Classes: {model_names}")

    # 2. Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense pipeline: {e}")
        sys.exit(1)

    # Warm up the camera
    for _ in range(30):
        pipeline.wait_for_frames()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    print("No frames received. Skipping...")
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                img_h, img_w = color_image.shape[:2]

                # 3. Prepare the image for YOLO
                img = torch.from_numpy(color_image).to(device)
                img = img.permute(2, 0, 1).float() / 255.0  # BGR to RGB and normalize
                img = img.unsqueeze(0)  # Add batch dimension

                # 4. Perform inference
                # with torch.no_grad():
                pred = model(img, augment=False, visualize=False)

                # 5. Apply Non-Maximum Suppression
                pred = non_max_suppression(pred, confidence_threshold, iou_threshold, agnostic=False)

                if len(pred) == 0:
                    # No detections
                    cv2.imshow("YOLO Object Tracking", color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # 6. Process detections
                for det in pred:
                    if len(det):
                        # Rescale boxes from img_size to original image size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], color_image.shape).round()

                        # Pick the detection with the highest confidence
                        best_det = det[det[:, 4].argmax()]
                        
                        # # Ensure best_det has at least 6 elements
                        # if best_det.shape[0] < 6:
                        #     print("Warning: Detected object has fewer than 6 elements. Skipping...")
                        #     continue

                        x1, y1, x2, y2, conf, cls = best_det[:6]

                        # Debug: Print class ID and confidence
                        print(f"Detected class ID: {int(cls)}, Confidence: {conf:.2f}")

                        # Check if cls is valid
                        # if not isinstance(cls, (int, float)):
                        #     print(f"Warning: Invalid class type {type(cls)}. Skipping...")
                        #     continue


                        # Extract bounding box coordinates
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        if w > 10 and h > 10:
                            # Draw bounding box
                            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # Safeguard against out-of-range class IDs
                            cls_int = int(cls)
                            if 0 <= cls_int < len(model_names):
                                label = f"{model_names[cls_int]}: {conf:.2f}"
                            else:
                                label = f"Class {cls_int}: {conf:.2f}"
                                print(f"Warning: Class ID {cls_int} is out of range.")

                            cv2.putText(color_image, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            try:
                                # 7. Extract depth intrinsics
                                intrinsics_data = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                                fx, fy = intrinsics_data.fx, intrinsics_data.fy
                                ppx, ppy = intrinsics_data.ppx, intrinsics_data.ppy
                                my_intrinsics = SimpleIntrinsics(fx, fy, ppx, ppy)

                                # 8. Create point cloud from bounding box
                                pcd = create_point_cloud_from_bbox(
                                    color_image, depth_image, (x, y, w, h),
                                    intrinsics=my_intrinsics
                                )

                                # Check if point cloud is valid
                                if pcd is None or len(pcd.points) == 0:
                                    print("Warning: Empty point cloud generated.")
                                    continue

                                # 9. Refine point cloud
                                pcd_refined = refine_point_cloud(pcd, voxel_size=0.01,
                                                                    nb_neighbors=20, std_ratio=2.0)
                                if pcd_refined is None:
                                    print("Warning: Point cloud refinement failed.")
                                    continue

                                pcd_clustered = keep_largest_cluster(pcd_refined, eps=0.02, min_points=20)
                                if pcd_clustered is None:
                                    print("Warning: Point cloud clustering failed.")
                                    continue

                                # 10. Compute 3D distance and orientation
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

                # 7. Show the results
                cv2.imshow("YOLO Object Tracking", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except KeyboardInterrupt:
                print("Interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
