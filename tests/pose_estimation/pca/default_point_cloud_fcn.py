import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
bbox = None

import numpy as np
import math
import open3d as o3d

def get_distance_and_orientation_along_axes(pcd):
    """
    Given an Open3D point cloud 'pcd', compute:
      1) distance along each axis from the origin (camera),
         i.e. x_offset, y_offset, z_offset of OBB center.
      2) the Euclidean distance to OBB center.
      3) orientation (roll, pitch, yaw) in degrees about X, Y, Z axes.

    Returns a dict:
      {
        "distance_x": float,   # offset along X
        "distance_y": float,   # offset along Y
        "distance_z": float,   # offset along Z
        "distance_euclidean": float, # sqrt(x^2 + y^2 + z^2)
        "roll": float,         # roll in degrees
        "pitch": float,        # pitch in degrees
        "yaw": float           # yaw in degrees
      }
    """

    # 1) Get Oriented Bounding Box (OBB) for the object
    obb = pcd.get_oriented_bounding_box()

    # 2) Extract the center (this is in camera coordinate frame)
    center = obb.center  # shape (3,) => [cx, cy, cz]

    distance_x = float(center[0])
    distance_y = float(center[1])
    distance_z = float(center[2])

    # Also compute the Euclidean distance from (0,0,0)
    distance_euclidean = np.linalg.norm(center)

    # 3) Extract rotation matrix R from OBB
    R = obb.R  # shape (3,3)
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    # 4) Convert rotation matrix to roll-pitch-yaw
    # Common Tait-Bryan intrinsic sequence (x->y->z):
    # roll  = atan2(r32, r33)
    # pitch = -asin(r31)
    # yaw   = atan2(r21, r11)
    roll = math.atan2(r32, r33)
    pitch = -math.asin(r31)
    yaw = math.atan2(r21, r11)

    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    return {
        "distance_x": distance_x,
        "distance_y": distance_y,
        "distance_z": distance_z,
        "distance_euclidean": distance_euclidean,
        "roll": roll_deg,
        "pitch": pitch_deg,
        "yaw": yaw_deg
    }


def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Removes points that deviate from the average distance to neighbors.
    - nb_neighbors: how many neighbors to consider
    - std_ratio: lower -> more aggressive removal
    """
    pcd_clean, inliers = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_clean   


def keep_largest_cluster(pcd, eps=0.02, min_points=10):
    """
    Performs DBSCAN clustering and keeps only the largest cluster of points.
    - eps: density parameter for clustering (distance threshold)
    - min_points: minimum points in a cluster
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    # -1 label means noise. We ignore that.
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        print("No valid clusters found!")
        return pcd
    
    # Find the label with the most points
    largest_label = np.argmax(np.bincount(valid_labels))
    # Select those points
    indices = np.where(labels == largest_label)[0]
    pcd_cluster = pcd.select_by_index(indices)
    return pcd_cluster


def refine_point_cloud(pcd, voxel_size=0.01, nb_neighbors=200, std_ratio=2.0):
    """
    Downsample and remove outliers from the point cloud to 'clean' it.

    - voxel_size: set how coarse you want the downsampling
    - nb_neighbors: number of neighbors to analyze in the outlier removal
    - std_ratio: threshold for removing points (larger -> less aggressive removal)
    """
    # 1) Downsample (if voxel_size=0.01, that's 1 cm in your coordinate scale)
    pcd_down = pcd.voxel_down_sample(voxel_size) if voxel_size > 0 else pcd

    # 2) Remove outliers using a statistical method
    #    remove_statistical_outlier returns two objects:
    #    - a new point cloud with outliers removed
    #    - an array of bool or indices of inliers
    pcd_denoised, inliers_idx = pcd_down.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # If you need only the inliers explicitly:
    # pcd_denoised = pcd_down.select_by_index(inliers_idx)

    return pcd_denoised

def draw_bbox(event, x, y, flags, param):
    global drawing, ix, iy, bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the bounding box
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the bounding box dimensions while dragging
            param['image_copy'] = param['image'].copy()
            cv2.rectangle(param['image_copy'], (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finalize the bounding box
        drawing = False
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        cv2.rectangle(param['image'], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        print(f"Bounding box selected: {bbox}")

# Create a point cloud from a bounding box
def create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics):
    x, y, w, h = bbox
    depth_bbox = depth_image[y:y+h, x:x+w]
    color_bbox = color_image[y:y+h, x:x+w]

    points = []
    colors = []

    for row in range(h):
        for col in range(w):
            depth_value = depth_bbox[row, col]
            if depth_value == 0:  # Ignore invalid depth points
                continue

            depth_in_meters = depth_value * 0.001  # Convert depth to meters
            pixel_x = x + col
            pixel_y = y + row
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_in_meters)
            points.append(point)
            colors.append(color_bbox[row, col] / 255.0)  # Normalize colors

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    return point_cloud

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # Wait for camera to stabilize
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture RGB and depth frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Select Bounding Box")
    params = {'image': color_image, 'image_copy': color_image.copy()}
    cv2.setMouseCallback("Select Bounding Box", draw_bbox, param=params)

    print("Draw a bounding box with the mouse.")

    while True:
        # Show the image with the bounding box (if any)
        cv2.imshow("Select Bounding Box", params.get('image_copy', color_image))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and bbox is not None:
            # Quit and process the selected bounding box
            print("Processing the selected bounding box...")
            break

    # Generate point cloud from the selected bounding box
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    raw_pcd = create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics)


    pcd_stat = remove_statistical_outliers(raw_pcd, nb_neighbors=20, std_ratio=2.0)

    pcd_largest = keep_largest_cluster(pcd_stat)

    # Refine the point cloud (downsample + remove outliers)
    refined_pcd = refine_point_cloud(raw_pcd,
                                     voxel_size=0.01,    # 1 cm voxel grid
                                     nb_neighbors=20,
                                     std_ratio=2.0)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd_largest])

    # from pose_estimation import get_object_pose
    # pose_info = get_object_pose(pcd_largest)

      # Print results
    result = get_distance_and_orientation_along_axes(pcd_largest)

    print("Distance along X:", result["distance_x"])
    print("Distance along Y:", result["distance_y"])
    print("Distance along Z:", result["distance_z"])
    print("Euclidean distance:", result["distance_euclidean"])
    print("roll  = %.2f deg" % result["roll"])
    print("pitch = %.2f deg" % result["pitch"])
    print("yaw   = %.2f deg" % result["yaw"])


    # # Visualization
    # # - Draw the pcd, AABB, and OBB
    # aabb_box = pose_info["aabb"]
    # aabb_box.color = (1, 0, 0)  # red
    # obb_box = pose_info["obb"]
    # obb_box.color = (0, 1, 0)   # green
    # o3d.visualization.draw_geometries([pcd_stat, aabb_box, obb_box])
finally:
    # Clean up
    pipeline.stop()
    cv2.destroyAllWindows()
