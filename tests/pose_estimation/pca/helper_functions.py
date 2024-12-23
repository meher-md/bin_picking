# helper_functions.py

import math
import numpy as np
import open3d as o3d


def create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics):
    """
    Creates an Open3D point cloud from the region defined by 'bbox' in the color/depth images.
    bbox = (x, y, w, h) in pixel coordinates.
    """
    x, y, w, h = bbox
    depth_bbox = depth_image[y:y+h, x:x+w]
    color_bbox = color_image[y:y+h, x:x+w]

    points = []
    colors = []

    for row in range(h):
        for col in range(w):
            depth_value = depth_bbox[row, col]
            if depth_value == 0:
                continue
            depth_in_meters = depth_value * 0.001  # Convert depth to meters
            pixel_x = x + col
            pixel_y = y + row
            # Deproject pixel to 3D
            pt = intrinsics.deproject(pixel_x, pixel_y, depth_in_meters)
            # (Alternatively: rs.rs2_deproject_pixel_to_point(...) if using pyrealsense2 directly)
            points.append(pt)
            colors.append(color_bbox[row, col] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def refine_point_cloud(pcd, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    """
    Downsample & remove outliers. 'voxel_size' in meters if RealSense is in meters.
    """
    # 1) Voxel downsample
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 2) Remove outliers (statistical)
    pcd_denoised, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    return pcd_denoised


def keep_largest_cluster(pcd, eps=0.02, min_points=20):
    """
    Use DBSCAN to find clusters, keep only the largest one.
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return pcd
    # Find the cluster with the max count
    largest_label = np.argmax(np.bincount(labels[valid_mask]))
    indices = np.where(labels == largest_label)[0]
    return pcd.select_by_index(indices)


def get_distance_and_orientation_along_axes(pcd):
    """
    Computes:
      - distance_x, distance_y, distance_z (OBB center)
      - distance_euclidean
      - roll, pitch, yaw (deg) from OBB rotation
    """
    if pcd.is_empty():
        return None

    obb = pcd.get_oriented_bounding_box()
    cx, cy, cz = obb.center
    distance_euclidean = float(np.linalg.norm(obb.center))

    R = obb.R
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    # Tait-Bryan angles: X->Y->Z
    roll = math.atan2(r32, r33)
    pitch = -math.asin(r31)
    yaw = math.atan2(r21, r11)

    return {
        "distance_x": float(cx),
        "distance_y": float(cy),
        "distance_z": float(cz),
        "distance_euclidean": distance_euclidean,
        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
    }


# For RealSense intrinsics convenience:
class SimpleIntrinsics:
    """
    A minimal re-implementation of 
    rs.rs2_deproject_pixel_to_point(...) 
    if you want a direct approach:
    """
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy

    def deproject(self, px, py, depth):
        # px, py in pixels
        # depth in meters
        x = (px - self.ppx) * depth / self.fx
        y = (py - self.ppy) * depth / self.fy
        z = depth
        return [x, y, z]
