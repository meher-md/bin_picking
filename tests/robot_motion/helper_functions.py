# helper_functions.py

import math
import numpy as np
import open3d as o3d
import open3d as o3d
import pyrealsense2 as rs

def register_cad_to_partial(cad_pcd, partial_points, voxel_size=0.01):
    """
    Register a CAD model (as a point cloud) to the partial point cloud and return the transformation matrix.
    """
    import open3d as o3d

    # 1. Downsample both point clouds
    cad_down = cad_pcd.voxel_down_sample(voxel_size)
    partial_down = partial_points.voxel_down_sample(voxel_size)

    # 2. Estimate normals
    cad_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    partial_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # 3. Perform ICP registration
    threshold = voxel_size * 5.0  # Maximum correspondence distance
    icp_result = o3d.pipelines.registration.registration_icp(
        source=cad_down,
        target=partial_down,
        max_correspondence_distance=threshold,
        init=np.eye(4),  # Initial guess (identity matrix)
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # 4. Check for significant transformation
    if icp_result.fitness < 0.5:
        print("Low fitness. ICP might have failed.")
    elif icp_result.inlier_rmse > 0.01:
        print("High RMSE. Registration might be inaccurate.")

    # 5. Return the transformation matrix
    return icp_result.transformation




def get_cad_pcd(file_path, number_of_points=100000):
    """
    Load a CAD file and generate a point cloud with uniform sampling.
    """
    import open3d as o3d

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    # Ensure the mesh has triangles
    if not mesh.has_triangles():
        raise ValueError(f"File {file_path} does not contain valid triangular mesh data.")


    # Uniformly sample the mesh to create a point cloud
    cad_pcd = mesh.sample_points_uniformly(number_of_points)

    # Check if the resulting point cloud is valid
    if len(cad_pcd.points) == 0:
        raise ValueError("Failed to generate point cloud from CAD mesh. Check input file or sampling parameters.")
    
    return cad_pcd


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


import numpy as np
import open3d as o3d
import math

def load_reference_point_cloud(file_path):
    """
    Load a reference point cloud from a file (e.g., CAD model).
    """
    reference_pcd = o3d.io.read_point_cloud(file_path)
    if reference_pcd.is_empty():
        raise ValueError("Reference point cloud is empty or invalid.")
    return reference_pcd

def register_point_cloud_to_reference(visible_pcd, reference_pcd, threshold=0.02):
    """
    Align the visible point cloud to the reference using ICP.
    """
    icp_result = o3d.pipelines.registration.registration_icp(
        source=visible_pcd,
        target=reference_pcd,
        max_correspondence_distance=threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result.transformation, icp_result.fitness

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to roll, pitch, yaw (Tait-Bryan angles).
    """
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = -math.asin(R[2, 0])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
