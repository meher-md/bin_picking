import numpy as np
import open3d as o3d

def get_object_pose(pcd):
    """
    Perform multiple geometric analyses on a point cloud 'pcd' to determine
    the object's pose with respect to the origin (0, 0, 0).

    Returns a dictionary with:
      - 'centroid': (x, y, z) of the point cloud's mean
      - 'aabb': AxisAlignedBoundingBox (with min_bound, max_bound)
      - 'obb': OrientedBoundingBox (with center, R, extent)
      - 'transform_obb': 4x4 matrix from OBB (R_obb, center)
      - 'transform_pca': 4x4 matrix from PCA (principal axes, centroid)
    """

    if pcd.is_empty():
        raise ValueError("Point cloud is empty. Cannot compute pose.")

    # -------------------------------------------------------------------------
    # 1) Centroid (the mean of all points)
    # -------------------------------------------------------------------------
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)  # shape: (3,)

    # -------------------------------------------------------------------------
    # 2) Axis-Aligned Bounding Box (AABB)
    # -------------------------------------------------------------------------
    aabb = pcd.get_axis_aligned_bounding_box()
    # aabb.min_bound, aabb.max_bound are the corners in world coordinates

    # -------------------------------------------------------------------------
    # 3) Oriented Bounding Box (OBB)
    #    - OBB has a rotation matrix 'R' and a center 'center'.
    #    - We can build a 4x4 transform from that.
    # -------------------------------------------------------------------------
    obb = pcd.get_oriented_bounding_box()
    R_obb = obb.R            # 3x3 rotation
    center_obb = obb.center  # 3D vector

    transform_obb = np.eye(4)
    transform_obb[0:3, 0:3] = R_obb
    transform_obb[0:3, 3]   = center_obb

    # -------------------------------------------------------------------------
    # 4) Principal Component Analysis (PCA)
    #    - We'll do SVD on the mean-centered points to get principal directions.
    #    - The columns of 'R_pca' form an orthonormal basis of the point cloud.
    # -------------------------------------------------------------------------
    shifted_points = points - centroid  # center the data at origin
    U, S, Vt = np.linalg.svd(shifted_points, full_matrices=False)
    # Vt is 3x3, each row is a principal axis; we want them as columns
    R_pca = Vt.T  # shape (3,3)

    transform_pca = np.eye(4)
    transform_pca[0:3, 0:3] = R_pca
    transform_pca[0:3, 3]   = centroid

    # -------------------------------------------------------------------------
    # Return a dictionary of all relevant geometry info
    # -------------------------------------------------------------------------
    return {
        "centroid": centroid,
        "aabb": aabb,
        "obb": obb,
        "transform_obb": transform_obb,
        "transform_pca": transform_pca
    }

# # -----------------------------------------------------------------------------
# # Example usage:
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Suppose you've already loaded a point cloud as 'pcd'
#     # pcd = o3d.io.read_point_cloud("path/to/your_cloud.pcd")
#     # or pcd from RealSense bounding-box extraction, etc.

#     # For demonstration, let's create a test sphere pcd:
#     pcd = o3d.geometry.PointCloud.create_sphere(radius=1.0)
#     pcd.transform([[1,0,0,1],  # shift sphere to (1,2,3) for demonstration
#                    [0,1,0,2],
#                    [0,0,1,3],
#                    [0,0,0,1]]) 

#     pose_info = get_object_pose(pcd)

#     # Print results
#     print("Object centroid:", pose_info["centroid"])
#     print("AABB bounds:", pose_info["aabb"].min_bound, pose_info["aabb"].max_bound)
#     print("OBB center:", pose_info["obb"].center)
#     print("OBB rotation:\n", pose_info["obb"].R)
#     print("Transform OBB:\n", pose_info["transform_obb"])
#     print("Transform PCA:\n", pose_info["transform_pca"])

#     # Visualization
#     # - Draw the pcd, AABB, and OBB
#     aabb_box = pose_info["aabb"]
#     aabb_box.color = (1, 0, 0)  # red
#     obb_box = pose_info["obb"]
#     obb_box.color = (0, 1, 0)   # green
#     o3d.visualization.draw_geometries([pcd, aabb_box, obb_box])
