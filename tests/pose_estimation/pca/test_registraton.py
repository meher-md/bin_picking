import open3d as o3d
import numpy as np
from helper_functions import register_point_cloud_to_reference

# Create two synthetic point clouds
def create_test_point_clouds():
    # Reference point cloud (a cube)
    ref_pcd = o3d.geometry.PointCloud()
    ref_points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)

    # Visible (transformed) point cloud
    trans_pcd = o3d.geometry.PointCloud()
    transform = np.array([
        [1, 0, 0, 0.5],  # Translation: +0.5 in X
        [0, 1, 0, 0.2],  # Translation: +0.2 in Y
        [0, 0, 1, 0.3],  # Translation: +0.3 in Z
        [0, 0, 0, 1]
    ])
    trans_points = np.dot(ref_points, transform[:3, :3].T) + transform[:3, 3]
    trans_pcd.points = o3d.utility.Vector3dVector(trans_points)

    return ref_pcd, trans_pcd, transform

# Test registration function
def test_registration():
    ref_pcd, visible_pcd, ground_truth_transform = create_test_point_clouds()

    # Visualize the original clouds (optional)
    print("Visualizing reference and visible point clouds...")
    ref_pcd.paint_uniform_color([1, 0, 0])  # Red
    visible_pcd.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([ref_pcd, visible_pcd])

    # Run ICP registration
    print("Running registration...")
    threshold = 0.05  # Maximum correspondence distance
    transformation, fitness = register_point_cloud_to_reference(visible_pcd, ref_pcd, threshold)

    print("Computed Transformation Matrix:")
    print(transformation)

    print("Fitness:", fitness)
    print("Ground Truth Transformation Matrix:")
    print(ground_truth_transform)

    # Apply the transformation to align the visible point cloud
    aligned_pcd = visible_pcd.transform(transformation)

    # Visualize the aligned clouds
    print("Visualizing aligned point clouds...")
    o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color([1, 0, 0]),
                                       aligned_pcd.paint_uniform_color([0, 1, 0])])

if __name__ == "__main__":
    test_registration()
