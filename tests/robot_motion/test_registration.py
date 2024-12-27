import open3d as o3d
import numpy as np
from numpy.testing import assert_almost_equal
from helper_functions import register_cad_to_partial  # Replace with your actual module

def test_register_cad_to_partial():
    # Create synthetic CAD point cloud
    cad_pcd = o3d.geometry.PointCloud()
    cad_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0.5, 0.5, 1]
    ])
    cad_pcd.points = o3d.utility.Vector3dVector(cad_points)

    # Create partial point cloud with known transformation
    transformation_true = np.array([
        [0.866, -0.5, 0, 1],
        [0.5, 0.866, 0, 0.5],
        [0, 0, 1, 0.2],
        [0, 0, 0, 1]
    ])
    partial_points = np.dot(cad_points, transformation_true[:3, :3].T) + transformation_true[:3, 3]
    partial_pcd = o3d.geometry.PointCloud()
    partial_pcd.points = o3d.utility.Vector3dVector(partial_points)

    # Visualize point clouds
    o3d.visualization.draw_geometries([cad_pcd, partial_pcd], window_name="Original Point Clouds")

    # Call the registration function
    voxel_size = 0.01
    transformation_estimated = register_cad_to_partial(cad_pcd, partial_pcd, voxel_size=voxel_size)

    # Validate the result
    try:
        assert_almost_equal(transformation_estimated, transformation_true, decimal=2)
        print("Test passed: Transformation matrix is accurate.")
    except AssertionError as e:
        print("Test failed: Transformation matrix is not accurate.")
        print(f"Estimated:\n{transformation_estimated}")
        print(f"True:\n{transformation_true}")
        raise e

    # Visualize the aligned point clouds
    cad_pcd.transform(transformation_estimated)
    o3d.visualization.draw_geometries([cad_pcd, partial_pcd], window_name="Aligned Point Clouds")

if __name__ == "__main__":
    test_register_cad_to_partial()
