import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and estimate normals for the point cloud."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd_down

def compute_fpfh_features(pcd, voxel_size):
    """Compute Fast Point Feature Histograms (FPFH) for the point cloud."""
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return fpfh

def register_with_ransac(source, target, source_features, target_features, voxel_size):
    """Perform RANSAC registration using FPFH features."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_features, target_features,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=400000, confidence=0.999)
    )
    return result

def main():
    # Load the standard and partial point clouds
    standard_pcd = o3d.io.read_point_cloud("standard.ply")  # Replace with your standard PCD file
    partial_pcd = o3d.io.read_point_cloud("partial.ply")    # Replace with your partial PCD file

    # Set voxel size for preprocessing
    voxel_size = 0.01

    # Preprocess point clouds
    standard_down = preprocess_point_cloud(standard_pcd, voxel_size)
    partial_down = preprocess_point_cloud(partial_pcd, voxel_size)

    # Compute FPFH features
    standard_features = compute_fpfh_features(standard_down, voxel_size)
    partial_features = compute_fpfh_features(partial_down, voxel_size)

    # Perform RANSAC registration
    print("Running RANSAC registration...")
    result_ransac = register_with_ransac(standard_down, partial_down, standard_features, partial_features, voxel_size)

    # Display the RANSAC result
    print("RANSAC Transformation Matrix:")
    print(result_ransac.transformation)

    # Transform the partial point cloud using the estimated transformation
    partial_pcd.transform(result_ransac.transformation)

    # Visualize the alignment
    o3d.visualization.draw_geometries([standard_pcd, partial_pcd], window_name="Feature-Based Registration")

if __name__ == "__main__":
    main()
