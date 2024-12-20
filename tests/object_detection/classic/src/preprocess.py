import open3d as o3d

def preprocess_pointcloud(pcd, voxel_size=0.002):
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd_down.orient_normals_consistent_tangent_plane(30)

    # Optional: Remove outliers
    pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd_down

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("../data/captured_pointclouds/scene.pcd")
    pcd = preprocess_pointcloud(pcd)
    o3d.io.write_point_cloud("../data/captured_pointclouds/scene_preprocessed.pcd", pcd)
