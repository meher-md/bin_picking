import open3d as o3d
import numpy as np

def segment_plane(pcd, dist_threshold=0.01, ransac_n=3, num_iterations=1000):
    # Plane segmentation
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    plane_cloud = pcd.select_by_index(inliers)
    object_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, plane_cloud, object_cloud

def cluster_objects(pcd, eps=0.02, min_points=50):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        clusters.append(cluster)
    return clusters

def select_topmost_object(clusters):
    # Decide topmost by highest max z
    max_z = -np.inf
    top_cluster = None
    for c in clusters:
        pts = np.asarray(c.points)
        z_max = np.max(pts[:,2])
        if z_max > max_z:
            max_z = z_max
            top_cluster = c
    return top_cluster

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("../data/captured_pointclouds/scene_preprocessed.pcd")
    plane_model, bin_plane, objects_pcd = segment_plane(pcd)
    clusters = cluster_objects(objects_pcd)
    top_object = select_topmost_object(clusters)
    o3d.io.write_point_cloud("../data/captured_pointclouds/top_object.pcd", top_object)
