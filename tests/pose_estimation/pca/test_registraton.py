import os
import open3d as o3d
from helper_functions import register_point_cloud_to_reference
import numpy as np

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

def process_and_visualize_point_clouds():
    # Path to the data folder containing .pcd files
    data_folder = "/home/dhanuzch/Documents/bin_picking/data"
    
    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: The folder '{data_folder}' does not exist.")
        return
    
    # Load the visible point cloud file
    visible_pcd_path = os.path.join(data_folder, "object_pcd.pcd")
    if not os.path.exists(visible_pcd_path):
        print(f"Error: File '{visible_pcd_path}' not found.")
        return
    raw_pcd = o3d.io.read_point_cloud(visible_pcd_path)
    # Preprocess the object pcd
    visible_pcd = remove_statistical_outliers(raw_pcd)
    # Load the reference point cloud file
    reference_pcd_path = os.path.join(data_folder, "generated_pcd.pcd")
    if not os.path.exists(reference_pcd_path):
        print(f"Error: File '{reference_pcd_path}' not found.")
        return
    reference_pcd = o3d.io.read_point_cloud(reference_pcd_path)
    
    # Visualize individual point clouds
    print("Displaying the visible point cloud...")
    o3d.visualization.draw_geometries([visible_pcd], window_name="Visible Point Cloud")
    
    print("Displaying the reference point cloud...")
    o3d.visualization.draw_geometries([reference_pcd], window_name="Reference Point Cloud")
    
    print("Original Bounding Boxes:")
    print("Visible Point Cloud Bounds:", visible_pcd.get_axis_aligned_bounding_box())
    print("Reference Point Cloud Bounds:", reference_pcd.get_axis_aligned_bounding_box())
    
    # Normalize the scale of the reference point cloud
    visible_size = (visible_pcd.get_max_bound() - visible_pcd.get_min_bound()).max()
    reference_size = (reference_pcd.get_max_bound() - reference_pcd.get_min_bound()).max()
    scale_factor = visible_size / reference_size
    print(f"Scaling reference point cloud by factor: {scale_factor}")
    reference_pcd.scale(scale_factor, center=reference_pcd.get_center())
    

    # Align the centers of the two point clouds
    visible_center = visible_pcd.get_center()
    reference_center = reference_pcd.get_center()
    translation_vector = visible_center - reference_center
    print(f"Translating reference point cloud by vector: {translation_vector}")
    reference_pcd.translate(translation_vector)
    

    
    # Visualize merged point clouds
    print("Displaying the merged point clouds...")
    o3d.visualization.draw_geometries([visible_pcd, reference_pcd], window_name="Merged Point Clouds (Aligned)")
    
    # Perform registration
    print("Performing registration...")
    transformation, fitness = register_point_cloud_to_reference(visible_pcd, reference_pcd)
    print("Transformation Matrix:")
    print(transformation)
    print(f"Fitness Score: {fitness}")
    
    # Apply the transformation and visualize the aligned point clouds
    visible_pcd.transform(transformation)
    print("Displaying the aligned point clouds...")
    o3d.visualization.draw_geometries([visible_pcd, reference_pcd], window_name="Aligned Point Clouds")


# Run the function to visualize individual point clouds
process_and_visualize_point_clouds()