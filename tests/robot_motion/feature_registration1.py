import pyrealsense2 as rs
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.neighbors import KDTree
from helper_functions import get_cad_pcd


def downsample_point_cloud(points, voxel_size):
    """
    Downsample a point cloud using a voxel grid.
    """
    grid = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    return points[unique_indices]


def compute_fpfh_features(points, voxel_size):
    """
    Compute Fast Point Feature Histograms (FPFH) using a KDTree.
    """
    tree = KDTree(points)
    fpfh_features = []
    radius = voxel_size * 5

    for point in points:
        indices = tree.query_radius([point], r=radius)[0]
        neighbors = points[indices]
        if len(neighbors) > 1:
            cov = np.cov(neighbors.T)
            fpfh_features.append(np.linalg.eigvals(cov))
        else:
            fpfh_features.append([0, 0, 0])  # Default if not enough neighbors

    return np.array(fpfh_features)


def ransac_registration(source_points, target_points, source_features, target_features, voxel_size):
    """
    Perform RANSAC-based registration using features.
    """
    from sklearn.metrics import pairwise_distances
    threshold = voxel_size * 1.5
    max_iterations = 1000
    best_inliers = 0
    best_transformation = np.eye(4)

    # Pairwise feature distances
    distances = pairwise_distances(source_features, target_features)
    for _ in range(max_iterations):
        # Randomly sample 4 correspondences
        source_indices = np.random.choice(len(source_points), 4, replace=False)
        target_indices = np.argmin(distances[source_indices], axis=1)

        # Estimate transformation
        source_sample = source_points[source_indices]
        target_sample = target_points[target_indices]
        centroid_src = np.mean(source_sample, axis=0)
        centroid_tgt = np.mean(target_sample, axis=0)
        H = (source_sample - centroid_src).T @ (target_sample - centroid_tgt)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        # Apply transformation
        transformed_source = (source_points @ R.T) + t

        # Count inliers
        inliers = np.sum(np.linalg.norm(transformed_source - target_points, axis=1) < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_transformation = np.eye(4)
            best_transformation[:3, :3] = R
            best_transformation[:3, 3] = t

    return best_transformation

from sklearn.cluster import DBSCAN
import numpy as np

def keep_largest_cluster(points, eps=0.02, min_points=10):
    """
    Keep the largest cluster from the input points.

    Parameters:
    -----------
    points : numpy.ndarray
        The input point cloud as a NumPy array of shape (N, 3).
    eps : float
        The maximum distance between points to consider them as part of the same cluster.
    min_points : int
        The minimum number of points to form a cluster.

    Returns:
    --------
    largest_cluster : numpy.ndarray
        The points belonging to the largest cluster.
    """
    if len(points) == 0:
        return np.empty((0, 3))  # Return empty array if no points exist

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)

    # Get the cluster labels
    labels = clustering.labels_

    # Identify the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 0 or np.max(unique_labels) == -1:
        return np.empty((0, 3))  # No valid clusters found

    largest_cluster_idx = unique_labels[np.argmax(counts[unique_labels >= 0])]
    largest_cluster = points[labels == largest_cluster_idx]

    return largest_cluster

def main():
    # Load CAD model and generate a point cloud
    cad_pcd = get_cad_pcd("data/VN_1400.stl")

    # Simulate a partial point cloud from the RealSense depth camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    # Warm up
    for _ in range(30):
        pipeline.wait_for_frames()

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    voxel_size = 0.01
    try:
        while True:
            # Get depth and color frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Manually select a bounding box (for simplicity)
            x, y, w, h = 100, 100, 200, 200
            depth_crop = depth_image[y:y+h, x:x+w]
            depth_crop[depth_crop == 0] = 1  # Replace zeros to avoid issues
            points = np.column_stack(np.nonzero(depth_crop))  # Convert to coordinates
            z_coords = depth_crop[points[:, 0], points[:, 1]]

            # Downsample and keep the largest cluster
            # all_points = np.column_stack((points, z_coords))
            # all_points = downsample_point_cloud(all_points, voxel_size)


            all_points = keep_largest_cluster(points)
            if all_points.shape[0] == 0:
                print("No points found. Skipping current iteration.")
                continue
            # Compute FPFH features
            features = compute_fpfh_features(all_points, voxel_size)

            # Perform RANSAC registration (mock target as CAD points)
            cad_points = np.asarray(cad_pcd.points)
            cad_features = compute_fpfh_features(cad_points, voxel_size)
            transformation = ransac_registration(all_points, cad_points, features, cad_features, voxel_size)

            # Apply the transformation
            transformed_points = (all_points @ transformation[:3, :3].T) + transformation[:3, 3]

            # Update the 3D scatter plot
            ax.cla()
            ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c="blue", s=1)
            ax.set_title("Real-Time CAD vs Partial Clouds")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.pause(0.01)

            # Show the color image
            cv2.imshow("Color Image", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
