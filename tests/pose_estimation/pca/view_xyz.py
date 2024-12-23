import open3d as o3d
import numpy as np

def load_xyz(filepath):
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            # If your .xyz has only x y z columns
            x_str, y_str, z_str = line.strip().split()[:3]
            points.append([float(x_str), float(y_str), float(z_str)])
    return np.array(points)

def visualize_xyz(filepath):
    # Load points
    points = load_xyz(filepath)
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Visualize
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    xyz_file = "data/output.xyz"  # Path to your .xyz file
    visualize_xyz(xyz_file)
