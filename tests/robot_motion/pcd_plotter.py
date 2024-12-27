import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def setup_matplotlib_visualizer():
    """
    Set up the Matplotlib visualizer for 3D point clouds.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D subplot for point cloud visualization.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real-Time CAD vs Partial Clouds")
    return fig, ax


def update_partial_cloud(ax, partial_pcd):
    """
    Update the partial point cloud in the Matplotlib visualizer.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The Matplotlib 3D subplot for visualization.
    partial_pcd : open3d.geometry.PointCloud
        The new partial point cloud to display.
    """
    # Clear the previous plot
    ax.cla()

    # Extract points and colors
    points = np.asarray(partial_pcd.points)
    colors = np.asarray(partial_pcd.colors)

    # Scatter plot for points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)

    # Set labels and title again (since `cla()` clears them)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real-Time CAD vs Partial Clouds")

    # Pause for real-time update
    plt.pause(0.01)
