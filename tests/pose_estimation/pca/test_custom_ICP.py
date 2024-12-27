import numpy as np

def create_test_point_clouds(case="translation"):
    ref_points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ], dtype=np.float64)  # Ensure float64 for compatibility with noise addition

    if case == "translation":
        transform = np.array([
            [1, 0, 0, 0.5],  # Translation: +0.5 in X
            [0, 1, 0, 0.2],  # Translation: +0.2 in Y
            [0, 0, 1, 0.3],  # Translation: +0.3 in Z
            [0, 0, 0, 1]
        ])
    elif case == "rotation":
        angle = np.pi / 4  # 45 degrees rotation around Z
        transform = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif case == "combined":
        angle = np.pi / 6  # 30 degrees rotation around Y
        transform = np.array([
            [np.cos(angle), 0, np.sin(angle), 0.5],  # Rotation + Translation
            [0, 1, 0, 0.2],
            [-np.sin(angle), 0, np.cos(angle), 0.3],
            [0, 0, 0, 1]
        ])
    elif case == "noise":
        transform = np.eye(4)  # No transformation
        noise = np.random.normal(0, 0.05, ref_points.shape)
        ref_points += noise
    elif case == "outliers":
        transform = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        outliers = np.random.uniform(-2, 2, (10, 3))
        ref_points = np.vstack([ref_points, outliers])
    
    elif case == "scale":
        scale_factor = 2.5  # Example scale factor
        transform = np.eye(4)  # Identity for scaling
        ref_points = ref_points * scale_factor  # Scale the reference points

    trans_points = np.dot(ref_points, transform[:3, :3].T) + transform[:3, 3]
    return ref_points, trans_points, transform

def visualize_point_clouds(ref_points, trans_points, aligned_points=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], color='r', label='Reference')
    ax.scatter(trans_points[:, 0], trans_points[:, 1], trans_points[:, 2], color='g', label='Transformed')

    if aligned_points is not None:
        ax.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], color='b', label='Aligned')

    ax.legend()
    plt.show()

def icp(src, dst, max_iterations=50, tolerance=1e-6):
    """Perform ICP registration."""
    src_h = np.hstack((src, np.ones((src.shape[0], 1))))  # Homogeneous coordinates
    prev_error = float('inf')
    
    for i in range(max_iterations):
        # Find nearest neighbors
        distances = np.linalg.norm(src[:, None] - dst[None, :], axis=2)
        indices = np.argmin(distances, axis=1)
        closest_points = dst[indices]

        # Compute transformation
        centroid_src = np.mean(src, axis=0)
        centroid_dst = np.mean(closest_points, axis=0)

        src_centered = src - centroid_src
        dst_centered = closest_points - centroid_dst

        H = np.dot(src_centered.T, dst_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_dst - np.dot(R, centroid_src)

        # Apply transformation
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t

        src_h = np.dot(transformation, src_h.T).T
        src = src_h[:, :3]

        # Check convergence
        mean_error = np.mean(np.linalg.norm(src - closest_points, axis=1))
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return transformation, prev_error

def icp_with_scale(src, dst, max_iterations=50, tolerance=1e-6):
    """Perform ICP registration with scale invariance."""
    # Compute scales
    scale_src = np.sqrt(np.mean(np.sum((src - np.mean(src, axis=0))**2, axis=1)))
    scale_dst = np.sqrt(np.mean(np.sum((dst - np.mean(dst, axis=0))**2, axis=1)))

    # Normalize point clouds
    src_normalized = src / scale_src
    dst_normalized = dst / scale_dst

    # Run ICP on normalized point clouds
    transformation, fitness = icp(src_normalized, dst_normalized, max_iterations, tolerance)

    # Adjust transformation matrix for scale
    transformation[:3, :3] *= scale_dst / scale_src  # Include scale in rotation
    transformation[:3, 3] *= scale_dst               # Include scale in translation

    return transformation, fitness


def test_cases():
    cases = ["translation", "rotation", "combined", "noise", "outliers", "scale"]
    for case in cases:
        print(f"\nTesting case: {case}")
        ref_points, trans_points, ground_truth_transform = create_test_point_clouds(case)

        visualize_point_clouds(ref_points, trans_points)

        # Run custom ICP
        transformation, fitness = icp_with_scale(trans_points, ref_points)

        print("Computed Transformation Matrix:")
        print(transformation)
        print("Fitness:", fitness)
        print("Ground Truth Transformation Matrix:")
        print(ground_truth_transform)

        # Check alignment
        aligned_points = np.dot(trans_points, transformation[:3, :3].T) + transformation[:3, 3]
        visualize_point_clouds(ref_points, trans_points, aligned_points)

if __name__ == "__main__":
    test_cases()
