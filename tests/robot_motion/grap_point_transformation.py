import numpy as np

def transform_points_cad_to_base(
    points_cad: np.ndarray, 
    T_c_from_cad: np.ndarray, 
    T_b_from_c: np.ndarray
) -> np.ndarray:
    """
    Transform a set of points from the CAD frame to the robot base frame.

    :param points_cad: Nx3 array of points in the CAD coordinate frame
    :param T_c_from_cad: 4x4 homogeneous transform matrix (camera <- CAD)
    :param T_b_from_c: 4x4 homogeneous transform matrix (base <- camera)
    :return: Nx3 array of points in the robot base frame
    """

    # Ensure points are in homogeneous form: Nx4
    num_points = points_cad.shape[0]
    ones = np.ones((num_points, 1))
    points_cad_h = np.hstack((points_cad, ones))  # Nx4

    # Transform CAD -> Camera
    points_c_in_h = (T_c_from_cad @ points_cad_h.T).T  # Nx4

    # Transform Camera -> Base
    points_b_in_h = (T_b_from_c @ points_c_in_h.T).T   # Nx4

    # Convert back from homogeneous Nx4 to Nx3
    points_b = points_b_in_h[:, :3] / points_b_in_h[:, [3]]

    return points_b


# --------------------------
# Example usage below
# --------------------------
if __name__ == "__main__":

    # EXAMPLE 1: Hard-coded transformations (replace with actual values)

    # Suppose we got these from ICP or registration:
    # T_c_from_cad = camera <- CAD
    T_c_from_cad = np.array([
        [ 0.999,  0.010,  0.035,  0.1 ],
        [-0.010,  0.999,  0.001,  0.0 ],
        [-0.035, -0.001,  0.999,  0.2 ],
        [ 0.0,    0.0,    0.0,    1.0 ]
    ])

    # Suppose from camera-to-robot-base calibration:
    # T_b_from_c = base <- camera
    T_b_from_c = np.array([
        [ 1.0,  0.0,  0.0,  0.5 ],
        [ 0.0,  1.0,  0.0, -0.2 ],
        [ 0.0,  0.0,  1.0,  0.7 ],
        [ 0.0,  0.0,  0.0,  1.0 ]
    ])

    # Known grasp points in the CAD coordinate frame
    # e.g., 3 points that the CAD model says are good for grasping
    grasp_points_cad = np.array([
        [0.0,  0.0,  0.0],   # Example point 1
        [0.02, 0.03, 0.0],   # Example point 2
        [0.01, 0.0,  0.05]   # Example point 3
    ])

    # Transform them
    grasp_points_base = transform_points_cad_to_base(
        grasp_points_cad, 
        T_c_from_cad, 
        T_b_from_c
    )

    print("CAD grasp points:\n", grasp_points_cad)
    print("Grasp points in robot base frame:\n", grasp_points_base)

