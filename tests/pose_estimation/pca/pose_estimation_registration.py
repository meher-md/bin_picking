import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
bbox = None

def draw_bbox(event, x, y, flags, param):
    """
    OpenCV mouse callback to draw a bounding box on an image.
    """
    global drawing, ix, iy, bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the bounding box
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the bounding box dimensions while dragging
            param['image_copy'] = param['image'].copy()
            cv2.rectangle(param['image_copy'], (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finalize the bounding box
        drawing = False
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        cv2.rectangle(param['image'], (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 2)
        print(f"Bounding box selected: {bbox}")

def create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics):
    """
    Create a point cloud from the selected bounding box in the RGB image and depth map.
    """
    x, y, w, h = bbox
    depth_bbox = depth_image[y:y+h, x:x+w]
    color_bbox = color_image[y:y+h, x:x+w]

    points = []
    colors = []

    for row in range(h):
        for col in range(w):
            depth_value = depth_bbox[row, col]
            if depth_value == 0:  # Ignore invalid depth points
                continue

            # Convert depth to meters
            depth_in_meters = depth_value * 0.001
            
            # Pixel coordinates in the full image
            pixel_x = x + col
            pixel_y = y + row
            
            # Deproject pixel to 3D point
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_in_meters)
            points.append(point)
            
            # Normalize colors
            colors.append(color_bbox[row, col] / 255.0)

    # Build an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def align_frames(frames):
    """
    Align depth frame to color frame using RealSense's align function.
    """
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    return aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()

def point_to_point_registration(source_pcd, target_pcd):
    if source_pcd.is_empty():
        raise ValueError("Source point cloud is empty. Cannot register.")
    if target_pcd.is_empty():
        raise ValueError("Target (reference) point cloud is empty. Cannot register.")

    threshold = 0.02  # example distance threshold
    voxel_size = 0.05  # Try 5 cm, or 0.01 for 1 cm, depending on your real scale

    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Remove outliers statistically
    source_down, _ = source_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    target_down, _ = target_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print("Source downsampled points:", len(source_down.points))
    print("Target downsampled points:", len(target_down.points))

    # Then run ICP
    threshold = 0.02  # 2 cm, adjust if needed
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    print("Transformation:", reg_p2p.transformation)
    return reg_p2p.transformation


def transform_point_cloud(pcd, transformation):
    """
    Apply a transformation to a point cloud.
    """
    pcd.transform(transformation)
    return pcd

def visualize_alignment(source_pcd, target_pcd, transformation):
    """
    Visualize the alignment of source and target point clouds with the transformation applied.
    """
    source_temp = source_pcd.clone()
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0, 0])  # Red for the source
    target_pcd_temp = target_pcd.clone()
    target_pcd_temp.paint_uniform_color([0, 1, 0])  # Green for the target
    o3d.visualization.draw_geometries(
        [source_temp, target_pcd_temp],
        window_name="Alignment Visualization"
    )

def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        print("Stabilizing camera (discarding initial frames)...")
        for _ in range(30):
            pipeline.wait_for_frames()

        print("Capturing frames...")
        frames = pipeline.wait_for_frames()
        color_frame, depth_frame = align_frames(frames)

        if not depth_frame or not color_frame:
            raise RuntimeError("Invalid frames received.")

        # Convert frames to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Set up OpenCV window and mouse callback for selecting a bounding box
        global bbox
        bbox = None
        params = {'image': color_image, 'image_copy': color_image.copy()}
        cv2.namedWindow("Select Bounding Box", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Bounding Box", draw_bbox, param=params)

        print("Draw a bounding box with the mouse. Press 'q' when done.")
        while True:
            cv2.imshow("Select Bounding Box", params.get('image_copy', color_image))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and bbox is not None:
                break

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        selected_pcd = create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics)

        # Load the reference point cloud from your data folder
        print("Loading reference point cloud from data/reference.xyz ...")
        reference_path = "data/output.xyz"  # Adjust path if needed
        reference_pcd = o3d.io.read_point_cloud(reference_path, format='xyz')
        
        if reference_pcd.is_empty():
            raise RuntimeError(f"Failed to load reference point cloud from {reference_path}")

        # Perform point-to-point registration
        print("Performing alignment to the reference point cloud...")
        try:
            transformation = point_to_point_registration(selected_pcd, reference_pcd)
        except Exception as e:
            print(e)
        print("Transformation Matrix:\n", transformation)

        # Visualize alignment
        #visualize_alignment(selected_pcd, reference_pcd, transformation)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
