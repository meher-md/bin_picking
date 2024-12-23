import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
bbox = None

def draw_bbox(event, x, y, flags, param):
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
        cv2.rectangle(param['image'], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        print(f"Bounding box selected: {bbox}")

# Create a point cloud from a bounding box
def create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics):
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

            depth_in_meters = depth_value * 0.001  # Convert depth to meters
            pixel_x = x + col
            pixel_y = y + row
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_in_meters)
            points.append(point)
            colors.append(color_bbox[row, col] / 255.0)  # Normalize colors

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    return point_cloud

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # Wait for camera to stabilize
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture RGB and depth frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Select Bounding Box")
    params = {'image': color_image, 'image_copy': color_image.copy()}
    cv2.setMouseCallback("Select Bounding Box", draw_bbox, param=params)

    print("Draw a bounding box with the mouse.")

    while True:
        # Show the image with the bounding box (if any)
        cv2.imshow("Select Bounding Box", params.get('image_copy', color_image))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and bbox is not None:
            # Quit and process the selected bounding box
            print("Processing the selected bounding box...")
            break

    # Generate point cloud from the selected bounding box
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    pcd = create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

finally:
    # Clean up
    pipeline.stop()
    cv2.destroyAllWindows()
