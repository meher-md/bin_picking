import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_RED, Quantity_NOC_GREEN, Quantity_NOC_BLUE
from OCC.Core.AIS import AIS_Line
from OCC.Extend.TopologyUtils import TopologyExplorer
import pyrealsense2 as rs
import cv2
import open3d as o3d
from OCC.Core.Geom import Geom_Line
from OCC.Core.AIS import AIS_Line

drawing = False
ix, iy = -1, -1
bbox = None


# Existing working functions for CAD model processing
def perform_pca(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]  # Sort by eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors, centroid, eigenvalues

def extract_vertices_from_step(step_file):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        raise RuntimeError(f"Failed to read STEP file: {step_file}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    topo_exp = TopologyExplorer(shape)
    points = []

    for vertex in topo_exp.vertices():
        point = BRep_Tool.Pnt(vertex)
        points.append([point.X(), point.Y(), point.Z()])
    
    if not points:
        raise RuntimeError("No vertices could be extracted from the STEP file.")
    
    return shape, np.array(points)

# Visualize the principal component axis for CAD
def visualize_principal_axis(display, centroid, eigenvectors, eigenvalues):
    principal_axis_index = eigenvalues.argmax()
    principal_axis = eigenvectors[:, principal_axis_index]

    start = gp_Pnt(*centroid)
    direction = gp_Dir(*principal_axis)
    # Create a Geom_Line using the start point and direction
    geom_line = Geom_Line(start, direction)
    # Create an AIS_Line from the Geom_Line
    line = AIS_Line(geom_line)
    line.SetColor(Quantity_Color(Quantity_NOC_RED))

    display.Context.Display(line, True)
    print("Principal Axis Visualized in Red.")



# Existing working functions for RealSense point cloud processing
def draw_bbox(event, x, y, flags, param):
    global drawing, ix, iy, bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        param['image_copy'] = param['image'].copy()
        cv2.rectangle(param['image_copy'], (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        cv2.rectangle(param['image'], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        print(f"Bounding box selected: {bbox}")

def create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics):
    x, y, w, h = bbox
    depth_bbox = depth_image[y:y+h, x:x+w]
    color_bbox = color_image[y:y+h, x:x+w]
    points = []

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

    return np.array(points)

# New function to calculate relative pose
def calculate_relative_pose(cad_points, real_points):
    cad_eigenvectors, cad_centroid, _ = perform_pca(cad_points)
    real_eigenvectors, real_centroid, _ = perform_pca(real_points)

    # Calculate rotation
    rotation = real_eigenvectors @ np.linalg.inv(cad_eigenvectors)

    # Calculate translation
    translation = real_centroid - (rotation @ cad_centroid)

    return rotation, translation

# Main program
if __name__ == "__main__":
    cad_file = "data/VN_1400.step"
    step_shape, cad_vertices = extract_vertices_from_step(cad_file)
    cad_eigenvectors, cad_centroid, _ = perform_pca(cad_vertices)

    # RealSense pipeline setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        # Wait for the camera to stabilize
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Set up OpenCV for bounding box selection
        cv2.namedWindow("Select Bounding Box")
        params = {'image': color_image, 'image_copy': color_image.copy()}
        cv2.setMouseCallback("Select Bounding Box", draw_bbox, param=params)

        print("Draw a bounding box with the mouse.")
        while True:
            cv2.imshow("Select Bounding Box", params['image_copy'])
            if cv2.waitKey(1) & 0xFF == ord('q') and bbox:
                break

        # Generate point cloud from bounding box
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        real_points = create_point_cloud_from_bbox(color_image, depth_image, bbox, intrinsics)

        # Calculate relative pose
        rotation, translation = calculate_relative_pose(cad_vertices, real_points)
        print("Rotation Matrix:\n", rotation)
        print("Translation Vector:\n", translation)

        # Close OpenCV windows
        cv2.destroyAllWindows()



    finally:
        # Ensure cleanup of resources
        pipeline.stop()
        cv2.destroyAllWindows()


# Visualize with OpenCASCADE
display, start_display, add_menu, add_function_to_menu = init_display()
display.DisplayShape(step_shape, update=True)
visualize_principal_axis(display, cad_centroid, cad_eigenvectors, np.array([1, 1, 1]))
start_display()
