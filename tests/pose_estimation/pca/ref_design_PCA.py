import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_RED, Quantity_NOC_GREEN, Quantity_NOC_BLUE
from OCC.Core.AIS import AIS_Line
from OCC.Extend.TopologyUtils import TopologyExplorer

# Perform PCA
def perform_pca(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]  # Sort by eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors, centroid, eigenvalues

# Extract vertices from STEP file
def extract_vertices_from_step(step_file):
    # Read STEP file
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        raise RuntimeError(f"Failed to read STEP file: {step_file}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    # Extract vertices
    topo_exp = TopologyExplorer(shape)
    points = []

    for vertex in topo_exp.vertices():
        point = BRep_Tool.Pnt(vertex)
        points.append([point.X(), point.Y(), point.Z()])
    
    if not points:
        raise RuntimeError("No vertices could be extracted from the STEP file.")
    
    return shape, np.array(points)

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1
from OCC.Core.Geom import Geom_Line
from OCC.Core.AIS import AIS_Line
from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GREEN, Quantity_NOC_BLUE

# Add arrows to visualize eigenvectors
def visualize_eigenvectors(display, centroid, eigenvectors):
    colors = [
        Quantity_Color(Quantity_NOC_RED),
        Quantity_Color(Quantity_NOC_GREEN),
        Quantity_Color(Quantity_NOC_BLUE),
    ]    
    axes = ["X", "Y", "Z"]
    
    for i, (eigenvector, color) in enumerate(zip(eigenvectors.T, colors)):
        # Define start point and direction for the eigenvector
        start = gp_Pnt(*centroid)
        direction = gp_Dir(*eigenvector)  # Convert eigenvector to a direction
        
        # Create a Geom_Line using the start point and direction
        geom_line = Geom_Line(start, direction)

        # Create an AIS_Line from the Geom_Line
        line = AIS_Line(geom_line)

        # Set the color for the line
        line.SetColor(color)

        # Display the line
        display.Context.Display(line, True)
        
        print(f"{axes[i]}-Axis Visualized as Line")


# Visualize the principal component axis
def visualize_principal_axis(display, centroid, eigenvectors, eigenvalues):
    # Identify the principal axis (corresponding to the largest eigenvalue)
    principal_axis_index = eigenvalues.argmax()
    principal_axis = eigenvectors[:, principal_axis_index]

    # Define start point and direction for the principal axis
    start = gp_Pnt(*centroid)
    direction = gp_Dir(*principal_axis)  # Convert principal axis vector to a direction

    # Create a Geom_Line using the start point and direction
    geom_line = Geom_Line(start, direction)

    # Create an AIS_Line from the Geom_Line
    line = AIS_Line(geom_line)

    # Set the color for the principal axis
    color = Quantity_Color(Quantity_NOC_RED)  # Red for principal axis
    line.SetColor(color)

    # Display the line
    display.Context.Display(line, True)

    print("Principal Axis Visualized in Red.")

# Main program
if __name__ == "__main__":
    # Path to the STEP file
    step_file = "/home/dhanuzch/Documents/bin_picking/data/VN_1400.step"  # Replace with your STEP file path

    try:
        # Extract vertices and shape
        shape, vertices = extract_vertices_from_step(step_file)
        print(f"Extracted {len(vertices)} vertices from the STEP file.")

        # Perform PCA
        eigenvectors, centroid, eigenvalues = perform_pca(vertices)
        print("PCA Results:")
        print("Centroid:", centroid)
        print("Eigenvalues (Variance along axes):", eigenvalues)
        print("Principal Axes (Eigenvectors):\n", eigenvectors)

        # Principal axis alignment
        print("\nPrincipal Axis Alignment:")
        print("X-Axis (Major):", eigenvectors[:, 0])
        print("Y-Axis (Intermediate):", eigenvectors[:, 1])
        print("Z-Axis (Minor):", eigenvectors[:, 2])

        # Initialize 3D Viewer
        display, start_display, add_menu, add_function_to_menu = init_display()

        # Display STEP shape
        display.DisplayShape(shape, update=True)

        # Visualize principal axes
        #visualize_eigenvectors(display, centroid, eigenvectors)
        visualize_principal_axis(display,centroid,eigenvectors,eigenvalues)
        # Start the viewer
        start_display()

    except Exception as e:
        print("Error:", e)
