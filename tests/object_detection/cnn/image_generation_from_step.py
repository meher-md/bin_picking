import os
import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Graphic3d import Graphic3d_BT_RGB
from OCC.Core.V3d import V3d_Viewer
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_Color
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Ax3, gp_Ax1, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

try:
    from OCC.Display.SimpleGui import init_display
    USE_DISPLAY = True
except ImportError:
    USE_DISPLAY = False


def load_step_file(step_filename):
    """Load STEP file and return the resulting shape."""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_filename)
    if status != IFSelect_RetDone:
        raise RuntimeError("Failed to read the STEP file.")
    step_reader.TransferRoots()
    shape = step_reader.Shape()
    return shape


def init_viewer():
    """Initialize a viewer with pythonOCC. Returns display and viewer objects."""
    if USE_DISPLAY:
        display, start_display, add_menu, add_function_to_menu = init_display()
        return display, display.View
    else:
        return None, None


def display_shape(display, shape):
    """Display shape in the 3D viewer."""
    ais_shape = AIS_Shape(shape)
    if display is not None:
        display.Context.EraseAll(True)
        display.Context.Display(ais_shape, True)
        display.FitAll()
    return ais_shape


def set_background_color(view, r=1.0, g=1.0, b=1.0):
    """Set the background color of the view."""
    view.SetBackgroundColor(Quantity_Color(r, g, b, Quantity_NOC_WHITE))


def save_view_to_image(view, filename, width=800, height=600):
    view.ChangeRenderingParams().ToShowEdges = False
    view.Update()
    if not view.Dump(filename):
        raise RuntimeError("Image dump failed. Check if off-screen rendering is supported.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_stl_file>")
        sys.exit(1)

    stl_filename = sys.argv[1]
    if not os.path.isfile(stl_filename):
        print(f"File not found: {stl_filename}")
        sys.exit(1)

    output_dir = "rendered_images"
    os.makedirs(output_dir, exist_ok=True)

    original_shape = load_step_file(stl_filename)

    display, view = init_viewer()
    if view is not None:
        set_background_color(view, 1.0, 1.0, 1.0)
    
    # Set a fixed camera position, looking down on the object from some distance
    # For example, camera at (0, -300, 200) looking at origin, up along Z
    if view:
        view.SetProj(0, 0, 1) 
        view.SetEye(0, -300, 200)
        view.SetAt(0, 0, 0)
        view.SetUp(0, 0, 1)

    # Define the angles at which we rotate the object
    azimuth_angles = range(0, 360, 30)
    elevation_angles = [0, 30, 60]

    # Rotation axis definitions
    # We'll rotate around Z-axis for azimuth and around X-axis for elevation, for example.
    from math import radians, sin, cos
    img_count = 0

    for elev in elevation_angles:
        for az in azimuth_angles:
            # Create a transformation for rotation
            trsf = gp_Trsf()

            # Apply elevation rotation around X-axis
            # Rotate about the origin (0,0,0)
            trsf.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(1,0,0)), radians(elev))

            # Apply azimuth rotation around Z-axis
            trsf2 = gp_Trsf()
            trsf2.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(0,0,1)), radians(az))

            # Combine transformations: first elevation, then azimuth
            trsf.Multiply(trsf2)

            # Transform the original shape
            transformer = BRepBuilderAPI_Transform(original_shape, trsf, True)
            rotated_shape = transformer.Shape()

            ais_shape = display_shape(display, rotated_shape) if display else None
            if view:
                # view.SetProj(0, 0, 1)
                # view.SetEye(0.0, 0.0, 0.4)   # Camera positioned 40cm (0.4m) along Z-axis
                # view.SetAt(0.0, 0.0, 0.0)    # Looking at the origin
                # view.SetUp(0.0, 1.0, 0.0)    # Y-axis as up
                view.ZFitAll()
                img_filename = os.path.join(output_dir, f"img_{img_count:03d}_az{az}_elev{elev}.png")
                save_view_to_image(view, img_filename)
            img_count += 1


if __name__ == "__main__":
    main()
