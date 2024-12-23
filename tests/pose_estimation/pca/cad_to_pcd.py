import os
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopoDS import TopoDS_Solid
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
import open3d as o3d

def load_step_file(filepath):
    """Load a STEP file and return its TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != 1:
        raise ValueError(f"Error reading STEP file: {filepath}")
    reader.TransferRoot()
    shape = reader.OneShape()
    return shape

def get_bounding_box(shape):
    """Get the bounding box of a TopoDS_Shape."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return (xmin, ymin, zmin, xmax, ymax, zmax)

def sample_points_on_shape(shape, sampling_distance):
    """Sample points on the surface of a shape at regular intervals."""
    points = []
    normals = []

    bbox = get_bounding_box(shape)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    x_steps = int((xmax - xmin) / sampling_distance) + 1
    y_steps = int((ymax - ymin) / sampling_distance) + 1
    z_steps = int((zmax - zmin) / sampling_distance) + 1

    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(shape, 1e-6)

    for i in range(x_steps):
        for j in range(y_steps):
            for k in range(z_steps):
                x = xmin + i * sampling_distance
                y = ymin + j * sampling_distance
                z = zmin + k * sampling_distance

                pnt = gp_Pnt(x, y, z)
                intersector.PerformNearest(pnt, 0.0, 1.0)
                if intersector.NbPnt() > 0:
                    point = intersector.Pnt(1)
                    face = intersector.Face(1)
                    u, v = intersector.UParameter(1), intersector.VParameter(1)

                    # Get normal
                    surface = BRep_Tool.Surface(face)
                    props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
                    if props.IsNormalDefined():
                        normal = props.Normal()
                        points.append([point.X(), point.Y(), point.Z()])
                        normals.append([normal.X(), normal.Y(), normal.Z()])

    return np.array(points), np.array(normals)

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps

def sample_points_on_shape_parametric(shape, sampling_distance=0.1):
    """
    Sample points on the surfaces of the shape in parametric space.
    'sampling_distance' here is used to set the number of samples per face.
    """
    points = []
    normals = []

    # Explorer for faces
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        
        # Wrap the face with BRepAdaptor_Surface to get parametric info
        adaptor = BRepAdaptor_Surface(face)
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

        # Example of converting 'sampling_distance' into discrete steps:
        # (Adjust logic here as needed, e.g., tie the steps to the bounding box
        # or to the param range's length, etc.)
        u_range = u_max - u_min
        v_range = v_max - v_min

        # For a simple approach, define how many steps to split each param range into:
        # Suppose each "step" in param space is about 'sampling_distance' fraction
        # of the total param range. 
        # e.g., steps = int(u_range / sampling_distance)
        # But if sampling_distance is in 3D space, this is approximate.
        # For demonstration, let's just fix a number of steps:
        u_steps = max(int(u_range / sampling_distance), 1)
        v_steps = max(int(v_range / sampling_distance), 1)

        # Sample over [u_min, u_max] x [v_min, v_max]
        for i in range(u_steps + 1):
            for j in range(v_steps + 1):
                u = u_min + (u_max - u_min) * i / u_steps
                v = v_min + (v_max - v_min) * j / v_steps
                
                # Evaluate the 3D point on the surface
                point_3d = adaptor.Value(u, v)

                # Compute normal via GeomLProp_SLProps
                prop = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 1, 1e-6)
                if prop.IsNormalDefined():
                    normal_3d = prop.Normal()
                    points.append([point_3d.X(), point_3d.Y(), point_3d.Z()])
                    normals.append([normal_3d.X(), normal_3d.Y(), normal_3d.Z()])

        exp.Next()

    return np.array(points), np.array(normals)
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box

def suggest_sampling_distance(shape, desired_samples_along_max_dim=50):
    # 1) Get bounding box
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    # 2) Largest dimension
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    max_dim = max(dx, dy, dz)
    
    # 3) Suggest sampling distance
    sampling_distance = max_dim / desired_samples_along_max_dim
    
    return sampling_distance

def write_point_cloud(filepath, points, normals):
    """Write points and normals to a .xyz file."""
    with open(filepath, 'w') as f:
        for p, n in zip(points, normals):
            f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")

def step_to_point_cloud(step_file, output_file):
    """Convert a STEP file to a point cloud."""
    shape = load_step_file(step_file)
    sampling_distance = suggest_sampling_distance(shape)
    points, normals = sample_points_on_shape_parametric(shape, sampling_distance)
    write_point_cloud(output_file, points, normals)
    print(f"Point cloud written to {output_file}")

# Example usage
if __name__ == "__main__":
    step_file = "/home/dhanuzch/Documents/bin_picking/data/VN_1400.step"  # Replace with your STEP file path
    output_file = "data/output.xyz"  # Output point cloud file
    step_to_point_cloud(step_file, output_file)
