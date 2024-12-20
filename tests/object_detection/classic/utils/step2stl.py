from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools_Write
from OCC.Core.StlAPI import StlAPI_Writer

def convert_step_to_stl(step_file_path, output_stl_path, mesh_precision=0.01):
    # Read the STEP file
    reader = STEPControl_Reader()
    reader.ReadFile(step_file_path)
    reader.TransferRoot()
    shape = reader.OneShape()

    # Mesh the STEP shape
    mesh = BRepMesh_IncrementalMesh(shape, mesh_precision)
    mesh.Perform()

    # Write to STL
    writer = StlAPI_Writer()
    writer.SetASCIIMode(False)  # Binary mode for smaller files
    writer.Write(shape, output_stl_path)
    print(f"Converted {step_file_path} to {output_stl_path}")

if __name__ == "__main__":
    step_file = "../data/cad_model/object.step"
    stl_output = "../data/cad_model/object.stl"
    convert_step_to_stl(step_file, stl_output)
