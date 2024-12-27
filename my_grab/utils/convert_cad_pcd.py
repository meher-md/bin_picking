
import open3d as o3d

def save_cad_pcd(file_path, number_of_points=100000, save_path=None):
    """
    Load a CAD file and generate a point cloud with uniform sampling.
    Optionally save the point cloud to a file.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    # Ensure the mesh has triangles
    if not mesh.has_triangles():
        raise ValueError(f"File {file_path} does not contain valid triangular mesh data.")

    # Uniformly sample the mesh to create a point cloud
    cad_pcd = mesh.sample_points_uniformly(number_of_points)

    # Check if the resulting point cloud is valid
    if len(cad_pcd.points) == 0:
        raise ValueError("Failed to generate point cloud from CAD mesh. Check input file or sampling parameters.")

    # Save the PCD to a file if save_path is specified
    if save_path:
        o3d.io.write_point_cloud(save_path, cad_pcd)
        print(f"Point cloud saved to {save_path}")

    return cad_pcd




if __name__ == "__main__":
    file_path = "assets/VB_1400.obj"  # Input CAD file
    save_path = "assets/VN_1400.pcd"  # Output PCD file
    cad_pcd = save_cad_pcd(file_path, save_path=save_path)