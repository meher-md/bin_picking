import open3d as o3d

def mesh_to_point_cloud(mesh_file, output_pcd_file, num_points=100000):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    # Sample points from the mesh
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Save the point cloud
    o3d.io.write_point_cloud(output_pcd_file, pcd)
    print(f"Saved point cloud to {output_pcd_file}")

if __name__ == "__main__":
    mesh_file = "../data/cad_model/object.stl"
    output_pcd_file = "../data/cad_model/object.ply"
    mesh_to_point_cloud(mesh_file, output_pcd_file)
