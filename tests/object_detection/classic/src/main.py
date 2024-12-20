import os
import open3d as o3d
from realsense_capture import capture_pointcloud
from preprocess import preprocess_pointcloud
from segmentation import segment_plane, cluster_objects, select_topmost_object
from registration import preprocess_point_cloud, compute_fpfh, global_registration, refine_registration
import copy

def main():
    # 1. Capture
    print("Capturing point cloud...")
    raw_pcd = capture_pointcloud()

    # 2. Preprocess
    print("Preprocessing point cloud...")
    pcd = preprocess_pointcloud(raw_pcd)
    o3d.io.write_point_cloud("../data/captured_pointclouds/scene_preprocessed.pcd", pcd)

    # 3. Segment plane (bin) and cluster objects
    print("Segmenting bin and objects...")
    plane_model, bin_plane, objects_pcd = segment_plane(pcd)
    clusters = cluster_objects(objects_pcd)
    top_object = select_topmost_object(clusters)
    o3d.io.write_point_cloud("../data/captured_pointclouds/top_object.pcd", top_object)

    # 4. Register CAD model to top object to find pose
    print("Estimating object pose...")
    voxel_size = 0.005
    source = o3d.io.read_point_cloud("../data/cad_model/object_cad.ply")
    target = top_object

    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # Global registration
    result_ransac = global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # ICP refinement
    result_icp = refine_registration(source, target, result_ransac.transformation, voxel_size)
    print("Transformation Matrix:\n", result_icp.transformation)

    # Save final alignment for inspection
    source_aligned = copy.deepcopy(source)
    source_aligned.transform(result_icp.transformation)
    o3d.io.write_point_cloud("../data/captured_pointclouds/aligned_object.pcd", source_aligned)

    # Now you have the final transformation that gives you the object pose in camera coordinates.
    # Apply any known hand-eye calibration transform to get the pose in robot base frame.

if __name__ == "__main__":
    main()
