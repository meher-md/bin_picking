import open3d as o3d
import copy

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    return pcd_down

def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, init_trans, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == "__main__":
    voxel_size = 0.005
    source = o3d.io.read_point_cloud("../data/cad_model/object_cad.ply")
    target = o3d.io.read_point_cloud("../data/captured_pointclouds/top_object.pcd")

    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # Global Registration (RANSAC)
    result_ransac = global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # ICP refinement
    result_icp = refine_registration(source, target, result_ransac.transformation, voxel_size)

    print("Transformation Matrix:\n", result_icp.transformation)
    # This transformation gives the pose of the CAD model in the camera frame
    # Save the transformed source for visualization
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(result_icp.transformation)
    o3d.io.write_point_cloud("../data/captured_pointclouds/aligned_object.pcd", source_transformed)
