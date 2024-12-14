import open3d as o3d
import numpy as np
import torch
import copy



def load_point_cloud_from_ply(file_path):
    """
    Load a point cloud from a .ply file using Open3D.
    """
    cloud = o3d.io.read_point_cloud(file_path)
    return cloud

def o3d_to_torch(o3d_cloud):
    """
    Convert an Open3D PointCloud to a PyTorch tensor.
    """
    # Step 1: Convert the Open3D point cloud to a NumPy array
    numpy_array = np.asarray(o3d_cloud.points)

    # Step 2: Convert the NumPy array to a PyTorch tensor
    torch_tensor = torch.from_numpy(numpy_array).float()

    return torch_tensor

def torch_to_o3d(torch_tensor):
    """
    Convert a PyTorch tensor (Nx3 or Nx6) to an Open3D PointCloud.
    Supports XYZ points.
    """
    # Step 1: Convert the PyTorch tensor to a NumPy array
    numpy_array = torch_tensor.cpu().numpy()

    # Step 2: Convert the NumPy array to Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(numpy_array)

    return point_cloud

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 5
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 20
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100, 0.01))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 10
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def initial_translation(pc1, pc2):
    """Estimate initial translation between two point clouds."""
    # Compute centroids
    centroid1 = np.mean(np.asarray(pc1.points), axis=0)
    centroid2 = np.mean(np.asarray(pc2.points), axis=0)
    
    # Compute translation vector
    translation = centroid2 - centroid1
    return translation

def align_point_clouds(pc1, pc2, voxel_size=0.05):
    """Align two point clouds using translation, RANSAC, and ICP. pc1 = source, pc2 = target"""
    # Step 1: Initial Translation
    translation = initial_translation(pc1, pc2)
    
    # Apply translation to pc1
    pc1.translate(translation)
    taslated_copy = copy.deepcopy(pc1)

    source_down, source_fpfh = preprocess_point_cloud(pc1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pc2, voxel_size)
    
    # Step 2: RANSAC for rough alignment
    reg_ransac = execute_global_registration(source_down, target_down,
                                             source_fpfh, target_fpfh,
                                             voxel_size)

    # Apply the RANSAC transformation
    #pc1.transform(reg_ransac.transformation)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pc1.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pc2.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Step 3: ICP for fine alignment
    reg_icp = refine_registration(pc1, pc2, source_fpfh, target_fpfh,
                                 voxel_size, reg_ransac)

    # Combine transformations: Initial translation -> RANSAC -> ICP ##@ reg_ransac.transformation
    combined_transformation = reg_icp.transformation  @ np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])
    
    return torch.from_numpy(combined_transformation), taslated_copy

def match_pointclouds(source_pcl, target_pcl):
    target_cloud = torch_to_o3d(target_pcl)
    source_cloud = torch_to_o3d(source_pcl)

    # 1. Perform RANSAC-based rough alignment
    transformation, translated_copy = align_point_clouds(copy.deepcopy(source_cloud), copy.deepcopy(target_cloud))
    print("Transformation matrix:\n", transformation)

    # Apply the rough transformation to source_cloud
    source_cloud.transform(transformation.numpy())

    return o3d_to_torch(source_cloud), torch.as_tensor(transformation), 0, o3d_to_torch(translated_copy)
