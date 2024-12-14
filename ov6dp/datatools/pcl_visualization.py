import open3d as o3d
import numpy as np
import torch


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



# Function to generate a random point cloud
def visualize_pointclouds(pointclouds):

    point_clouds = []
    colors = [[0, 0, 1], 
              [0, 1, 0], 
              [1, 0, 0], 
              [1, 0.5, 0], 
              [0, 1, 0.5], 
              [0.5, 0, 1], 
              [0, 0, 0], 
              [0.5, 0.5, 0.5]]  # Colors for each point cloud (red, green, blue)

    for i, pcl in enumerate(pointclouds):
        #print(pcl)
        pcd = torch_to_o3d(pcl) 
        
        # Assign a color to the point cloud
        pcd.paint_uniform_color(colors[i % len(colors)])  # Set the color for each point cloud

        point_clouds.append(pcd)

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(point_clouds)

    return


def remove_zero_points(pointcloud: torch.Tensor) -> torch.Tensor:
    """
    Takes a pointcloud of shape (H, W, 3) and returns it as shape (N, 3) by removing points 
    with coordinates (0, 0, 0).

    Args:
        pointcloud (torch.Tensor): The input point cloud of shape (H, W, 3).
    
    Returns:
        torch.Tensor: The point cloud with shape (N, 3) where N is the number of valid points 
                      (excluding those with coordinates (0, 0, 0)).
    """
    # Ensure input is of the correct shape
    if pointcloud.dim() == 3 and pointcloud.shape[2] == 3:
        # Reshape the pointcloud to (H*W, 3) so each point is a row
        pointcloud_reshaped = pointcloud.view(-1, 3)
    else:
        pointcloud_reshaped = pointcloud

    # Create a mask where (0, 0, 0) points are False
    mask = ~(torch.all(pointcloud_reshaped == 0, dim=1))

    # Apply the mask to keep only the valid points
    valid_points = pointcloud_reshaped[mask]

    return valid_points