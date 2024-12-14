import torch
from plyfile import PlyData

def read_ply_to_tensor(ply_file_path):
    # Read the .ply file
    print(ply_file_path)
    ply_data = PlyData.read(ply_file_path)
    
    # Extract vertex data from the file (assuming the file contains vertex data)
    vertex_data = ply_data['vertex']
    
    # Extract the points (x, y, z coordinates) and convert to a list of tuples
    points = [(vertex['x'], vertex['y'], vertex['z']) for vertex in vertex_data]
    
    # Convert the list of points to a PyTorch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    return points_tensor


class PointcloudCreator:

    def __init__(self, camera_matrix=torch.Tensor([[900.27978516,   0.        , 960.        ],
                                               [  0.        , 900.07507324, 540.        ],
                                               [  0.        ,   0.        ,   1.        ]])) -> None:
        self.camera_intrinsics = camera_matrix
        
    def image_to_pointcloud(self, image: torch.tensor):
        """
        iamge in format depth or rgbd image (c, H, W)
        returns shape (H, W, 3/6) with the last dimension being (X, Y, Z, R, G, B)
        """
        # Extract dimensions
        channels, H, W = image.shape

        # Intrinsic camera parameters
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]

        # Meshgrid of pixel coordinates (u, v)
        u = torch.arange(0, W).float().unsqueeze(0).repeat(H, 1)  # Shape (H, W)
        v = torch.arange(0, H).float().unsqueeze(1).repeat(1, W)  # Shape (H, W)

        # Extract the depth channel
        if channels == 4:
            depth = image[3, :, :]  # Shape (H, W)
        elif channels == 1:
            depth = image[0, :, :]
        else:
            raise ValueError(f"Expected depth or rgbd image. Got {channels} channels insted.")

        # Mask out invalid depth values (optional step)
        valid_mask = depth > 0  # Keep only valid depth points

        # Compute the 3D coordinates (camera space)
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        # Stack the 3D coordinates (X, Y, Z) in the last dimension
        points_3d = torch.stack((X, Y, Z), dim=-1)  # Shape (H, W, 3)

        if channels == 4:
            # Extract RGB channels and permute to shape (H, W, 3)
            rgb = image[:3, :, :].permute(1, 2, 0)  # Shape (H, W, 3)

            # Combine the 3D coordinates with the RGB values along the last dimension
            point_cloud = torch.cat((points_3d, rgb), dim=-1)  # Shape (H, W, 6)
        else:
            point_cloud = points_3d

        # Apply the mask to filter out invalid points (optional)
        point_cloud[~valid_mask] = 0  # Set invalid points to zero or another value

        return point_cloud

    def extract_masked_pointcloud(self, point_cloud, mask):
        """
        Extract the masked point cloud from a point cloud using a boolean mask.
        
        Args:
            point_cloud: Tensor of shape (H, W, 6) where each point contains (X, Y, Z, R, G, B).
            mask: Boolean mask of shape (H, W) indicating which points to keep.
            
        Returns:
            masked_pointcloud: Tensor of shape (N, 6), where N is the number of points where mask is True.
                            Each point contains (X, Y, Z, R, G, B).
        """
        # Ensure the mask is a boolean tensor
        mask = mask.bool()

        # Use the mask to filter the points in the point cloud
        masked_pointcloud = point_cloud[mask]  # Shape (N, 6), where N is the number of True values in the mask
        
        return masked_pointcloud