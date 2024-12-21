import torch
import copy

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def pca_align(source: torch.Tensor, target: torch.Tensor):
    """
    Perform PCA-based alignment between two point clouds (source and target).
    Args:
    - source: (N, 3) tensor representing the source point cloud.
    - target: (N, 3) tensor representing the target point cloud.
    
    Returns:
    - R: (3, 3) rotation matrix for initial alignment.
    - t: (3,) translation vector for initial alignment.
    """
    # Center the point clouds
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    source_centered = source - source_center
    target_centered = target - target_center

    # Compute the covariance matrices
    cov_source = torch.matmul(source_centered.T, source_centered)
    cov_target = torch.matmul(target_centered.T, target_centered)

    # Eigen decomposition for PCA
    print(cov_source.float().dtype)
    _, source_eigvec = torch.linalg.eigh(cov_source.float())
    _, target_eigvec = torch.linalg.eigh(cov_target.float())

    # The initial rotation matrix aligns the principal components
    R = torch.matmul(target_eigvec, source_eigvec.T)
    
    # The initial translation aligns the centers
    t = target_center - torch.matmul(R, source_center)

    return R, t

def apply_transformation(points, R, t):
    """
    Apply a rigid transformation to a point cloud.
    Args:
    - points: (N, 3) tensor representing the point cloud.
    - R: (3, 3) rotation matrix.
    - t: (3,) translation vector.
    
    Returns:
    - transformed_points: (N, 3) tensor of transformed points.
    """
    return torch.matmul(points, R.T) + t

def invert_transformation(R, t):
    """
    Compute the inverse of a rigid transformation defined by a rotation matrix R and a translation vector t.
    
    Args:
        R (torch.Tensor): (3, 3) rotation matrix.
        t (torch.Tensor): (3,) translation vector.
        
    Returns:
        R_inv (torch.Tensor): (3, 3) inverse rotation matrix.
        t_inv (torch.Tensor): (3,) inverse translation vector.
    """
    R_inv = R.T  # Inverse of a rotation matrix is its transpose
    t_inv = -torch.matmul(R_inv, t)  # Inverse translation
    return R_inv, t_inv

def apply_inverse_transformation(points, R, t):
    """
    Apply the inverse of a rigid transformation to a point cloud.
    
    Args:
        points (torch.Tensor): (N, 3) tensor representing the point cloud.
        R (torch.Tensor): (3, 3) rotation matrix of the original transformation.
        t (torch.Tensor): (3,) translation vector of the original transformation.
        
    Returns:
        torch.Tensor: (N, 3) tensor of points after applying the inverse transformation.
    """
    # Get inverse transformation parameters
    R_inv, t_inv = invert_transformation(R, t)
    
    # Apply the inverse transformation
    return torch.matmul(points, R_inv.T) + t_inv

def icp(source, target, max_iterations=20, tolerance=1e-5):
    """
    Perform ICP refinement to align source to target.
    Args:
    - source: (N, 3) tensor representing the source point cloud.
    - target: (N, 3) tensor representing the target point cloud.
    - max_iterations: maximum number of ICP iterations.
    - tolerance: tolerance for stopping criteria based on error improvement.
    
    Returns:
    - R: refined rotation matrix.
    - t: refined translation vector.
    """
    
    # Initialize transformation on device
    R, t = torch.eye(3, device=device, dtype=torch.float32), torch.zeros(3, device=device, dtype=torch.float32)
    prev_error = float('inf')
    
    for i in range(max_iterations):
        # Apply current transformation
        source_transformed = apply_transformation(source, R, t)
        
        # Find nearest neighbors in target
        distances = torch.cdist(source_transformed, target)
        closest_idx = torch.argmin(distances, dim=1)
        closest_points = target[closest_idx]
        
        # Compute cross-covariance matrix for refinement
        source_centered = source_transformed - source_transformed.mean(dim=0)
        closest_centered = closest_points - closest_points.mean(dim=0)
        H = torch.matmul(source_centered.T, closest_centered)
        
        # SVD to get the rotation refinement
        U, _, Vt = torch.linalg.svd(H.float())
        R_icp = torch.matmul(Vt.T, U.T)
        
        # Fix reflection case
        if torch.linalg.det(R_icp.float()) < 0:
            Vt[-1, :] *= -1
            R_icp = torch.matmul(Vt.T, U.T)
        
        # Update rotation and translation
        t_icp = closest_points.mean(dim=0) - torch.matmul(R_icp, source_transformed.mean(dim=0))
        
        # Update the cumulative transformation
        R = torch.matmul(R_icp, R)
        t = torch.matmul(R_icp, t) + t_icp

        # Calculate mean squared error and check convergence
        error = torch.mean(torch.norm(source_transformed - closest_points, dim=1) ** 2)
        if torch.abs(prev_error - error) < tolerance:
            break
        prev_error = error
        print(f"{i}: {error}")

    return R, t, prev_error

def to_homogeneous_matrix(R, t):
    """
    Combine rotation and translation into a 4x4 homogeneous transformation matrix.
    Args:
    - R: (3, 3) rotation matrix.
    - t: (3,) translation vector.
    
    Returns:
    - T: (4, 4) homogeneous transformation matrix.
    """
    T = torch.eye(4, device=device, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def voxel_downsample(points, voxel_size):
    # Divide each point by voxel size and floor to get voxel grid indices
    voxel_indices = torch.floor(points / voxel_size)
    # Find unique voxel grid indices and keep one point per voxel
    unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
    # Compute centroids or mean points within each voxel
    downsampled_points = torch.stack([points[inverse_indices == i].mean(dim=0) for i in range(len(unique_voxels))])
    return downsampled_points


def match_pointclouds(source_pcl: torch.Tensor, target_pcl: torch.Tensor):
    source = voxel_downsample(copy.deepcopy(source_pcl).float().to(device), 1)
    target = voxel_downsample(copy.deepcopy(target_pcl).float().to(device), 1)
    
    # Step 1: PCA Initial Alignment
    print("PCA align...")
    R_pca, t_pca = pca_align(source, target)

    # Step 2: Apply Initial Transformation
    source_aligned = apply_transformation(source, R_pca, t_pca)
    #target_aligned = apply_inverse_transformation(target, R_pca, t_pca)

    # Step 3: ICP Refinement
    print("ICP align...")
    R_icp, t_icp, error = icp(source_aligned, target, max_iterations=25, tolerance=1e-5)

    # Final transformation
    R_final = torch.matmul(R_icp, R_pca)
    t_final = torch.matmul(R_icp, t_pca) + t_icp
    R_final = R_final.float()
    t_final = t_final.float()

    # Create 4x4 transformation matrix
    #T_final = to_homogeneous_matrix(R_final, t_final).inverse()
    T_final = to_homogeneous_matrix(R_final.inverse(), -R_final.inverse() @ t_final).inverse()

    # Apply final transformation to the original source point cloud
    source_transformed = apply_transformation(source_pcl.to(device).float(), R_final, t_final).float()
    #target_transformed = apply_inverse_transformation(target_pcl.to(device).float(), R_final, t_final).float()

    #return target_transformed, T_final, error, target_aligned
    return source_transformed, T_final, error, source_aligned
    

def uniform_rotation_samples(n=5):

    u = torch.arange(0, n).float() / n
    n0 = torch.zeros_like(u)
    n1 = torch.ones_like(u)
    u2pi = u * 2 * torch.pi
    usin = torch.sin(u2pi)
    ucos = torch.cos(u2pi)
    usqrt = torch.sqrt(u)
    
    v = torch.tensor([torch.kron(ucos, usqrt),
                      torch.kron(usin, usqrt),
                      torch.kron(n1, torch.sqrt(1-u))]) # (3, n^2)

    # v01 = v[0, :] * v[1, :]
    # v02 = v[0, :] * v[2, :]
    # v12 = v[1, :] * v[2, :]
    # H = torch.tensor([[v[0, :]^2, v01, v02],
    #                   [v01, v[1, :]^2, v12],
    #                   [v02, v12, v[2, :]^2]]) #(3, 3, n^2)
    v = v.permute(1, 0)
    H = torch.matmul(v[:, :, None], v[: None, :])

    R = torch.tensor([[ucos, usin, n0],
                      [-usin, ucos, n0],
                      [n0, n0, n1]]) #(3, 3, n)
    R = R.permute(2, 0, 1)

    M = -torch.matmul(H[None, ...], R[:, None, :, :])
    M.reshape(n**3, 3, 3)
    return M

# Example usage
if __name__ == "__main__":
    # Assuming source and target are (N, 3) PyTorch tensors representing two point clouds
    source = torch.randn((1000, 3), device=device, dtype=torch.float32)  # Example source point cloud
    target = torch.randn((1000, 3), device=device, dtype=torch.float32)  # Example target point cloud

    # Match point clouds and get final aligned source, transformation matrix, and error
    source_transformed, T_final, error, source_aligned = match_pointclouds(source, target)
    print("Transformation Matrix:\n", T_final)
    print("Alignment Error:", error)
