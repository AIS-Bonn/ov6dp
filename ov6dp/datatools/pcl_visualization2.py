import pyvista as pv
import numpy as np
import torch



def torch_to_pyvista(pointcloud):
    """
    Converts a torch tensor (N, 3) to a PyVista PolyData object.
    
    Args:
        pointcloud (torch.Tensor): A tensor of shape (N, 3) representing a 3D point cloud.
        
    Returns:
        pyvista.PolyData: A PyVista object representing the point cloud.
    """
    pointcloud_np = pointcloud.cpu().numpy()  # Convert tensor to NumPy array
    return pv.PolyData(pointcloud_np)


class PCLVisualizer():
    def __init__(self, stream=False) -> None:
        self.stream = stream
        self.opened = False
        self.refresh_plotter()


    def visualize_pointclouds(self, pointclouds):
        """
        Visualizes a list of pointclouds (each as a torch tensor of shape (N, 3))
        using PyVista.

        Args:
            pointclouds (list): A list of PyTorch tensors, each of shape (N, 3), 
                                representing different point clouds.
        """
        if not self.opened:
            # First-time setup for the plotter
            self.plotter.clear()

        # Color palette for point clouds
        colors = [
            "#000000",   # Black
            "#0000FF",   # Blue
            "#00FF00",   # Green
            "#FF0000",   # Red
            "#FF8000",   # Orange
            "#00FF80",   # Light Green
            "#8000FF",   # Purple
            "#808080"    # Gray
        ]

        # Loop over the list of point clouds and add them to the plotter
        for i, pcl in enumerate(pointclouds):
            # Convert the PyTorch tensor to a PyVista PolyData object
            pcd = torch_to_pyvista(pcl)

            # Assign a color to the point cloud based on its index
            color = colors[i % len(colors)]  # Cycle through the colors
            self.plotter.add_points(pcd, color=color, point_size=1)

        
        # Enhance UX: Set background color and axes visibility
        self.plotter.set_background('white')  # Set background to white for better visibility
        self.plotter.show_grid()  # Show grid lines for easier orientation
        self.plotter.show_axes()  # Show the axes in the 3D scene

        if not self.stream:
            # Show the interactive plot (non-streaming mode)
            self.plotter.show(interactive=True)
            self.refresh_plotter()
        else:
            if not self.opened:
                # Show plotter window with interactive streaming
                self.plotter.show(interactive=True, interactive_update=True)
                self.opened = True  # Mark the plotter as open
            else:
                # Update the visualization in streaming mode
                self.plotter.render()

    def refresh_plotter(self):
        self.plotter = pv.Plotter()




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
