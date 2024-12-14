import matplotlib
import torch
import cv2
import numpy as np


import pykinect_azure as pykinect


from model_components.object_detector import ObjectDetector
from ov6dp import OV6DP
from datatools.pcl_visualization2 import PCLVisualizer, remove_zero_points

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	# print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	flag = False
	while not flag:
		# Get capture
		capture = device.update()

		# Get the color image from the capture
		ret_color, color_image = capture.get_color_image()

		# Get the colored depth
		ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

		if  ret_color and ret_depth:
			flag = True

	print(type(color_image), type(transformed_depth_image))
	print(color_image.shape)
	print(transformed_depth_image.shape)
	print(color_image.dtype, np.max(color_image), np.min(color_image))
	print(transformed_depth_image.dtype, np.max(transformed_depth_image), np.min(transformed_depth_image))

	color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
	color_image = torch.tensor(color_image).permute(2, 0, 1)
	transformed_depth_image = torch.tensor(transformed_depth_image).float()
	depth_image = torch.zeros(1, transformed_depth_image.shape[0], transformed_depth_image.shape[1])
	depth_image[0] = transformed_depth_image

	ov6dp = OV6DP()
	voacb = "banana. tomato can. apple."
	class_names, transform, pointclouds, image_pointcloud = ov6dp.get_poses_from_image(color_image, depth_image, voacb)
	
	print(class_names)
	print(len(pointclouds))
	

	pcls = [pcl for pcl in pointclouds]
	pcls.insert(len(pcls), remove_zero_points(image_pointcloud))
	

	for pcl in pcls:
		print(pcl.shape)
	print("Now Visualizing")
	
	pcl_vis = PCLVisualizer()
	pcl_vis.visualize_pointclouds(pcls)
