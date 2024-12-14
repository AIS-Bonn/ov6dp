import json
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from itertools import groupby

class COCOWriter:

    def __init__(self, class_names):
        # Initialize COCO JSON structure
        self.coco_data = {
            "info": {
                "description": "COCO-format dataset",
                "version": "1.0",
                "year": 2024,
                "date_created": "",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        self.image_id = 1
        self.detection_id = 1
        self.category_map = {name: i + 1 for i, name in enumerate(class_names)}
        for name, id in self.category_map.items():
            self.coco_data["categories"].append({
                "id": id,
                "name": name,
                "supercategory": "",
            })


    def convert_to_coco_format(self, image_name, class_names, bboxes, masks):
        """
        Convert object detection and segmentation data to COCO format.

        Parameters:
        - images: List of dictionaries containing image info (e.g., [{'id': 1, 'file_name': 'image1.jpg', 'width': 640, 'height': 480}]).
        - class_names: List of class names corresponding to object detections.
        - bboxes: List of bounding boxes for each image. Each element is a list of bboxes in xyxy format for the respective image.
        - masks: List of binary masks for each image. Each element is a list of binary masks for the respective image.
        - output_path: Path to save the resulting COCO JSON file.

        Returns:
        - None
        """

        # Add image info
        self.coco_data["images"].append({
                "id": self.image_id,
                "file_name": image_name,
                "width": 1280,
                "height": 720,
            })

        for bbox, mask, class_name in zip(bboxes, masks, class_names):
            # Convert xyxy bbox to COCO format (x, y, width, height)
            x_min, y_min, x_max, y_max = bbox
            coco_bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

            # Add annotation
            self.coco_data["annotations"].append({
                "id": self.detection_id,
                "image_id": self.image_id,
                "category_id": self.category_map[class_name],
                "bbox": coco_bbox,
                "area": int(coco_bbox[2] * coco_bbox[3]),  # width * height
                "iscrowd": 0,
                "score": 1.0,
                "segmentation": binary_mask_to_rle(mask),
            })

            self.detection_id += 1

        
        self.image_id += 1
        return
    
    def write(self, path):
        with open(path, 'w') as f:
            json.dump(self.coco_data, f, indent=4)

def mask_to_coco_format(binary_mask):
    """
    Convert a binary mask to COCO RLE segmentation format.

    Parameters:
    - binary_mask: Binary mask as a 2D numpy array.

    Returns:
    - RLE or polygon representation of the mask.
    """
    from pycocotools import mask as coco_mask
    rle = coco_mask.binary_m(np.asfortranarray(binary_mask.cpu().numpy().astype('uint8')))
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert byte counts to string
    return rle

def binary_mask_to_rle(binary_mask):
    binary_mask = binary_mask.cpu().numpy()
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def write_transform(matrices, class_names, filename):
    """
    Writes the translation components and orientation in XYZ Euler angles
    from a 4x4 transformation matrix to a file.

    Parameters:
    - matrix (torch.Tensor): 4x4 homogeneous transformation matrix as a PyTorch tensor.
    - filename (str): Path to the output file.

    Returns:
    - None
    """
    
    with open(filename, 'w') as file:
        for class_name, matrix in zip (class_names, matrices):

            # Ensure the matrix is on the CPU and convert to numpy
            matrix = matrix.cpu().numpy()

            # Extract translation components (x, y, z)
            translation = matrix[:3, 3]

            # Extract the rotation matrix
            rotation_matrix = matrix[:3, :3]

            # Convert rotation matrix to Euler angles in XYZ order
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)  # Degrees for human readability

            # Write to file
            file.write(f"{class_name} {translation[0]} {translation[1]} {translation[2]} {euler_angles[0]} {euler_angles[1]} {euler_angles[2]}\n")


def decode_coco_rle(rle):
    """
    Decode a COCO RLE into a binary mask.

    Parameters:
    - rle: COCO RLE format with 'counts' as a string and 'size' as [height, width].

    Returns:
    - Binary mask as a 2D numpy array.
    """
    counts = list(map(int, rle['counts'].split()))
    height, width = rle['size']
    
    # Initialize an empty mask
    mask = np.zeros(height * width, dtype=np.uint8)
    
    # Decode the RLE counts
    start = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Foreground runs
            mask[start:start+count] = 1
        start += count
    
    # Reshape the flat mask to the original 2D shape
    mask = mask.reshape((height, width))
    return mask


if __name__ == "__main__":
    # Example usage
    example_rle = {
        'counts': '3 2 5',
        'size': [3, 3]
    }
    binary_mask = decode_coco_rle(example_rle)
    print(binary_mask)