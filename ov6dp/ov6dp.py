import math
import time
from tkinter import E
import numpy as np
import json
import torch
import torch.nn as nn
from torchvision.transforms.functional import crop, to_pil_image, to_tensor
from torchvision import transforms
import os
import re
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
from pathlib import Path
import pathlib
from sklearn.metrics.pairwise import cosine_similarity
import cv2

from model_components.object_detector import ObjectDetector
#from model_components.pcl_matching import match_pointclouds
from model_components.torch_matching import match_pointclouds 
from datatools.object_model_processor import read_ply_to_tensor, PointcloudCreator
from datatools.pcl_visualization import visualize_pointclouds, remove_zero_points
from coco_writer import COCOWriter, write_transform

OBJECT_NAMES = [
    "A bottle of Livio oil.",
    "A Senseo coffee machine.",
    "A pack of Potatoe Dumplings.",
    "A box of peppermint tea.",
    "A box of apple tea.",
    "A green mixing bowl.",
    "A pack of Edeka coffee filters.",
    "A carton of soy drink.",
    "A green cup.",
    "A packet of Instant Sauce.",
    "A Powdered Sugar Mill.",
    "A carton of Milk Drink Vanilla.",
    "A box of wildberry tea.",
    "A pack of corny cereal bars.",
    "A frying pan.",
    "A carton of bio-milk.",
    "A carton of milk.",
    "plate.",
    "An egg whisk.",
    "A small mixing bowl.",
    "A can of corn.",
    "A ladle.",
    "A salad spoon.",
    "A box of Milk Rice.",
    "fork.",
    "A tablespoon.",
    "A cutting board.",
    "A pack of Melitta coffee filters.",
    "A large cup.",
    "A black knife.",
    "A bottle of apple juice.",
    "A draining rack.",
    "A can of chips.",
    "A Master Chef can.",
    "A box of crackers.",
    "A box of sugar.",
    "A can of tomato soup.",
    "A bottle of mustard.",
    "A can of tuna fish.",
    "A box of pudding.",
    "A box of gelatin.",
    "A can of potted meat.",
    "banana.",
    "A strawberry.",
    "apple.",
    "A lemon.",
    "A peach.",
    "A Knaeckebrot Rye.",
    "A box of organic herb tea.",
    "A large box of Ravioli.",
    "A Hot Pot 2.",
    "A bottle of Fruit Drink.",
    "A pack of j-cups.",
    "A pear.",
    "A Danish Ham.",
    "A pack of Mashed Potatoes.",
    "A pack of Amicelli.",
    "A pack of Cough Drops Berries.",
    "A pack of Instant Soup.",
    "A Green Cup.",
    "A pack of Potatoe Sticks.",
    "A Hering Tin.",
    "A box of organic fruit tea.",
    "An orange.",
    "A plum.",
    "A pack of i-cups.",
    "A can of Chicken Soup.",
    "A pack of Cough Drops Lemon.",
    "A pitcher base.",
    "A pack of Koala Candy.",
    "A Yellow Salt Cube 2.",
    "A pack of Cough Drops Honey.",
    "A pack of Instant Ice Coffee.",
    "bowl.",
    "A pack of Fennel Tea.",
    "A pack of Choc Marshmallows.",
    "A mug.",
    "A pack of Fruit Tea.",
    "A pack of Ceylon Tea.",
    "A Flower Cup.",
    "A pack of h-cups.",
    "A pack of Instant Tomato Soup.",
    "A pack of g-cups.",
    "A pack of Instant Mousse.",
    "A pack of Broccoli Soup.",
    "A pack of Coffee Filters 2.",
    "A pack of Sauce Thickener.",
    "A pack of Herb Salt.",
    "A pack of Rice.",
    "soup can.",
    "A pack of Fruit Bars.",
    "A pack of Sweetener.",
    "A pack of Potatoe Starch.",
    "A Blue Salt Cube.",
    "A pack of e-cups.",
    "A Coke Plastic Small Grasp.",
    "A bottle of Tomato Sauce.",
    "A pack of Powdered Sugar.",
    "A pack of Rusk.",
    "A Yellow Salt Cylinder.",
    "A pack of Strawberry Porridge.",
    "A pack of Nut Candy.",
    "A skillet.",
    "A pack of Muesli Bars.",
    "A pack of Jam Sugar.",
    "A bottle of Tomato Herb Sauce.",
    "A spatula.",
    "A bottle of Livio Sunflower Oil.",
    "A pack of Soft Cake Orange.",
    "A pack of Choc Sticks.",
    "A pack of Coffee Cookies."
]

OBJECT_DICT_OLD = {obj: num+1 for num, obj in enumerate(OBJECT_NAMES)}

OBJECT_NAMES = [
    "banana.",
    "fork.",
    "plate.",
    "apple.",
    "bowl.",
    "soup can."
]

OBJECT_DICT = {obj: OBJECT_DICT_OLD[obj] for obj in OBJECT_NAMES}


def combine_and_invert_masks(mask_list):
    """
    Combines a list of binary masks and returns the inverse of the combined mask.
    
    Args:
        mask_list (list of torch.Tensor): A list of boolean or binary masks (0/1) of the same shape.
        
    Returns:
        torch.Tensor: The inverse of the combined mask.
    """
    # Start with the first mask in the list as the initial combined mask
    combined_mask = mask_list[0].clone()
    
    # Apply logical OR with each mask in the list to combine them
    for mask in mask_list[1:]:
        combined_mask = torch.logical_or(combined_mask, mask)
    
    # Return the inverted combined mask
    return torch.logical_not(combined_mask)

class OV6DP(nn.Module):

    def __init__(self, camera_matrix=torch.Tensor([[900.27978516,   0.        , 960.        ],
                                               [  0.        , 900.07507324, 540.        ],
                                               [  0.        ,   0.        ,   1.        ]]), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.object_detector = ObjectDetector()
        self.pcl_creator = PointcloudCreator(camera_matrix=camera_matrix)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


        self.data_folder = pathlib.Path(__file__).parent.parent.resolve() / "models"
        self.model_files = self.get_model_files(self.data_folder)
        # self.model_pointclouds = [read_ply_to_tensor(file) for file in model_files]

        self.model_name_clip_encodings = self.get_clip_embeddings_text(OBJECT_NAMES)
        self.model_mesh_dict = {}

        self.class_names = None 
        self.confidences = None 
        self.input_boxes = None 
        self.masks= None 
        self.rgb_image = None

    def get_model_files(self, directory: Path):
        # Define a regular expression pattern to match file names like 'obj_000007.ply'
        pattern = re.compile(r'^obj_\d{6}\.ply$')
        
        # List to store the matching file paths
        matching_files = []
        
        # Traverse the directory
        for file in directory.iterdir():
            # Check if it is a file and matches the pattern
            if file.is_file() and pattern.match(file.name):
                matching_files.append(file.resolve())  # Get the full path
        
        # Sort the matching files alphabetically
        matching_files.sort()

        return matching_files
    
    def read_object_names(file_path):
        # Initialize an empty list to store names
        names = []
        
        # Open the file and read the lines
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into number and name
                parts = line.strip().split(maxsplit=1)
                if len(parts) > 1:  # Ensure there is a name part
                    names.append(parts[1])  # Append only the name part
        
        return names
    
    # Function to get embeddings for a list of strings
    def get_clip_embeddings_text(self, strings):
        # Tokenize and preprocess the input text
        inputs = self.clip_processor(text=strings, return_tensors="pt", padding=True)

        # Get the text embeddings from the CLIP model
        with torch.no_grad():
            embeddings = self.clip_model.get_text_features(**inputs)

        # Normalize the embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    #Function to get image features for a specified bounding box
    def get_clip_embeddings_images(self, image_tensor, bounding_boxes):
        """
        Extract CLIP embeddings for multiple regions of an image in parallel.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (3, H, W).
            bounding_boxes (list of tuples): List of tuples (x_min, y_min, x_max, y_max) defining the bounding boxes.

        Returns:
            torch.Tensor: A tensor of normalized CLIP image embeddings, one for each bounding box.
        """
        # Ensure the input is a torch tensor of shape (3, H, W)
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError("Input image must be a torch tensor of shape (3, H, W).")

        # Crop all regions and convert to PIL images
        cropped_regions = []
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox.astype(int)
            region = crop(image_tensor, top=y_min, left=x_min, height=y_max - y_min, width=x_max - x_min)
            region_pil = to_pil_image(region)
            cropped_regions.append(region_pil)

        # Preprocess all cropped regions together
        inputs = self.clip_processor(images=cropped_regions, return_tensors="pt", padding=True)

        # Get the image embeddings from the CLIP model in parallel
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)

        # Normalize the embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    def get_detection_cv2_images(self):
        return self.object_detector.annotate_image(self.rgb_image, 
                                                   self.class_names, 
                                                   self.confidences, 
                                                   self.input_boxes, self.masks)
    

    def get_poses_from_image(self, rgb_image, depth_image, vocabulary, CLIP_image_encode=False):

        print(rgb_image.shape)
        print(depth_image.shape)

        try:
            class_names, confidences, input_boxes, masks = self.object_detector.detect_on_image(rgb_image, vocabulary)
        except AssertionError:
            print("Warning: Assertion failed in detect_on_image")
            class_names = None
            confidences = None
            input_boxes = None
            masks = None

        self.class_names = class_names 
        self.confidences = confidences 
        self.input_boxes = input_boxes 
        self.masks= masks
        self.rgb_image = rgb_image

        if class_names is None or class_names == []:
            print("Result: No detections found")
            return None, None, None, None

        image_pointcloud = self.pcl_creator.image_to_pointcloud(depth_image)
        detection_pointclouds = [self.pcl_creator.extract_masked_pointcloud(image_pointcloud, mask) for mask in masks]
        image_pointcloud = image_pointcloud[combine_and_invert_masks(masks)]
        print("SHAPE:", image_pointcloud.shape)
        #visualize_pointclouds(detection_pointclouds)

        
        if CLIP_image_encode:
            clip_encoded_names = self.get_clip_embeddings_images(rgb_image, input_boxes)
        else:
            clip_encoded_names = self.get_clip_embeddings_text(class_names)

        self.model_name_clip_encodings

        similarities = cosine_similarity(clip_encoded_names.cpu().numpy(), self.model_name_clip_encodings.cpu().numpy())
        similarities = torch.as_tensor(similarities)
        best_indices = similarities.argmax(dim=1)

        for i in range(len(class_names)):
            print(f"'{class_names[i]}' matched to '{OBJECT_NAMES[best_indices[i]]}'")

        transforms, pointclouds, fitness_list = [], [], []
        #return class_names, transforms, pointclouds, image_pointcloud, input_boxes, masks

        for i, name in enumerate(class_names):
            print(f"matching for {name}")
            object_pcl = detection_pointclouds[i]
            #model_pcl = self.model_pointclouds[best_indices[i]]
            model_pcl = self.model_mesh_dict.get(OBJECT_NAMES[best_indices[i]], None)
            if model_pcl is None:
                j = OBJECT_DICT[OBJECT_NAMES[best_indices[i]]]
                model_pcl = read_ply_to_tensor(self.data_folder / f"obj_{str(j).zfill(6)}.ply")
                self.model_mesh_dict[OBJECT_NAMES[best_indices[i]]] = model_pcl

            #result_pcl, transform, fitness, translated_copy = match_pointclouds(source_pcl=object_pcl, target_pcl=model_pcl)
            result_pcl, transform, fitness, translated_copy = match_pointclouds(source_pcl=model_pcl, target_pcl=object_pcl)

            transforms.append(transform)
            pointclouds.append(result_pcl)
            #pointclouds.append(translated_copy)
            pointclouds.append(object_pcl)
            fitness_list.append(fitness)

            print(f"{name}: {fitness}\n{transform}")

        return class_names, transforms, pointclouds, image_pointcloud, input_boxes, masks


def load_image(path):
    # Load the image using PIL
    image = Image.open(path)

    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    return image_tensor

def load_image_depth(path):
    # Load the image using PIL
    image = np.load(path).astype(int)

    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(image)
    #print(image_tensor.shape)
    return image_tensor

def depth_shenanigains(depth_image):
    new_image = torch.zeros(1, depth_image.shape[1], depth_image.shape[2])
    new_image[0] = (depth_image[0].float() + depth_image[1].float() * (2**8) + depth_image[2].float() * (2**16)) / 1000

    return new_image

def match_by_initials(source_list, target_list):
    """
    Matches strings from source_list with strings in target_list based on initial characters.

    Args:
        source_list (list of str): List of source strings.
        target_list (list of str): List of target strings to be matched.

    Returns:
        list of str: Matched strings from target_list.
    """
    result = []
    for source in source_list:
        if "soup" in source:
            result.append("campbells_soup_can")
        else:
            for target in target_list:
                if target.lower().startswith(source.lower()[:len(source)]):
                    result.append(target)
    return result

if __name__ == "__main__":
    #rgb_image_path = Path("D:/data/ov6dp/KITchen_rgb_1/000/000000.png")
    #depth_image_path = Path("D:/data/ov6dp/KITchen_depth_1/000/000000.png")
    #voacb = "banana. soup-can."
    classes = ["bowl", "plate", "banana", "fork", "campbells_soup_can", "apple"]
    vocab = "bowl. plate. banana. fork. soup can. apple."
    name_map_dict ={"bowl": "024_bowl",
                    "banana": "011_banana",
                    "campbells_soup_can": "005_tomato_soup_can",
                    "apple": "013_apple",
                    "plate": "029_plate",
                    "fork": "030_fork"}

    data_folder = pathlib.Path(__file__).parent.parent.resolve() / "saved_images"
    output_folder = pathlib.Path(__file__).parent.parent.resolve() / "outputs"

    ov6dp = OV6DP()
    coco_writer = COCOWriter(classes)

    for i in range(1):
        rgb_image_path = data_folder / f"image_rgb_{i}.jpg"
        depth_image_path = data_folder / f"image_depth_{i}.npy"

        rgb_image = load_image(rgb_image_path)
        depth_image = load_image_depth(depth_image_path)[None, ...]

        print(rgb_image.shape, rgb_image.max(), rgb_image.min())
        print(depth_image.shape, depth_image.max(), depth_image.min())
        #print(depth_image[0].shape, depth_image[0].max(), depth_image[0].min())
        #print(depth_image[1].shape, depth_image[1].max(), depth_image[1].min())
        #print(depth_image[2].shape, depth_image[2].max(), depth_image[2].min())

        #depth_image = depth_shenanigains(depth_image)
        #print(depth_image.shape, depth_image.dtype, depth_image.max(), depth_image.min())
        class_names, transform_matrices, pointclouds, image_pointcloud, input_boxes, masks = ov6dp.get_poses_from_image(rgb_image, depth_image, vocab)
        #annotated_frame, annotated_frame_masks = ov6dp.get_detection_cv2_images()
        #cv2.imshow("annotated_iamge", annotated_frame_masks)

        coco_names = match_by_initials(class_names, classes)
        #coco_writer.convert_to_coco_format(f"image_rgb_{i}.jpg", coco_names, input_boxes, masks)
        transform_names = [name_map_dict[name] for name in coco_names]
        #write_transform(transform_matrices, transform_names, output_folder / f"image_rgb_{i}annotations.txt")

    #coco_writer.write(output_folder / f"coco_detections.json")

    pcls = [pcl for pcl in pointclouds]
    pcls.insert(0, remove_zero_points(image_pointcloud))

    visualize_pointclouds(pcls)
