import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def get_package_path():
    # Get the current file's directory (assuming this script is part of a package)
    current_file_path = os.path.abspath(__file__)

    # Navigate to the parent directory to find the package's root (assuming it's a simple package)
    package_directory = os.path.dirname(current_file_path)

    return Path(package_directory) / "../.."


CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

TEXT_PROMPT = "car. tire."
IMG_PATH = get_package_path() / "truck.jpg"


class ObjectDetector:

    def __init__(self) -> None:
        """
        Parameters
        """
        grounding_model = "IDEA-Research/grounding-dino-tiny"
        sam2_checkpoint = get_package_path() / "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.output_dir = get_package_path() / Path("outputs")

        self.box_threshold = 0.3
        self.text_threshold = 0.3
        
        # create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # environment settings
        # use bfloat16
        torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # build SAM2 image predictor
        sam2_checkpoint = sam2_checkpoint
        model_cfg = sam2_model_config
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino from huggingface
        model_id = grounding_model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    
    def detect_on_image(self, image: torch.Tensor, vocabulary: str):
        """
        image needs be in torch rgb (3, H, W) in range [0, 255]
        setup the input image and text prompt for SAM 2 and Grounding DINO
        VERY important: text queries need to be lowercased + end with a dot
        """
        text = vocabulary
        image = image.permute(1, 2, 0).cpu().numpy()
        #print(image.shape)
        image = Image.fromarray(image)

        self.sam2_predictor.set_image(np.array(image.convert("RGB")))

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )

        """
        Results is a list of dict with the following structure:
        [
            {
                'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
                'labels': ['car', 'tire', 'tire', 'tire'], 
                'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                                [1392.4701,  554.4064, 1628.6133,  777.5872],
                                [ 436.1182,  621.8940,  676.5255,  851.6897],
                                [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
            }
        ]
        """

        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )


        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = torch.from_numpy(masks)


        confidences = results[0]["scores"].cpu()
        class_names = results[0]["labels"]

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        print("\n".join(labels))
                
        return class_names, confidences, input_boxes, masks 


    def save_annotated_image(self, image: torch.Tensor, class_names, confidences, input_boxes, masks):
        
        
        annotated_frame, annotated_frame_masks = self.annotate_image(iamge=image, 
                                                                     class_names=class_names,
                                                                     input_boxes=input_boxes,
                                                                     masks=masks)

        cv2.imwrite(os.path.join(self.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_masks)
        return

    def annotate_image(self, image: torch.Tensor, class_names, confidences, input_boxes, masks):
        """
        image needs be in torch rgb (3, H, W) in range [0, 255]
        Visualize image with supervision useful API
        """ 
        img = cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
        if class_names == None:
            return img, img
        class_ids = np.array(list(range(len(class_names))))
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.bool().cpu().numpy(),  # (n, h, w)
            class_id=class_ids
        )

        """
        Note that if you want to use default color map,
        you can set color=ColorPalette.DEFAULT
        """
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        #cv2.imwrite(os.path.join(self.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame_masks = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        #cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        return annotated_frame, annotated_frame_masks
