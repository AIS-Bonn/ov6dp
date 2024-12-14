from typing import Any, List, Optional, Tuple

import argparse
import time
import numpy as np
import cv2
import json
import yaml

from dataclasses import dataclass

import numpy as np
import torch

from model_components.object_detector import ObjectDetector
from ov6dp import OV6DP
from datatools.pcl_visualization2 import PCLVisualizer, remove_zero_points

# Load Ice interfaces first before any other ArmarX include

from armarx.ice_manager import get_proxy
from armarx.slice_loader import load_armarx_slice

load_armarx_slice("RobotAPI", "objectpose/ObjectPoseProvider.ice")
load_armarx_slice("RobotAPI", "objectpose/ObjectPoseStorageInterface.ice")

# Now import all ArmarX stuff

from armarx.objpose import ObjectPoseStorageInterfacePrx
from armarx.pose_helper import mat2pose
from armarx.data import ObjectID

from armarx_vision.image_processor import ImageProcessor
from armarx import MetaInfoSizeBase

from armarx_core.time.date_time import DateTimeIceConverter
from armarx.objpose.data import ProvidedObjectPose
from armarx_core.pose_helper import mat2pose


def load_camera_matrix(file_name):
    # Load the JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Extract the camera matrix from the JSON data
    camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)

    return camera_matrix

def convert_2d_bbox_to_3d(bbox, depth_image, camera_intrinsics):
    """
    Convert 2D bounding box to 3D point using depth image and camera intrinsics.

    :param bbox: 2D bounding box in the format [x1, y1, x2, y2].
    :param depth_image: Depth image corresponding to the 2D image.
    :param camera_intrinsics: Camera intrinsics matrix [3x3].
    :return: 3D point as (X, Y, Z).
    """
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Compute the center of the bounding box
    x1, y1, x2, y2 = bbox.cpu().numpy()
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2

    # Get the depth value at the center of the bounding box
    z = depth_image[int(v), int(u)] / 1000
    # if z == 0:  # Check if depth is valid
    #    raise ValueError("Depth value is zero at the center of the bounding box")

    # Convert the 2D center point to a 3D point
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return [x, y, z]


def create_4x4_pose_matrix(tvec, rvec):
    """
    Create a 4x4 transformation matrix from translation vector and rotation vector.

    :param tvec: Translation vector (3,)
    :param rvec: Rotation vector (3,)
    :return: 4x4 transformation matrix
    """
    # Ensure rvec is of the correct type
    rvec = rvec.astype(np.float32)
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Create a 4x4 identity matrix
    pose_matrix = np.eye(4)

    # Set the top-left 3x3 part to the rotation matrix
    pose_matrix[:3, :3] = rotation_matrix

    # Set the top-right 3x1 part to the translation vector
    pose_matrix[:3, 3] = tvec

    return pose_matrix

@dataclass
class OV6DPProcessorConfig:
    weights: str ='best.pt'  # model path or triton URL
    imgsz: List =(640, 640)  # inference size (height, width)
    conf_thres: float =0.85  # confidence threshold
    max_det: int =300  # maximum detections per image
    device: str =''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes: List[int] =None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms: bool =False  # class-agnostic NMS
    augment: bool=False  # augmented inference
    half: bool=False  # update all models
    line_thickness: Optional[int] =None  # bounding box thickness (pixels)
    show_labels: bool=True  # show labels
    show_conf: bool=True  # show confidences
    #provider_name: str = "AzureKinectPointCloudProvider"
    provider_name: str = "OpenNIPointCloudProvider"
    #agent_name: str = "Armar7"
    agent_name: str = "Armar6"
    frame_name: str = "AzureKinect_RGB"
    camera_matrix_file: str=""
    vocab: str="banana. can. apple. plate."


class OV6DPProcessor(ImageProcessor):
    
    def __init__(self, config: OV6DPProcessorConfig):
        self.config = config
        self.vocab = self.config.vocab
        
        super().__init__(provider_name=self.config.provider_name, num_result_images=1)
        
        # Load a OV6DP model
        self.ov6dp = OV6DP()
            
        # FIXME remove
        #self.camera = load_camera_matrix(self.config.camera_matrix_file)
        
        self.pose_storage_prx = get_proxy(ObjectPoseStorageInterfacePrx, "ObjectMemory")
        

    def process_images(self, images: np.ndarray, info: MetaInfoSizeBase) -> np.ndarray | Tuple[np.ndarray, Any]:
        
    
        ts_1 = time.time()
        
        rgb = images[0, :, :, :]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = images[1, :, :, :]
        
        
        # depth_image = np.left_shift(g.astype(np.uint64), 8).astype(np.float64) + r.astype(np.uint64)
        depth_image = (depth[:, :, 1].astype(np.uint16) << 8) + depth[:, :, 0]
        # depth_image /= 1000
        
        ts_2 = time.time()

        # Transform images to pytorch
        rbg_torch = torch.tensor(rgb).permute(2, 0, 1)
        print(torch.tensor(depth).shape)
        depth_torch = torch.tensor(depth_image.astype(int))[None, :, :]
        print(depth_torch.shape, torch.max(depth_torch), torch.min(depth_torch))
        
        # pass images through ov6dp
        results = self.ov6dp.get_poses_from_image(rbg_torch, depth_torch, self.vocab, CLIP_image_encode=False)
        class_names, transform, pointclouds, image_pointcloud = results

        ts_3 = time.time()
        time_provided = int(time.time() * 1e6)
        
        if class_names != None and len(class_names)>0 :
            transform = [t.cpu().numpy() for t in transform]
            ts_4 = time.time()
            
            name_map_dict = {"banana": "YCB/011_banana",
                             "can": "YCB/005_tomato_soup_can",
                             "apple": "YCB/013_apple",
                             "plate": "YCB/029_plate"
                             }

            class_names = [name_map_dict[name] for name in class_names]

            self._provide_poses('OV6DP', class_names, transform, self.config.agent_name, self.config.frame_name, time_provided)

        ts_5 = time.time()

        annotated_frame, annotated_frame_masks = self.ov6dp.get_detection_cv2_images()
        
        ts_6 = time.time()
        
        info.time_provided = time_provided
        
        # print(f"Overall: {ts_5 - ts_1}s")
        # print(f"Preprocessing: {ts_2 - ts_1}s")
        # print(f"Inference: {ts_3 - ts_2}s")
        # print(f"Postprocessing 1: {ts_4 - ts_3}s")
        # print(f"Postprocessing 2: {ts_5 - ts_4}s")
        # print(f"Postprocessing 2: {ts_6 - ts_5}s")
        
        return annotated_frame_masks, info
        
        
        
    def _provide_poses(
            self,
            provider_name: str,
            object_names: List[str],
            poses: List[np.ndarray],
            agent: str,
            object_pose_frame: str,
            timestamp: int = None,
        ):
            """
            Writes object poses to the memory

            :param provider_name: name of the provider
            :param object_name: list of object names
            :param poses: list of object poses
            """
            # pose_topic = get_topic(ObjectPoseTopicPrx, "ObjectPoseTopic")
            
            classCnt = {}
            
            object_poses = []
            for name, pose in zip(object_names, poses):
                pose = mat2pose(pose, agent=agent, frame=object_pose_frame)
                timestamp = timestamp or (time.time() * 1e6)
                dataset, object_name = name.split("/")
                
                if not name in classCnt.keys():
                    classCnt[name] = 0
                else:
                    classCnt[name] += 1
                
                object_id = ObjectID(dataset, object_name, f"{classCnt[name]}")
                time_converter = DateTimeIceConverter()
                t = time_converter.to_ice(timestamp)

                object_pose = ProvidedObjectPose(
                    providerName=provider_name,
                    objectPose=pose,
                    objectPoseFrame=object_pose_frame,
                    objectID=object_id,
                    timestamp=t,

                )
                object_poses.append(object_pose)
                
            r = self.pose_storage_prx.begin_reportObjectPoses(provider_name, object_poses)

            # FIXME do something with r





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default="best.pt", help='model path or triton URL')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720,1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.85, help='confidence threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--line-thickness', default=None, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--show-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--show-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument("--provider_name", default="OpenNIPointCloudProvider", type=str,
                        help="name of the image provider")
    parser.add_argument("--agent_name", default="Armar6", type=str,
                        help="name of the robot/agent")
    parser.add_argument("--frame_name", default="AzureKinectCamera", type=str,
                        help="name of the frame that the pose is reported to")
    parser.add_argument("--camera_matrix_file", default="ARMAR_DE_camera.json", type=str,
                        help="location of the camera matrix file")
    parser.add_argument('--half', default=False, help="Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    
    config = OV6DPProcessorConfig(**vars(opt))
    
    yolo = OV6DPProcessor(config)
    #yolo.run_armarx(**vars(opt))

    yolo.on_connect()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
