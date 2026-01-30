from ultralytics import YOLO
from configs.config import MODEL_NAME, KEYPOINTS_MAPPING
from configs.ml_conf import (
    CONF,
    IOU,
    DEVICE,
    HALF,
    IMG_SIZE,
    VERBOSE,
    MAX_DETECTION,
    AGNOSTIC_NMS,
)

from typing import List, Tuple
from models.video_models import KeyPoint
import torch

# YOLO model to be used for prediction.
model = YOLO(MODEL_NAME)


def get_key_points(img) -> Tuple[List[KeyPoint], float]:
    """Takes an image, extract model prediction and returns a tuple of a list of keypoints for that frame and pose_score."""
    outputs: List[KeyPoint] = []
    results = model.predict(
        img,
        conf=CONF,
        iou=0.45,
        classes=[0],  # Person class only
        device=DEVICE,
        half=HALF,  # 3. Only use half precision on GPU
        imgsz=IMG_SIZE,  # Use smaller image size for faster performance
        verbose=VERBOSE,
        max_det=MAX_DETECTION,  # Requirements is for one person only.
        agnostic_nms=AGNOSTIC_NMS,  # Faster NMS
    )  # predict on an image
    poses_scores = []
    for res in results:
        if res.keypoints is not None:
            # To use numpy, we must move tensors from GPUs if it's on GPU to cpu.
            keypoints = res.keypoints.xy.cpu().numpy()
            confidences = res.keypoints.conf.cpu().numpy()

            for i, person_kpts in enumerate(
                keypoints
            ):  # NOTE: The requirements stated to have only one person, so we can actually skip this iteration.
                # For reference on these keypoints, please visit https://docs.ultralytics.com/tasks/pose/
                for j, (x, y) in enumerate(person_kpts):
                    conf = confidences[i][j]
                    # Only get points that has higher confidence than our configuration.
                    if conf > CONF:
                        outputs.append(
                            KeyPoint(name=KEYPOINTS_MAPPING[j], x=x, y=y, score=conf)
                        )
        if res.boxes is not None and len(res.boxes) > 0:
            # Get the highest confidence detection
            poses_scores.append(
                float(res.boxes.conf.max().cpu().numpy())
            )  # always guranteed to be at least one 1. Just calculating the average.
    pose_score = (
        sum(poses_scores) / float(len(poses_scores)) if len(poses_scores) > 0 else 0.0
    )
    return outputs, pose_score
