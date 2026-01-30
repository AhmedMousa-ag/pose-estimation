from ultralytics import YOLO
from configs.config import MODEL_NAME, CONF, KEYPOINTS_MAPPING
from typing import List, Tuple
from models.video_models import KeyPoint


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
        device="cuda:0",
        half=True,
        verbose=False,
    )  # predict on an image
    pose_score = 0.0
    num_poses = 1
    for res in results:
        if res.keypoints is not None:
            # NOTE this assume we are using PyTorch
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
            pose_score += (
                float(res.boxes.conf.max().cpu().numpy()) / num_poses
            )  # always guranteed to be at least one 1. Just calculating the average.
            num_poses += 1

    return outputs, pose_score
