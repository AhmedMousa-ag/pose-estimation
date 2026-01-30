from pydantic import BaseModel
from typing import List


# NOTE: self described, no need to document it. Used pydantic for validation and consistency.
class VideoMetadata(BaseModel):
    fps: int
    width: int
    height: int
    frame_count: int


class KeyPoint(BaseModel):
    name: str
    x: float
    y: float
    score: float


class FrameMetadata(BaseModel):
    frame_index: int
    timestamp_ms: int
    pose_score: float
    keypoints: List[KeyPoint]


class VideoResult(BaseModel):
    meta: VideoMetadata
    keypoint_format: List[str] = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]  # These are the default values as per requirements.
    frames: List[FrameMetadata]
