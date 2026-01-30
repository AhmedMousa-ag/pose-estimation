from pydantic import BaseModel
from typing import List


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
    ]
    frames: List[FrameMetadata]
