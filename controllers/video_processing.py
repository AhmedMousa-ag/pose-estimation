import cv2
from controllers.yolo import get_key_points
from models.video_models import VideoResult, VideoMetadata, FrameMetadata
from typing import List
import time


def process_video(video_path: str) -> VideoResult:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(
            f"Faild to open video at path: {video_path}, please check if the video exists"
        )
    video_metadata = __extract_video_metadata(video)
    skip_frames = False
    if (
        video_metadata.fps > 30
    ):  # for optimization, no need to process more than 30 frames per second.
        print(f"FPS: {video_metadata.fps} and will skip half these frames.")
        skip_frames = True

    frame_count = 0
    frames_metadata: List = []
    avg_frame_proc_time = 0.0
    calculated_frame_time_count = 1
    while video.isOpened():
        if (
            skip_frames and frame_count % 2
        ):  # for optimization, no need to process more than 30 frames per second.
            frame_count += 1
            continue
        start_time = time.time()
        success, img = video.read()
        if not success:
            print("Video frame is empty or has been successfully processed.")
            break
        frame_count += 1
        timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
        keypoints, pose_score = get_key_points(img)
        frames_metadata.append(
            FrameMetadata(
                frame_index=frame_count,
                timestamp_ms=int(timestamp_ms),
                pose_score=pose_score,
                keypoints=keypoints,
            )
        )
        end_time = time.time()
        avg_frame_proc_time += (
            end_time - start_time
        ) / calculated_frame_time_count  # Guranteed to be 1 at least.
        calculated_frame_time_count += 1
    print(f"Processed frame time average is: {avg_frame_proc_time:.4f} seconds ")

    return VideoResult(meta=video_metadata, frames=frames_metadata)


def __extract_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(fps=fps, width=width, height=height, frame_count=frame_count)
