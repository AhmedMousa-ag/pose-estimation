import cv2
from controllers.yolo import get_key_points
from models.video_models import VideoResult, VideoMetadata, FrameMetadata
from typing import List
import time
from configs.config import SKELETON_CONNECTIONS


# NOTE: The only reason I combined the logic of extracting pose estimation and writing a video in the same place is to be able to calculate the frame average time, otherwise I would have splitted it.
def process_video(video_path: str, output_video_path: str) -> VideoResult:
    """
    Process video frame by frame, detects pose estimation, saves annotated video and returns Video annotation results model.

    Args:
        video_path: Local path for the video to be processed.
        output_video_path: The destination path where the annotated video will be saved.

    Returns:
        A VideoResult object containing pos estimation metadata as per the provided schema.
    """
    video = cv2.VideoCapture(video_path)  # Opens video.
    if not video.isOpened():  # Check if the video is valid.
        raise FileNotFoundError(
            f"Faild to open video at path: {video_path}, please check if the video exists"
        )
    video_metadata = __extract_video_metadata(video)  # Extracts video metadata.
    output_video_fbs = video_metadata.fps  # Used for the video output.
    skip_frames = False  # For optimization.
    if (
        video_metadata.fps > 30
    ):  # for optimization, no need to process more than 30 frames per second.
        print(f"FPS: {video_metadata.fps} and will skip half these frames.")
        skip_frames = True
        output_video_fbs /= 2  # Since we are annotating half the frames, we should reduce the output video to the same half of frames.
    output_video = cv2.VideoWriter(  # Annotated video that would have the output.
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_video_fbs,
        (video_metadata.width, video_metadata.height),
    )
    frame_count = 0  # Keep track of it for the metadata.
    frames_metadata: List = []  # Will be saved with the VideoResult model.
    frame_times = []  # Used to calculate the average frame time.
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
        frame_metadata = FrameMetadata(
            frame_index=frame_count,
            timestamp_ms=int(timestamp_ms),
            pose_score=pose_score,
            keypoints=keypoints,
        )
        frames_metadata.append(frame_metadata)
        output_video.write(annotate_frame(img, frame_metadata))
        end_time = time.time()
        frame_times.append(end_time - start_time)
    print(
        f"Processed frame time average is: {sum(frame_times)/len(frame_times):.4f} seconds "
    )
    # Clean up
    video.release()
    output_video.release()
    cv2.destroyAllWindows()
    return VideoResult(meta=video_metadata, frames=frames_metadata)


def __extract_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    """Extracts metadata from video and returns VideoMetadata object"""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(fps=fps, width=width, height=height, frame_count=frame_count)


def annotate_frame(frame, frame_metadata: FrameMetadata):
    """Annotate a frame as per the extract pose estimation."""
    # 1. Create a dictionary to look up KeyPoint objects by their name
    keypoint_map = {kp.name: kp for kp in frame_metadata.keypoints}

    # 2. Draw Skeleton Connections first (so lines are behind the dots)
    for start_name, end_name in SKELETON_CONNECTIONS:
        if start_name in keypoint_map and end_name in keypoint_map:
            kp_start = keypoint_map[start_name]
            kp_end = keypoint_map[end_name]

            # Convert to int for OpenCV
            start_point = (int(kp_start.x), int(kp_start.y))
            end_point = (int(kp_end.x), int(kp_end.y))

            # Draw the line
            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    # 3. Draw Keypoints (Joints)
    for keypoint in frame_metadata.keypoints:
        x, y = int(keypoint.x), int(keypoint.y)

        # Draw the circle
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    return frame
