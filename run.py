from controllers.video_processing import process_video

# Predict with the model
frame_keypoints = process_video(
    "YTDowncom_YouTube_Dancing-hip-hop-mocap-animation_Media_VDIQ8Kn-PWw_001_1080p.mp4"
)

print(frame_keypoints)
