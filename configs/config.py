INPUT_IMG_RESIZE = (640, 640)

MODEL_NAME = "yolo26l-pose.pt"  # "yolo26x-pose.pt"
CONF = 0.25

# For reference on these keypoints, please visit https://docs.ultralytics.com/tasks/pose/
# Also the document did point them out.
KEYPOINTS_MAPPING = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}
