# Use pretrained YOLO model, default value for YOLO26 Large model which should be enough for this task, you can use bigger for higher accuracy or smaller for faster results.
MODEL_NAME = [
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
][2]

# For reference on these keypoints, please visit https://docs.ultralytics.com/tasks/pose/
# Also the document did point them out. it's needed for the YOLO output index and body part.
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

# For drawing lines between each body part
SKELETON_CONNECTIONS = [
    ("nose", "left_eye"),  # Line between nose and left eye...etc
    ("left_eye", "left_ear"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
