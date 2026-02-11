import torch


# Will detect only confidence scores >= this value will be kept.
CONF = 0.25

# When multiple bounding boxes overlap, NMS keeps the highest confidence box and removes boxes that overlap with it by more than this threshold.
IOU = 0.45

# Automatically selects GPU if available, otherwise uses cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enable half-precision (FP16) inference for faster computation.
# FP16 uses 16-bit floating point instead of 32-bit, reducing memory and increasing speed
# Only beneficial on GPU with tensor cores (modern NVIDIA GPUs)
# On CPU, half precision can be slower and less accurate
HALF = True if DEVICE == "cuda" else False

# Images are resized to this dimension before input to the ml model.
# Smaller sizes faster inference but lower accuracy.
# 320, 416, 480, 640
IMG_SIZE = 480

# Print prediction details to console.
VERBOSE = False

# Limits the number of objects detected to improve performance.
# Requirements are one person only.
MAX_DETECTION = 1

# Whether to perform class-agnostic Non-Maximum Suppression.
# Good for single-class detection or when different classes can overlap
# For single-class detection (person only), False is optimal
AGNOSTIC_NMS = False
