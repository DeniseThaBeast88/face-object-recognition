"""Paths and default settings shared by the application, the evaluation scripts, and the tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCREENSHOTS = ROOT / "screenshots"

# Model files (all run on the CPU with OpenCV and ONNX Runtime; no PyTorch is needed)
YOLO_MODEL = MODELS / "yolov10n.onnx"                       # object detector, 80 COCO classes
CLASSIFIER_MODEL = MODELS / "mobilenetv3_large_imagenet.onnx"  # 1,000 ImageNet classes, used for "school bus"
IMAGENET_LABELS = MODELS / "imagenet_classes.txt"
YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"   # face detector with 5 landmarks
SFACE_MODEL = MODELS / "face_recognition_sface_2021dec.onnx"  # 128-value face embedding
HAAR_MODEL = MODELS / "haarcascade_frontalface_default.xml"   # classical baseline for comparison

# Stored face gallery (embeddings only, never face photos) and the background set used for calibration
GALLERY_FILE = DATA / "face_gallery.npz"
BACKGROUND_FILE = DATA / "background_embeddings.npy"

# Default thresholds
OBJECT_CONF = 0.35        # minimum detector confidence shown on screen
OBJECT_IOU = 0.50         # overlap allowed before non-maximum suppression removes a duplicate box
FACE_SCORE = 0.80         # minimum YuNet face score
SFACE_DEFAULT_COSINE = 0.363  # threshold published with the SFace model in the OpenCV model zoo
SCHOOL_BUS_PROB = 0.30    # minimum classifier probability for relabeling a bus as a school bus
REFINE_LABELS = ("bus", "truck")  # detector classes checked by the refiner (school buses are sometimes boxed as trucks)

# ImageNet class indices used by the fine-grained refiner
IMAGENET_SCHOOL_BUS = 779
