import os

BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

FL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "fl_outputs")
METRIC_DIR    = os.path.join(FL_OUTPUT_DIR, "metrics")
PLOT_DIR      = os.path.join(FL_OUTPUT_DIR, "plots")

CHEST_CLASSES = [
    "Normal",
    "Pneumonia",
    "COVID-19 Pneumonia",
    "Tuberculosis",
    "Pleural Effusion",
    "Cardiomegaly",
    "Atelectasis",
]

INPUT_SIZE  = 224
NUM_CLIENTS = 3

UPLOAD_DIR         = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024

SECRET_KEY = "fl-inference-major-project"
