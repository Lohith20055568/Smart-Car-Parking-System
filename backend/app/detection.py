from typing import List, Dict, Tuple
import cv2
import numpy as np
from .config import USE_YOLO

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'motorbike'}
_yolo_model = None
