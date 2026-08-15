from typing import List, Dict, Tuple
import cv2
import numpy as np
from .config import USE_YOLO

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'motorbike'}
_yolo_model = None

def calculate_iou(box_a: Dict, box_b: Dict) -> float:
    x_left = max(box_a['x1'], box_b['x1'])
    y_top = max(box_a['y1'], box_b['y1'])
    x_right = min(box_a['x2'], box_b['x2'])
    y_bottom = min(box_a['y2'], box_b['y2'])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(1, (box_a['x2'] - box_a['x1']) * (box_a['y2'] - box_a['y1']))
    area_b = max(1, (box_b['x2'] - box_b['x1']) * (box_b['y2'] - box_b['y1']))
    return inter / float(area_a + area_b - inter)

def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO('yolov8n.pt')
        return _yolo_model
    except Exception:
        return None
