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


def detect_vehicles_yolo(frame: np.ndarray) -> List[Dict]:
    model = _load_yolo()
    if model is None:
        return []
    results = model(frame, verbose=False, conf=0.35)
    detections = []
    for r in results:
        names = r.names
        for b in r.boxes:
            cls_name = names[int(b.cls[0])]
            if cls_name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'class_name': cls_name,
                    'confidence': float(b.conf[0])
                })
    return detections

