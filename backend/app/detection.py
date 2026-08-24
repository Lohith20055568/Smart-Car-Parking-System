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
        _yolo_model = YOLO('yolo11n.pt')
        return _yolo_model
    except Exception:
        return None

#References from https://docs.ultralytics.com 
#https://github.com/ultralytics/ultralytics
#https://docs.opencv.org/
#https://docs.ultralytics.com/usage/python/
#https://docs.ultralytics.com/tasks/detect/


def detect_vehicles_yolo(frame: np.ndarray) -> List[Dict]:
    model = _load_yolo()
    if model is None:
        return []
    results = model(frame, verbose=False, conf=0.20, imgsz=960)
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

#https://cocodataset.org/#home
#https://docs.ultralytics.com/tasks/detect
#https://docs.ultralytics.com/reference/results
#https://docs.ultralytics.com/tasks/detect

def detect_vehicles_opencv(frame: np.ndarray) -> List[Dict]:
    """Lightweight fallback detector for demos when YOLO is unavailable.
    It finds large vehicle-like objects using contours and works on sample/static parking images.
    """
    resized = cv2.resize(frame, (640, 520))

    # Dashboard diagrams often contain empty parking bays drawn as large white
    # rectangles.  Edge/contour detection alone treats those outlines as cars.
    # First look for solid, coloured vehicle bodies; this correctly handles the
    # bundled parking-diagram sample without turning every bay into a vehicle.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    colour_mask = cv2.inRange(hsv, (5, 70, 35), (179, 255, 245))
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colour_detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if 1500 <= area <= 30000 and 0.45 <= w / max(1, h) <= 2.5:
            colour_detections.append({
                'x1': int(x), 'y1': int(y), 'x2': int(x + w), 'y2': int(y + h),
                'class_name': 'vehicle_like_object', 'confidence': 0.60
            })
    if colour_detections:
        return colour_detections

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / max(1, h)
        # Thin rectangular outlines are parking bays, not vehicles.  Requiring
        # a reasonably filled contour prevents those false positives.
        fill_ratio = cv2.contourArea(c) / max(1, area)
        if 2800 <= area <= 70000 and 0.55 <= aspect <= 3.2 and fill_ratio >= 0.35:
            detections.append({
                'x1': int(x), 'y1': int(y), 'x2': int(x + w), 'y2': int(y + h),
                'class_name': 'vehicle_like_object', 'confidence': 0.55
            })
    return detections


#References from https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html'
#https://numpy.org/doc/stable/

def detect_vehicles(frame: np.ndarray) -> List[Dict]:
    if USE_YOLO:
        yolo_detections = detect_vehicles_yolo(frame)
        if yolo_detections:
            return yolo_detections
    return detect_vehicles_opencv(frame)

def update_slot_status(slots: List[Dict], detections: List[Dict], iou_threshold: float = 0.15) -> Tuple[List[Dict], Dict]:
    updated = []
    occupied = 0
    for slot in slots:
        best_iou = 0.0
        best_detection = None
        for det in detections:
            val = calculate_iou(slot, det)
            if val > best_iou:
                best_iou = val
                best_detection = det
        row = dict(slot)
        row['iou'] = round(best_iou, 3)
        row['status'] = 'occupied' if best_iou >= iou_threshold else 'vacant'
        row['matched_class'] = best_detection['class_name'] if best_detection else None
        if row['status'] == 'occupied':
            occupied += 1
        updated.append(row)
    summary = {
        'total_slots': len(slots),
        'occupied_slots': occupied,
        'vacant_slots': max(0, len(slots) - occupied),
        'vehicle_count': len(detections),
        'occupancy_rate': round((occupied / len(slots)) * 100, 2) if slots else 0
    }
    return updated, summary

#https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection
#https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

def draw_results(frame: np.ndarray, slots: List[Dict], detections: List[Dict]) -> np.ndarray:
    output = cv2.resize(frame.copy(), (640, 520))
    for det in detections:
        cv2.rectangle(output, (det['x1'], det['y1']), (det['x2'], det['y2']), (255, 255, 0), 2)
        cv2.putText(output, det['class_name'], (det['x1'], max(20, det['y1'] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for slot in slots:
        color = (0, 0, 255) if slot.get('status') == 'occupied' else (0, 180, 0)
        cv2.rectangle(output, (slot['x1'], slot['y1']), (slot['x2'], slot['y2']), color, 3)
        label = f"{slot['slot_id']} {slot.get('status', 'unknown')}"
        cv2.putText(output, label, (slot['x1'], slot['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output

#https://fastapi.tiangolo.com/
#https://pymongo.readthedocs.io/

