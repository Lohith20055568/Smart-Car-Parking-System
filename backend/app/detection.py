import os
from typing import List, Dict, Tuple
import cv2
import numpy as np
from .config import USE_YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike"}
_yolo_model = None


def calculate_iou(a: Dict, b: Dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))

    return inter / float(area_a + area_b - inter)


def _load_yolo():
    global _yolo_model

    if _yolo_model is not None:
        return _yolo_model

    try:
        from ultralytics import YOLO

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "yolo11n.pt"
        )

        _yolo_model = YOLO(model_path)
        print("YOLO11 loaded:", model_path)

        return _yolo_model

    except Exception as e:
        print("YOLO11 loading failed:", e)
        return None


def detect_vehicles_yolo(frame: np.ndarray) -> List[Dict]:
    model = _load_yolo()

    if model is None:
        return []

    detections = []

    for result in model(frame, verbose=False, conf=0.35):
        for box in result.boxes:
            name = result.names[int(box.cls[0])]

            if name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_name": name,
                    "confidence": float(box.conf[0])
                })

    return detections


def detect_vehicles_opencv(frame: np.ndarray) -> List[Dict]:
    frame = cv2.resize(frame, (640, 520))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        40, 120
    )

    contours, _ = cv2.findContours(
        cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8)
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ratio = w / max(1, h)
        fill = cv2.contourArea(c) / max(1, area)

        if (
            2800 <= area <= 70000
            and 0.55 <= ratio <= 3.2
            and fill >= 0.35
        ):
            detections.append({
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "class_name": "vehicle_like_object",
                "confidence": 0.55
            })

    return detections


def detect_vehicles(frame: np.ndarray) -> List[Dict]:
    if USE_YOLO:
        detections = detect_vehicles_yolo(frame)

        if detections:
            return detections

    return detect_vehicles_opencv(frame)


def update_slot_status(
    slots: List[Dict],
    detections: List[Dict],
    threshold: float = 0.15
) -> Tuple[List[Dict], Dict]:

    updated = []
    occupied = 0

    for slot in slots:
        best_iou = 0
        best = None

        for detection in detections:
            iou = calculate_iou(slot, detection)

            if iou > best_iou:
                best_iou = iou
                best = detection

        row = dict(slot)
        row["iou"] = round(best_iou, 3)
        row["status"] = (
            "occupied"
            if best_iou >= threshold
            else "vacant"
        )
        row["matched_class"] = (
            best["class_name"] if best else None
        )

        if row["status"] == "occupied":
            occupied += 1

        updated.append(row)

    total = len(slots)

    return updated, {
        "total_slots": total,
        "occupied_slots": occupied,
        "vacant_slots": max(0, total - occupied),
        "vehicle_count": len(detections),
        "occupancy_rate": (
            round(occupied / total * 100, 2)
            if total else 0
        )
    }


def draw_results(
    frame: np.ndarray,
    slots: List[Dict],
    detections: List[Dict]
) -> np.ndarray:

    output = cv2.resize(
        frame.copy(),
        (640, 520)
    )

    for d in detections:
        cv2.rectangle(
            output,
            (d["x1"], d["y1"]),
            (d["x2"], d["y2"]),
            (255, 255, 0),
            2
        )

        cv2.putText(
            output,
            d["class_name"],
            (d["x1"], max(20, d["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

    for slot in slots:
        occupied = slot.get("status") == "occupied"
        color = (0, 0, 255) if occupied else (0, 180, 0)

        cv2.rectangle(
            output,
            (slot["x1"], slot["y1"]),
            (slot["x2"], slot["y2"]),
            color,
            3
        )

        cv2.putText(
            output,
            f'{slot["slot_id"]} {slot.get("status", "unknown")}',
            (slot["x1"], slot["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    return output
