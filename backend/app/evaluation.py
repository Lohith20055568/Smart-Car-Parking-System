import json
import os
import time
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .detection import calculate_iou, detect_vehicles


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA = os.path.join(ROOT, "datasets/PKLot")
ANN = os.path.join(DATA, "annotations.json")
IMG = os.path.join(DATA, "images copy")


def inside(slot, v):
    cx = (v["x1"] + v["x2"]) / 2
    cy = (v["y1"] + v["y2"]) / 2

    return (
        slot["x1"] <= cx <= slot["x2"]
        and slot["y1"] <= cy <= slot["y2"]
    )


def evaluate():
    data = json.load(open(ANN))

    cats = {
        c["id"]: c["name"]
        for c in data["categories"]
    }

    imgs = {
        i["id"]: i["file_name"]
        for i in data["images"]
    }

    y, yp, ious, times = [], [], [], []

    for iid, name in imgs.items():
        frame = cv2.imread(os.path.join(IMG, name))

        if frame is None:
            continue

        s = time.perf_counter()
        det = detect_vehicles(frame)
        times.append(time.perf_counter() - s)

        for a in data["annotations"]:
            if a["image_id"] != iid:
                continue

            if cats.get(a["category_id"]) not in [
                "space-empty",
                "space-occupied"
            ]:
                continue

            slot = {
                "x1": a["bbox"][0],
                "y1": a["bbox"][1],
                "x2": a["bbox"][0] + a["bbox"][2],
                "y2": a["bbox"][1] + a["bbox"][3],
            }

            occupied = False
            best = 0

            for d in det:
                iou = calculate_iou(slot, d)

                vehicle_center_inside = inside(slot, d)

                slot_center_x = (
                    slot["x1"] + slot["x2"]
                ) / 2

                slot_center_y = (
                    slot["y1"] + slot["y2"]
                ) / 2

                slot_center_inside_vehicle = (
                    d["x1"] <= slot_center_x <= d["x2"]
                    and d["y1"] <= slot_center_y <= d["y2"]
                )

                occupied = (
                    occupied
                    or vehicle_center_inside
                    or slot_center_inside_vehicle
                    or iou >= 0.05
                )

                best = max(best, iou)

            y.append(
                cats[a["category_id"]] == "space-occupied"
            )

            yp.append(occupied)
            ious.append(best)

    cm = confusion_matrix(y, yp)
    tn, fp, fn, tp = cm.ravel()

    latency = sum(times) / len(times) if times else 0

    result = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": round(
            accuracy_score(y, yp) * 100, 2
        ),
        "Precision": round(
            precision_score(y, yp, zero_division=0) * 100, 2
        ),
        "Recall": round(
            recall_score(y, yp, zero_division=0) * 100, 2
        ),
        "F1": round(
            f1_score(y, yp, zero_division=0) * 100, 2
        ),
        "IoU": round(
            sum(ious) / len(ious), 3
        ) if ious else 0,
        "Latency_ms": round(
            latency * 1000, 2
        ),
        "FPS": round(
            1 / latency, 2
        ) if latency else 0,
        "Images_tested": len(times),
    }

    print(json.dumps(result, indent=2))

    with open(
        os.path.join(ROOT, "evaluation_results.json"),
        "w"
    ) as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    evaluate()
