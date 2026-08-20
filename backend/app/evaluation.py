import json, os, time, cv2
from .detection import detect_vehicles, calculate_iou

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATASETS = {

    "PKLot":
    os.path.join(ROOT,"datasets/PKLot"),

    "CNRPark-EXT":
    os.path.join(ROOT,"datasets/CNRPark-EXT")

}
ANN = os.path.join(DATA, "annotations.json")
IMG = os.path.join(DATA, "images copy")


def evaluate():
    data = json.load(open(ANN))
    names = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i["file_name"] for i in data["images"]}

    times, ious = [], []

    for image_id, filename in images.items():
        frame = cv2.imread(os.path.join(IMG, filename))
        if frame is None:
            continue

        start = time.perf_counter()
        detections = detect_vehicles(frame)
        times.append(time.perf_counter() - start)

        spaces = [
            a for a in data["annotations"]
            if a["image_id"] == image_id and
            names.get(a["category_id"]) in ("space-empty", "space-occupied")
        ]

        for space in spaces:
            box = {
                "x1": space["bbox"][0],
                "y1": space["bbox"][1],
                "x2": space["bbox"][0] + space["bbox"][2],
                "y2": space["bbox"][1] + space["bbox"][3]
            }

            if detections:
                ious.append(max(calculate_iou(box, d) for d in detections))

    latency = sum(times) / len(times) if times else 0
    fps = 1 / latency if latency else 0
    mean_iou = sum(ious) / len(ious) if ious else 0

    print({
        "images_tested": len(times),
        "latency_ms": round(latency * 1000, 2),
        "FPS": round(fps, 2),
        "mean_IoU": round(mean_iou, 3)
    })


if __name__ == "__main__":
    evaluate()
