from ultralytics import YOLO

class VehicleDetector:
    def __init__(self):
        # Automatically downloads the model if it's not already available
        self.model = YOLO("yolo11n.pt")

    def detect(self, frame):
        results = self.model(frame)
        return results
