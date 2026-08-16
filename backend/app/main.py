import os
import cv2
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import CORS_ORIGINS, UPLOAD_DIR
from .database import (
    seed_slots,
    get_slots,
    upsert_slots,
    save_detection,
    latest_detections,
    using_memory,
)
from .detection import detect_vehicles, update_slot_status, draw_results


app = FastAPI(
    title="Smart Car Parking Detection API",
    version="1.0.0",
)

origins = ["*"] if CORS_ORIGINS == "*" else [
    x.strip() for x in CORS_ORIGINS.split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Slot(BaseModel):
    slot_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    status: str = "unknown"


@app.on_event("startup")
def startup():
    seed_slots()


@app.get("/")
def root():
    return {
        "message": "Smart Car Parking Detection API is running",
        "database": "memory" if using_memory() else "mongodb",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "database": "memory" if using_memory() else "mongodb",
        "time": datetime.utcnow().isoformat(),
    }


@app.get("/api/slots")
def slots():
    return {"slots": get_slots()}


@app.post("/api/slots")
def replace_slots(slots: List[Slot]):
    return {"slots": upsert_slots([s.model_dump() for s in slots])}


@app.get("/api/detections")
def detections():
    return {"detections": latest_detections(30)}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    name = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    input_path = os.path.join(UPLOAD_DIR, f"{name}_{file.filename}")
    output_path = os.path.join(UPLOAD_DIR, f"{name}_result.jpg")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    frame = cv2.imread(input_path)
    if frame is None:
        raise HTTPException(400, "Could not read uploaded image")

    frame = cv2.resize(frame, (640, 520))

    vehicles = detect_vehicles(frame)
    slots, summary = update_slot_status(get_slots(), vehicles)
    slots = upsert_slots(slots)

    result = draw_results(frame, slots, vehicles)
    cv2.imwrite(output_path, result)

    record = save_detection({
        "source_type": "image",
        "filename": file.filename,
        "detections": vehicles,
        "slots": slots,
        "summary": summary,
        "result_image": os.path.basename(output_path),
    })

    return {
        "record": record,
        "result_url": f"/api/result/{os.path.basename(output_path)}",
    }


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file")

    name = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    input_path = os.path.join(UPLOAD_DIR, f"{name}_{file.filename}")
    output_path = os.path.join(UPLOAD_DIR, f"{name}_result.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not read uploaded video")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (640, 520),
    )

    summaries = []
    slots = get_slots()
    vehicles = []

    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 520))

        if frame_count % 15 == 0:
            vehicles = detect_vehicles(frame)
            slots, summary = update_slot_status(slots, vehicles)
            summary["frame_index"] = frame_count
            summaries.append(summary)

        writer.write(draw_results(frame, slots, vehicles))
        frame_count += 1

    cap.release()
    writer.release()

    slots = upsert_slots(slots)

    average_occupancy = round(
        sum(x["occupancy_rate"] for x in summaries) / len(summaries), 2
    ) if summaries else 0

    record = save_detection({
        "source_type": "video",
        "filename": file.filename,
        "frames_analyzed": frame_count,
        "average_occupancy_rate": average_occupancy,
        "frame_summaries": summaries,
        "result_video": os.path.basename(output_path),
    })

    return {
        "record": record,
        "result_video_url": f"/api/video-result/{os.path.basename(output_path)}",
    }


@app.get("/api/result/{filename}")
def result_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "Result file not found")

    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/video-result/{filename}")
def video_result(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "Result video not found")

    return FileResponse(path, media_type="video/mp4")
