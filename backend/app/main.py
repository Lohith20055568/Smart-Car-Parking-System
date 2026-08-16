from datetime import datetime
import os
import subprocess
import imageio_ffmpeg
from typing import List
from bson import ObjectId
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import CORS_ORIGINS, UPLOAD_DIR
from .database import (
    seed_slots, get_slots, upsert_slots, save_detection,
    latest_detections, using_memory
)
from .detection import detect_vehicles, update_slot_status, draw_results


app = FastAPI(
    title="Smart Car Parking Detection API",
    version="1.0.0",
    json_encoders={ObjectId: str}
)

origins = ["*"] if CORS_ORIGINS == "*" else [
    x.strip() for x in CORS_ORIGINS.split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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
        "database_mode": "memory_fallback" if using_memory() else "mongodb"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "database": "memory_fallback" if using_memory() else "mongodb",
        "time": datetime.utcnow().isoformat()
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

    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    input_path = os.path.join(UPLOAD_DIR, f"{stamp}_{file.filename}")
    output_path = os.path.join(UPLOAD_DIR, f"{stamp}_result.jpg")

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
        "result_image": os.path.basename(output_path)
    })

    return {
        "record": record,
        "slots": slots,
        "result_url": f"/api/result/{os.path.basename(output_path)}"
    }


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file")

    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    input_path = os.path.join(UPLOAD_DIR, f"{stamp}_{file.filename}")
    preview_path = os.path.join(UPLOAD_DIR, f"{stamp}_preview.jpg")
    raw_path = os.path.join(UPLOAD_DIR, f"{stamp}_raw.mp4")
    video_path = os.path.join(UPLOAD_DIR, f"{stamp}_processed.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not read uploaded video")

    summaries = []
    latest_slots = None
    vehicles = []
    writer = None
    frame_index = 0
    best_output = None

    while frame_index < 90:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 520))

        if frame_index % 30 == 0:
            vehicles = detect_vehicles(frame)
            latest_slots, summary = update_slot_status(
                get_slots(), vehicles
            )

            summary["frame_index"] = frame_index
            summary["slots"] = latest_slots
            summaries.append(summary)

            best_output = draw_results(
                frame, latest_slots, vehicles
            )

        if latest_slots is not None:
            result = draw_results(
                frame, latest_slots, vehicles
            )

            if writer is None:
                writer = cv2.VideoWriter(
                    raw_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (640, 520)
                )

            writer.write(result)

        frame_index += 1

    cap.release()

    if writer:
        writer.release()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run([
            ffmpeg,
            "-y",
            "-i", raw_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            video_path
        ], stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL,
           check=False)

        if not os.path.exists(video_path):
            video_path = raw_path

    if latest_slots:
        upsert_slots(latest_slots)

    if best_output is not None:
        cv2.imwrite(preview_path, best_output)

    avg_occ = round(
        sum(x["occupancy_rate"] for x in summaries) /
        len(summaries), 2
    ) if summaries else 0

    record = save_detection({
        "source_type": "video",
        "filename": file.filename,
        "frames_analyzed": frame_index,
        "average_occupancy_rate": avg_occ,
        "frame_summaries": summaries,
        "result_image": os.path.basename(preview_path)
        if best_output is not None else None,
        "result_video": os.path.basename(video_path)
        if writer else None
    })

    return {
        "record": record,
        "slots": latest_slots,
        "result_url": f"/api/result/{os.path.basename(preview_path)}"
        if best_output is not None else None,
        "result_video_url": f"/api/video-result/{os.path.basename(video_path)}"
        if writer else None
    }


@app.get("/api/result/{filename}")
def result_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "Result file not found")

    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/video-result/{filename}")
def video_result_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "Result video not found")

    return FileResponse(path, media_type="video/mp4")
