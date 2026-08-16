import os
import cv2
import subprocess
import imageio_ffmpeg
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import CORS_ORIGINS, UPLOAD_DIR
from .database import (
    seed_slots, get_slots, upsert_slots,
    save_detection, latest_detections, using_memory
)
from .detection import detect_vehicles, update_slot_status, draw_results


app = FastAPI(title="Smart Car Parking Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == "*" else CORS_ORIGINS.split(","),
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
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    seed_slots()


@app.get("/")
def root():
    return {
        "message": "Smart Car Parking Detection API is running",
        "database": "memory" if using_memory() else "mongodb"
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/slots")
def slots():
    return {"slots": get_slots()}


@app.post("/api/slots")
def replace_slots(slots: List[Slot]):
    return {"slots": upsert_slots([s.model_dump() for s in slots])}


@app.get("/api/detections")
def detections():
    return {"detections": latest_detections(30)}


# IMAGE
@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image")

    name = f"image_{file.filename}"
    path = os.path.join(UPLOAD_DIR, name)
    result = os.path.join(UPLOAD_DIR, f"result_{name}.jpg")

    with open(path, "wb") as f:
        f.write(await file.read())

    frame = cv2.imread(path)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    frame = cv2.resize(frame, (640, 520))
    vehicles = detect_vehicles(frame)
    slots, summary = update_slot_status(get_slots(), vehicles)

    upsert_slots(slots)
    cv2.imwrite(result, draw_results(frame, slots, vehicles))

    record = save_detection({
        "source_type": "image",
        "filename": file.filename,
        "detections": vehicles,
        "slots": slots,
        "summary": summary
    })

    return {
        "record": record,
        "result_url": f"/api/result/{os.path.basename(result)}"
    }


# VIDEO
@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):

    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video")

    name = f"video_{file.filename}"
    path = os.path.join(UPLOAD_DIR, name)
    temp = os.path.join(UPLOAD_DIR, f"temp_{name}")
    result = os.path.join(UPLOAD_DIR, f"result_{name}")

    with open(path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise HTTPException(400, "Invalid video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    writer = cv2.VideoWriter(
        temp,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (512, 416)
    )

    slots = get_slots()
    vehicles = []
    summaries = []
    frame = 0

    while True:

        ok, image = cap.read()

        if not ok:
            break

        image = cv2.resize(image, (512, 416))

        # Detection every 30 frames for faster processing
        if frame % 30 == 0:
            vehicles = detect_vehicles(image)
            slots, summary = update_slot_status(slots, vehicles)
            summary["frame_index"] = frame
            summaries.append(summary)

        writer.write(
            draw_results(image, slots, vehicles)
        )

        frame += 1

    cap.release()
    writer.release()

    # Convert to browser-compatible MP4
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run([
        ffmpeg, "-y",
        "-i", temp,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        result
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp):
        os.remove(temp)

    upsert_slots(slots)

    occupancy = round(
        sum(x["occupancy_rate"] for x in summaries) /
        len(summaries), 2
    ) if summaries else 0

    record = save_detection({
        "source_type": "video",
        "filename": file.filename,
        "frames_analyzed": frame,
        "average_occupancy_rate": occupancy,
        "frame_summaries": summaries
    })

    return {
        "record": record,
        "slots": slots,
        "result_video_url":
            f"/api/video-result/{os.path.basename(result)}"
    }


@app.get("/api/result/{filename}")
def result(filename: str):

    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "Result not found")

    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/video-result/{filename}")
def video_result(filename: str):

    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")

    return FileResponse(path, media_type="video/mp4")
