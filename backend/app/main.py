from datetime import datetime
import json
import os
import subprocess
import time
from typing import List
from bson import ObjectId
import cv2

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import CORS_ORIGINS, UPLOAD_DIR
from .database import (
    get_slots,
    latest_detections,
    save_detection,
    seed_slots,
    upsert_slots,
    using_memory,
)

from .detection import (
    detect_vehicles,
    update_slot_status,
    draw_results,
)

app = FastAPI(
    title='Smart Car Parking Detection API',
    version='1.0.0',
    json_encoders={ObjectId: str}
)

origins = ['*'] if CORS_ORIGINS == '*' else [o.strip() for o in CORS_ORIGINS.split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

class Slot(BaseModel):
    slot_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    status: str = 'unknown'

@app.on_event('startup')
def startup():
    seed_slots()

@app.get('/')
def root():
    return {
        'message': 'Smart Car Parking Detection API is running',
        'database_mode': 'memory_fallback' if using_memory() else 'mongodb'
    }

@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'database': 'memory_fallback' if using_memory() else 'mongodb',
        'time': datetime.utcnow().isoformat()
    }

@app.get('/api/evaluation')
def evaluation():
    path = os.path.join(
        os.path.dirname(__file__),
        "../evaluation_results.json"
    )

    if not os.path.exists(path):
        return {
            "message": "Evaluation not completed yet"
        }

    with open(path) as f:
        return json.load(f)

@app.get('/api/slots')
def slots():
    return {'slots': get_slots()}

@app.post('/api/slots')
def replace_slots(slots: List[Slot]):
    return {'slots': upsert_slots([s.model_dump() for s in slots])}

@app.get('/api/detections')
def detections():
    return {'detections': latest_detections(30)}

@app.post('/api/detect/image')
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, 'Please upload an image file')

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    in_path = os.path.join(UPLOAD_DIR, f'{timestamp}_{file.filename}')
    out_path = os.path.join(UPLOAD_DIR, f'{timestamp}_result.jpg')

    with open(in_path, 'wb') as f:
        f.write(await file.read())

    frame = cv2.imread(in_path)
    if frame is None:
        raise HTTPException(400, 'Could not read uploaded image')

    frame = cv2.resize(frame, (640, 520))

    start = time.perf_counter()
    raw_detections = detect_vehicles(frame)
    latency = time.perf_counter() - start

    updated_slots, summary = update_slot_status(get_slots(), raw_detections)
    summary['latency_ms'] = round(latency * 1000, 2)
    summary['fps'] = round(1 / latency, 2) if latency else 0

    updated_slots = upsert_slots(updated_slots)

    output = draw_results(frame, updated_slots, raw_detections)
    cv2.imwrite(out_path, output)

    record = {
        'source_type': 'image',
        'filename': file.filename,
        'detections': raw_detections,
        'slots': updated_slots,
        'summary': summary,
        'result_image': os.path.basename(out_path)
    }

    saved = save_detection(record)

    return {
        'record': saved,
        'result_url': f'/api/result/{os.path.basename(out_path)}'
    }

@app.post('/api/detect/video')
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(400, 'Please upload a video file')

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    in_path = os.path.join(UPLOAD_DIR, f'{timestamp}_{file.filename}')
    out_path = os.path.join(UPLOAD_DIR, f'{timestamp}_preview.jpg')
    processed_video_path = os.path.join(UPLOAD_DIR, f'{timestamp}_processed.mp4')
    raw_video_path = os.path.join(UPLOAD_DIR, f'{timestamp}_raw.mp4')

    with open(in_path, 'wb') as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise HTTPException(400, 'Could not read uploaded video')

    summaries = []
    best_output = None
    latest_slots = None
    latest_detections_list = []
    video_writer = None
    frame_index = 0

    while frame_index < 60:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 520))

        if frame_index % 20 == 0:
            latest_detections_list = detect_vehicles(frame)
            updated_slots, summary = update_slot_status(
                get_slots(), latest_detections_list
            )

            latest_slots = updated_slots
            summary['frame_index'] = frame_index
            summary['slots'] = updated_slots
            summaries.append(summary)

            best_output = draw_results(
                frame, updated_slots, latest_detections_list
            )
            
        annotated_frame = best_output if (latest_slots is not None and best_output is not None) else frame

        if video_writer is None:
            video_writer = cv2.VideoWriter(
                raw_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (640, 520)
            )

        video_writer.write(annotated_frame)
        frame_index += 1

    cap.release()

    if video_writer is not None:
        video_writer.release()

        converted = subprocess.run(
            [
                'ffmpeg', '-y', '-i', raw_video_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                processed_video_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        if converted.returncode != 0 or not os.path.exists(processed_video_path):
            processed_video_path = raw_video_path

    if latest_slots is not None:
        upsert_slots(latest_slots)

    if best_output is not None:
        cv2.imwrite(out_path, best_output)

    avg_occ = round(
        sum(s['occupancy_rate'] for s in summaries) / len(summaries), 2
    ) if summaries else 0

    record = {
        'source_type': 'video',
        'filename': file.filename,
        'frames_analyzed': len(summaries),
        'average_occupancy_rate': avg_occ,
        'frame_summaries': summaries,
        'result_image': os.path.basename(out_path) if best_output is not None else None,
        'result_video': os.path.basename(processed_video_path) if video_writer is not None else None
    }

    saved = save_detection(record)

    return {
        'record': saved,
        'result_url': f'/api/result/{os.path.basename(out_path)}' if best_output is not None else None,
        'result_video_url': f'/api/video-result/{os.path.basename(processed_video_path)}' if video_writer is not None else None
    }

@app.get('/api/result/{filename}')
def result_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, 'Result file not found')

    return FileResponse(path, media_type='image/jpeg')

@app.get('/api/video-result/{filename}')
def video_result_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(404, 'Result file not found')

    return FileResponse(path, media_type='video/mp4')
