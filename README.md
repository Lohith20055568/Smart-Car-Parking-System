# Smart Car Parking Detection System Using OpenCV, YOLO-ready Logic and MongoDB

This is a complete implementation package for a smart parking project. It includes:

- FastAPI backend
- OpenCV image/video processing
- YOLO-ready detection module
- Parking slot IoU mapping
- MongoDB database storage
- React dashboard
- Render deployment files
- Docker Compose local setup

## Main Features

1. Upload parking image or video.
2. Detect vehicle-like objects using OpenCV fallback.
3. Optional YOLO inference can be enabled by installing `ultralytics` and setting `USE_YOLO=true`.
4. Map detections to parking slots using IoU.
5. Mark slots as occupied or vacant.
6. Store detection history in MongoDB.
7. Show dashboard analytics and recent detections.
8. Ready for Render deployment.

## Local Run Without Docker

### Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# or: cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open backend API:

```text
https://smart-car-parking-system-6snu.onrender.com/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

##Render Deployment files 


Frontend:

```text
https://smart-parking-frontend-7mey.onrender.com
```

Backend:

```text
https://smart-car-parking-backend.onrender.com
```

## MongoDB Setup

For local MongoDB, keep:

```env
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=smart_parking
```

For MongoDB Atlas, replace with your Atlas connection string:

```env
MONGO_URI=mongodb+srv://username:password@cluster-url/smart_parking?retryWrites=true&w=majority
```

## Render Deployment

1. Push this folder to GitHub.
2. Go to Render Dashboard.
3. Select **New Blueprint**.
4. Choose your GitHub repository.
5. Render reads `render.yaml` automatically.
6. Add environment variable `MONGO_URI` manually in backend service.
7. Add `VITE_API_URL` in frontend service after backend is deployed.

Example:

```text
VITE_API_URL=https://smart-parking-backend.onrender.com
```

## Enabling Real YOLO

By default this project uses an OpenCV fallback detector so it can run easily on normal laptops and Render free tier.

To enable YOLO locally:

1. Open `backend/requirements.txt`.
2. Uncomment:

```text
ultralytics==8.3.49
```

3. Set:

```env
USE_YOLO=true
```

4. Restart backend.

YOLO will download `yolov8n.pt` automatically during first run.

## Important API Endpoints

- `GET /api/health` checks backend and database status.
- `GET /api/slots` lists parking slots.
- `POST /api/slots` replaces parking slot coordinates.
- `POST /api/detect/image` uploads and processes image.
- `POST /api/detect/video` uploads and processes video.
- `GET /api/detections` shows detection history.

## Project Explanation for Report

The backend processes uploaded parking images or videos. OpenCV extracts frames and detects vehicle-like objects. The system compares each detected bounding box with predefined parking slot coordinates using Intersection over Union. If the IoU is above the threshold, the slot is marked occupied. Otherwise, it is marked vacant. Results are saved in MongoDB and displayed on the React dashboard.
