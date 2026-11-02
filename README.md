# FaceSight AI

**Real-time face detection and visual analytics — without identity recognition.**

FaceSight AI is a portfolio project for detecting faces in uploaded images, recorded videos, and live webcam frames. It uses OpenCV's pretrained **YuNet** detector and a FastAPI backend.

## Current capabilities

- Detect multiple faces in an image.
- Return bounding boxes and confidence scores.
- Count detected faces.
- Optionally blur detected faces in the annotated preview.
- Expose a frame endpoint that can be used by a browser webcam client.
- Health and model-status endpoints.

> **Privacy by design:** This project detects the presence and location of faces. It does not identify people, match identities, or build biometric profiles. Webcam frames are processed in memory and are not stored by the API.

## Architecture

```text
Browser (React / webcam / file upload)
                |
                v
          FastAPI backend
                |
                v
       OpenCV YuNet detector
                |
                v
   boxes + confidence + face count
                |
                v
     annotated preview / analytics
```

## Quick start (backend)

Requirements: Python 3.10+.

```bash
cd backend
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

On first startup, the backend downloads the small YuNet ONNX model into `backend/models/`. An internet connection is required only for this first download.

Open:
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Quick start (frontend)

In a second terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Start the backend first at http://127.0.0.1:8000.

Optional frontend environment variable:

```bash
# frontend/.env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The dashboard currently supports image uploads, browser webcam capture, and client-side video sampling (up to 20 frames per video). Video frames are sent individually to the existing frame-detection endpoint; the backend does not store the source video.

## API

### `GET /health`
Returns service and model status.

### `POST /api/v1/detect/image`
Multipart form fields:
- `file`: JPG, PNG, or WEBP image
- `blur_faces`: optional boolean (default `false`)

Returns face count, bounding boxes, confidence values, processing time, and a base64-encoded annotated image.

### `POST /api/v1/detect/frame`
Same request/response format as image detection. Intended for individual webcam frames sent by a browser client.

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `FACE_MODEL_PATH` | `models/face_detection_yunet_2023mar.onnx` | Model file path |
| `FACE_SCORE_THRESHOLD` | `0.75` | Minimum detector confidence |
| `FACE_NMS_THRESHOLD` | `0.3` | Non-maximum suppression threshold |
| `MAX_UPLOAD_MB` | `10` | Maximum image upload size |

## Docker

From the repository root:

```bash
docker build -t facesight-ai ./backend
docker run --rm -p 8000:8000 facesight-ai
```

## Roadmap

- [x] Repository and backend foundation
- [x] Pretrained YuNet image/frame detection API
- [ ] Recorded video processing endpoint
- [x] React dashboard, image upload, webcam controls, and video frame sampling
- [ ] Detection analytics and performance metrics
- [ ] Automated tests and GitHub Actions
- [ ] Production deployment

## Model

YuNet is a lightweight face detector distributed in ONNX format by the OpenCV Zoo project. See the [official YuNet model documentation](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

## Responsible use

Use only images and camera streams you have permission to process. Do not use this project for covert surveillance, identity inference, or decisions about individuals. Provide clear notice and obtain consent before using a live camera in a real environment.

## License

This project code is released under the MIT License. The pretrained model is distributed under the license specified by its upstream provider.
