"""FaceSight AI HTTP API."""
from __future__ import annotations

import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .detector import FaceDetector

detector = FaceDetector()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load at startup so model download errors are visible immediately.
    detector.load()
    yield


app = FastAPI(
    title="FaceSight AI",
    description="Privacy-conscious face detection API. Detection only; no identity recognition.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "FaceSight AI", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok" if detector.ready else "model_not_loaded", "model": "YuNet", "ready": detector.ready}


async def run_detection(file: UploadFile, blur_faces: bool) -> dict[str, object]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    started = time.perf_counter()
    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The file could not be decoded as an image.")

    try:
        result = detector.detect(image, blur_faces=blur_faces)
    except (cv2.error, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    ok, encoded = cv2.imencode(".jpg", result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the annotated preview.")

    height, width = image.shape[:2]
    return {
        "filename": file.filename,
        "face_count": len(result["faces"]),
        "faces": result["faces"],
        "image_width": width,
        "image_height": height,
        "processing_ms": round((time.perf_counter() - started) * 1000, 2),
        "blur_faces": blur_faces,
        "annotated_image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


@app.post("/api/v1/detect/image")
async def detect_image(
    file: Annotated[UploadFile, File(description="JPG, PNG, or WEBP image")],
    blur_faces: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Detect faces in an uploaded still image."""
    return await run_detection(file, blur_faces)


@app.post("/api/v1/detect/frame")
async def detect_frame(
    file: Annotated[UploadFile, File(description="A single webcam frame as JPG, PNG, or WEBP")],
    blur_faces: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Detect faces in one frame sent from a browser webcam client."""
    return await run_detection(file, blur_faces)
