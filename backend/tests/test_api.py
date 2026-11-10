"""API contract tests for FaceSight AI.

The detector is replaced with a deterministic stub so tests do not download
or load the YuNet model and remain fast in CI.
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture()
def client(monkeypatch):
    def fake_detect(image, blur_faces=False):
        height, width = image.shape[:2]
        annotated = image.copy()
        return {
            "faces": [
                {
                    "box": {"x": 5, "y": 6, "width": 20, "height": 22},
                    "confidence": 0.9876,
                }
            ],
            "annotated_image": annotated,
        }

    monkeypatch.setattr(main.detector, "load", lambda: None)
    monkeypatch.setattr(main.detector, "detect", fake_detect)
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture()
def png_bytes():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_health_reports_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "YuNet",
        "ready": True,
    }


@pytest.mark.parametrize("endpoint", ["image", "frame"])
def test_detection_endpoints_return_expected_contract(client, png_bytes, endpoint):
    response = client.post(
        f"/api/v1/detect/{endpoint}",
        files={"file": ("sample.png", png_bytes, "image/png")},
        data={"blur_faces": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["face_count"] == 1
    assert body["image_width"] == 64
    assert body["image_height"] == 48
    assert body["blur_faces"] is True
    assert body["faces"][0]["confidence"] == pytest.approx(0.9876)
    assert body["faces"][0]["box"] == {
        "x": 5, "y": 6, "width": 20, "height": 22
    }
    assert body["processing_ms"] >= 0
    assert body["annotated_image_base64"]


def test_rejects_unsupported_media_type(client):
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_rejects_corrupt_image(client):
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("broken.png", b"not-a-real-image", "image/png")},
    )
    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]


def test_rejects_empty_upload(client):
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_rejects_upload_over_configured_limit(client, png_bytes, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 16)
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("large.png", png_bytes, "image/png")},
    )
    assert response.status_code == 413
