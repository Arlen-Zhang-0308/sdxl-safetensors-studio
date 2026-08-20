from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.engine import DeviceInfo
from app.storage import Storage


class FakeEngine:
    device = DeviceInfo(kind="cpu", label="CPU test", cuda_available=False, dtype="float32")

    def generate(self, request, model_path, lora_path, on_image):
        on_image(Image.new("RGB", (32, 32), "#e6a23c"), 42, 0)
        return [{"seed": 42, "index": 0}]


def test_status_and_refresh(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    monkeypatch.setattr(main, "storage", storage)
    client = TestClient(main.app)

    assert client.get("/api/status").json()["defaults"]["width"] == 832
    assert client.post("/api/weights/refresh").json()["models"] == ["sdxl.safetensors"]


def test_generate_saves_complete_history(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    monkeypatch.setattr(main, "storage", storage)
    monkeypatch.setattr(main, "engine", FakeEngine())
    client = TestClient(main.app)

    response = client.post(
        "/api/generate",
        json={"model": "sdxl.safetensors", "prompt": "amber robot", "negative_prompt": "blur"},
    )
    assert response.status_code == 200
    history = storage.list_history()
    assert history[0]["prompt"] == "amber robot"
    assert history[0]["negative_prompt"] == "blur"
    assert history[0]["seed"] == 42
    assert history[0]["width"] == 832


def test_reject_bad_dimensions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "storage", Storage(tmp_path))
    response = TestClient(main.app).post(
        "/api/generate", json={"model": "x.safetensors", "prompt": "x", "width": 831}
    )
    assert response.status_code == 422
