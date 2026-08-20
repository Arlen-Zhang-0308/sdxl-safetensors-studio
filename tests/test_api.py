from io import BytesIO
from pathlib import Path
import time

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.engine import DeviceInfo
from app.storage import Storage


class FakeEngine:
    device = DeviceInfo(kind="cpu", label="CPU test", cuda_available=False, dtype="float32")

    last_init_image = None
    last_mask_image = None
    last_ip_adapter_image = None
    last_ip_adapter_path = None

    def generate(
        self,
        request,
        model_path,
        lora_path,
        ip_adapter_path,
        on_image,
        on_progress=None,
        init_image=None,
        mask_image=None,
        ip_adapter_image=None,
    ):
        self.last_init_image = init_image
        self.last_mask_image = mask_image
        self.last_ip_adapter_image = ip_adapter_image
        self.last_ip_adapter_path = ip_adapter_path
        if on_progress is not None:
            on_progress(request.steps, request.steps * request.batch_size, 1)
        on_image(Image.new("RGB", (32, 32), "#e6a23c"), 42, 0)
        return [{"seed": 42, "index": 0}]


class FailingEngine(FakeEngine):
    def generate(self, *args, **kwargs):
        raise TypeError("missing local adapter argument")


def test_status_and_refresh(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    monkeypatch.setattr(main, "storage", storage)
    client = TestClient(main.app)

    assert client.get("/api/status").json()["defaults"]["width"] == 832
    assert client.get("/api/status").json()["defaults"]["clip_skip"] == 2
    assert len(client.get("/api/status").json()["samplers"]) == 21
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
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    task = None
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert task is not None
    assert task["status"] == "completed"
    assert task["progress"] == 100
    assert task["current_step"] == 30
    assert task["images"][0]["seed"] == 42
    history = storage.list_history()
    assert history[0]["prompt"] == "amber robot"
    assert history[0]["negative_prompt"] == "blur"
    assert history[0]["seed"] == 42
    assert history[0]["width"] == 832
    assert history[0]["clip_skip"] == 2


def test_generation_returns_unexpected_error_details(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    monkeypatch.setattr(main, "storage", storage)
    monkeypatch.setattr(main, "engine", FailingEngine())
    client = TestClient(main.app)

    response = client.post(
        "/api/generate",
        json={"model": "sdxl.safetensors", "prompt": "test"},
    )
    task_id = response.json()["task_id"]
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] == "failed":
            break
        time.sleep(0.01)

    assert task["error"] == "TypeError: missing local adapter argument"


def test_reject_bad_dimensions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "storage", Storage(tmp_path))
    response = TestClient(main.app).post(
        "/api/generate", json={"model": "x.safetensors", "prompt": "x", "width": 831}
    )
    assert response.status_code == 422


def test_generate_with_ip_adapter(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    adapter = storage.ip_adapters_dir / "sdxl-plus"
    (adapter / "image_encoder").mkdir(parents=True)
    (adapter / "ip-adapter-plus_sdxl_vit-h.safetensors").touch()
    fake_engine = FakeEngine()
    monkeypatch.setattr(main, "storage", storage)
    monkeypatch.setattr(main, "engine", fake_engine)
    client = TestClient(main.app)

    source = BytesIO()
    Image.new("RGB", (64, 64), "#775533").save(source, format="PNG")
    image_name = client.post(
        "/api/images/upload", files={"file": ("style.png", source.getvalue(), "image/png")}
    ).json()["filename"]
    response = client.post(
        "/api/generate",
        json={
            "model": "sdxl.safetensors",
            "prompt": "portrait",
            "ip_adapter": "sdxl-plus",
            "ip_adapter_image": image_name,
            "ip_adapter_scale": 0.75,
        },
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert task["status"] == "completed"
    assert fake_engine.last_ip_adapter_path.name == "ip-adapter-plus_sdxl_vit-h.safetensors"
    assert fake_engine.last_ip_adapter_image.size == (64, 64)
    history = storage.list_history()[0]
    assert history["ip_adapter_scale"] == 0.75
    assert history["ip_adapter_image_url"] == f"/inputs/{image_name}"


def test_ip_adapter_requires_weight_and_image_pair(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "storage", Storage(tmp_path))
    response = TestClient(main.app).post(
        "/api/generate",
        json={
            "model": "x.safetensors",
            "prompt": "x",
            "ip_adapter": "sdxl-plus",
        },
    )
    assert response.status_code == 422


def test_txt2img_rejects_mask(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "storage", Storage(tmp_path))
    response = TestClient(main.app).post(
        "/api/generate",
        json={
            "model": "x.safetensors",
            "prompt": "x",
            "mask_image": "mask.png",
        },
    )
    assert response.status_code == 422


def test_reject_bad_clip_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "storage", Storage(tmp_path))
    response = TestClient(main.app).post(
        "/api/generate", json={"model": "x.safetensors", "prompt": "x", "clip_skip": 13}
    )
    assert response.status_code == 422


def test_unknown_task_returns_not_found() -> None:
    response = TestClient(main.app).get("/api/tasks/missing")
    assert response.status_code == 404


def test_upload_and_generate_img2img(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    fake_engine = FakeEngine()
    monkeypatch.setattr(main, "storage", storage)
    monkeypatch.setattr(main, "engine", fake_engine)
    client = TestClient(main.app)

    source = BytesIO()
    Image.new("RGB", (48, 64), "#335577").save(source, format="JPEG")
    uploaded = client.post(
        "/api/images/upload", files={"file": ("source.jpg", source.getvalue(), "image/jpeg")}
    )
    assert uploaded.status_code == 200
    filename = uploaded.json()["filename"]
    assert (storage.inputs_dir / filename).is_file()

    response = client.post(
        "/api/generate",
        json={
            "mode": "img2img",
            "model": "sdxl.safetensors",
            "prompt": "watercolor city",
            "init_image": filename,
            "strength": 0.6,
        },
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    task = None
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert task is not None and task["status"] == "completed"
    assert fake_engine.last_init_image is not None
    assert fake_engine.last_init_image.size == (832, 1216)
    history = storage.list_history()[0]
    assert history["mode"] == "img2img"
    assert history["strength"] == 0.6
    assert history["init_image_url"] == f"/inputs/{filename}"


def test_generate_img2img_with_mask(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "sdxl.safetensors").touch()
    fake_engine = FakeEngine()
    monkeypatch.setattr(main, "storage", storage)
    monkeypatch.setattr(main, "engine", fake_engine)
    client = TestClient(main.app)

    source = BytesIO()
    Image.new("RGB", (48, 64), "#335577").save(source, format="PNG")
    mask = BytesIO()
    Image.new("L", (48, 64), 255).save(mask, format="PNG")
    init_name = client.post(
        "/api/images/upload", files={"file": ("source.png", source.getvalue(), "image/png")}
    ).json()["filename"]
    mask_name = client.post(
        "/api/images/upload", files={"file": ("mask.png", mask.getvalue(), "image/png")}
    ).json()["filename"]

    response = client.post(
        "/api/generate",
        json={
            "mode": "img2img",
            "model": "sdxl.safetensors",
            "prompt": "replace the masked area",
            "init_image": init_name,
            "mask_image": mask_name,
        },
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert task["status"] == "completed"
    assert fake_engine.last_mask_image is not None
    assert fake_engine.last_mask_image.mode == "L"
    assert fake_engine.last_mask_image.size == (832, 1216)
    history = storage.list_history()[0]
    assert history["mask_image"] == mask_name
    assert history["mask_image_url"] == f"/inputs/{mask_name}"


def test_img2img_requires_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "storage", Storage(tmp_path))
    response = TestClient(main.app).post(
        "/api/generate",
        json={"mode": "img2img", "model": "x.safetensors", "prompt": "x"},
    )
    assert response.status_code == 422


def test_upload_rejects_non_image() -> None:
    response = TestClient(main.app).post(
        "/api/images/upload", files={"file": ("note.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415


def test_delete_history_endpoint(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path)
    history_id = "b" * 32
    (storage.history_dir / f"{history_id}.png").touch()
    storage.save_metadata(history_id, {"prompt": "remove"})
    monkeypatch.setattr(main, "storage", storage)
    client = TestClient(main.app)

    assert client.delete(f"/api/history/{history_id}").status_code == 204
    assert client.delete(f"/api/history/{history_id}").status_code == 404
    assert client.delete("/api/history/not-valid").status_code == 422
