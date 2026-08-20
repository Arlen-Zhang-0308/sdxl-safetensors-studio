from __future__ import annotations

import threading
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.engine import GenerationEngine
from app.schemas import GenerateRequest
from app.storage import Storage
from app.tasks import TaskStore

ROOT = Path(__file__).resolve().parents[1]
storage = Storage(ROOT)
engine = GenerationEngine()
tasks = TaskStore()
app = FastAPI(title="SDXL Safetensors Studio", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/history", StaticFiles(directory=storage.history_dir), name="history")
app.mount("/inputs", StaticFiles(directory=storage.inputs_dir), name="inputs")

SAMPLERS = [
    "Euler a",
    "Euler",
    "Euler Karras",
    "Heun",
    "Heun Karras",
    "LMS",
    "LMS Karras",
    "DDIM",
    "PNDM",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ 2S",
    "DPM++ 2S Karras",
    "DEIS",
    "UniPC",
]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {
        "device": engine.device.__dict__,
        "samplers": SAMPLERS,
        "defaults": {
            "width": 832,
            "height": 1216,
            "cfg": 4.0,
            "steps": 30,
            "sampler": "Euler a",
            "clip_skip": 2,
            "seed": -1,
            "batch_size": 1,
            "lora_scale": 1.0,
            "ip_adapter_scale": 0.6,
        },
    }


@app.post("/api/weights/refresh")
def refresh_weights() -> dict[str, list[str]]:
    return storage.scan_weights()


@app.get("/api/history")
def history() -> list[dict]:
    return storage.list_history()


@app.delete("/api/history/{history_id}", status_code=204)
def delete_history(history_id: str) -> None:
    try:
        deleted = storage.delete_history(history_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="历史记录不存在")


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="仅支持 PNG、JPEG 或 WebP 图片")
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="参考图不能超过 20MB")
    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="无法识别该图片文件") from exc
    filename = f"{uuid.uuid4().hex}.png"
    image.save(storage.inputs_dir / filename, format="PNG", optimize=True)
    return {"filename": filename, "image_url": f"/inputs/{filename}"}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="生成任务不存在或服务已重启")
    return task


def run_generation(
    task_id: str,
    request: GenerateRequest,
    model_path: Path,
    lora_path: Path | None,
    ip_adapter_path: Path | None,
    init_image_path: Path | None,
    mask_image_path: Path | None,
    ip_adapter_image_path: Path | None,
) -> None:
    generated: list[dict] = []

    def save_image(image, seed: int, index: int) -> None:
        stem = uuid.uuid4().hex
        image.save(storage.history_dir / f"{stem}.png", format="PNG")
        parameters = request.model_dump()
        parameters["seed"] = seed
        storage.save_metadata(
            stem,
            {**parameters, "device": engine.device.label, "batch_index": index},
        )
        generated.append({"id": stem, "image_url": f"/history/{stem}.png", "seed": seed})

    def update_progress(completed: int, total: int, current_batch: int) -> None:
        percent = min(99, round(completed / max(total, 1) * 100))
        tasks.update(
            task_id,
            status="running",
            progress=percent,
            current_step=completed,
            total_steps=total,
            current_batch=current_batch,
            message=f"正在推理第 {current_batch}/{request.batch_size} 张图片",
        )

    tasks.update(task_id, status="running", message="正在加载模型并准备推理")
    try:
        init_image = None
        if init_image_path is not None:
            with Image.open(init_image_path) as source:
                init_image = source.convert("RGB").resize(
                    (request.width, request.height), Image.Resampling.LANCZOS
                )
        mask_image = None
        if mask_image_path is not None:
            with Image.open(mask_image_path) as source:
                mask_image = source.convert("L").resize(
                    (request.width, request.height), Image.Resampling.NEAREST
                )
        ip_adapter_image = None
        if ip_adapter_image_path is not None:
            with Image.open(ip_adapter_image_path) as source:
                ip_adapter_image = source.convert("RGB")
        engine.generate(
            request,
            model_path,
            lora_path,
            ip_adapter_path,
            save_image,
            update_progress,
            init_image=init_image,
            mask_image=mask_image,
            ip_adapter_image=ip_adapter_image,
        )
        tasks.update(
            task_id,
            status="completed",
            progress=100,
            current_step=(
                max(1, round(request.steps * request.strength)) * request.batch_size
                if request.mode == "img2img"
                else request.steps * request.batch_size
            ),
            images=generated,
            message="生成完成",
        )
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "CUDA 显存不足。请降低宽高或批量数后重试。"
        elif "local_files_only" in message.lower() or "config" in message.lower():
            message = (
                "模型配置不完整，无法离线加载。请换用完整的 SDXL checkpoint，"
                "或检查 README 的模型兼容性说明。"
            )
        tasks.update(task_id, status="failed", error=message, message="生成失败")
    except Exception as exc:
        tasks.update(
            task_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            message="生成失败",
        )


@app.post("/api/generate", status_code=202)
def generate(request: GenerateRequest) -> dict:
    if request.sampler not in SAMPLERS:
        raise HTTPException(status_code=422, detail="不支持的采样器")
    try:
        model_path = storage.model_path(request.model)
        lora_path = storage.lora_path(request.lora) if request.lora else None
        ip_adapter_path = (
            storage.ip_adapter_path(request.ip_adapter) if request.ip_adapter else None
        )
        init_image_path = (
            storage.input_image_path(request.init_image) if request.mode == "img2img" else None
        )
        mask_image_path = (
            storage.input_image_path(request.mask_image) if request.mask_image else None
        )
        ip_adapter_image_path = (
            storage.input_image_path(request.ip_adapter_image)
            if request.ip_adapter_image
            else None
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    total_steps = (
        max(1, round(request.steps * request.strength)) * request.batch_size
        if request.mode == "img2img"
        else request.steps * request.batch_size
    )
    task_id = tasks.create(total_steps)
    threading.Thread(
        target=run_generation,
        args=(
            task_id,
            request,
            model_path,
            lora_path,
            ip_adapter_path,
            init_image_path,
            mask_image_path,
            ip_adapter_image_path,
        ),
        daemon=True,
    ).start()
    return {"task_id": task_id}
