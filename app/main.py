from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

SAMPLERS = ["Euler a", "Euler", "DPM++ 2M Karras", "UniPC"]


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
            "seed": -1,
            "batch_size": 1,
            "lora_scale": 1.0,
        },
    }


@app.post("/api/weights/refresh")
def refresh_weights() -> dict[str, list[str]]:
    return storage.scan_weights()


@app.get("/api/history")
def history() -> list[dict]:
    return storage.list_history()


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="生成任务不存在或服务已重启")
    return task


def run_generation(
    task_id: str, request: GenerateRequest, model_path: Path, lora_path: Path | None
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
        engine.generate(request, model_path, lora_path, save_image, update_progress)
        tasks.update(
            task_id,
            status="completed",
            progress=100,
            current_step=request.steps * request.batch_size,
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
    except Exception:
        tasks.update(
            task_id,
            status="failed",
            error=(
                "生成失败。请确认所选文件是完整 SDXL checkpoint，LoRA 与 SDXL 兼容，"
                "并查看启动窗口中的详细错误。"
            ),
            message="生成失败",
        )


@app.post("/api/generate", status_code=202)
def generate(request: GenerateRequest) -> dict:
    if request.sampler not in SAMPLERS:
        raise HTTPException(status_code=422, detail="不支持的采样器")
    try:
        model_path = storage.model_path(request.model)
        lora_path = storage.lora_path(request.lora) if request.lora else None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    total_steps = request.steps * request.batch_size
    task_id = tasks.create(total_steps)
    threading.Thread(
        target=run_generation,
        args=(task_id, request, model_path, lora_path),
        daemon=True,
    ).start()
    return {"task_id": task_id}
