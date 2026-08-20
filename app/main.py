from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import GenerationEngine
from app.schemas import GenerateRequest
from app.storage import Storage

ROOT = Path(__file__).resolve().parents[1]
storage = Storage(ROOT)
engine = GenerationEngine()
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


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> dict:
    if request.sampler not in SAMPLERS:
        raise HTTPException(status_code=422, detail="不支持的采样器")
    try:
        model_path = storage.model_path(request.model)
        lora_path = storage.lora_path(request.lora) if request.lora else None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    generated: list[dict] = []

    def save_image(image, seed: int, index: int) -> None:
        stem = uuid.uuid4().hex
        image.save(storage.history_dir / f"{stem}.png", format="PNG")
        parameters = request.model_dump()
        parameters["seed"] = seed
        storage.save_metadata(
            stem,
            {
                **parameters,
                "device": engine.device.label,
                "batch_index": index,
            },
        )
        generated.append({"id": stem, "image_url": f"/history/{stem}.png", "seed": seed})

    try:
        await asyncio.to_thread(engine.generate, request, model_path, lora_path, save_image)
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "CUDA 显存不足。请降低宽高或批量数后重试。"
        elif "local_files_only" in message.lower() or "config" in message.lower():
            message = (
                "模型配置不完整，无法离线加载。请换用完整的 SDXL checkpoint，"
                "或检查 README 的模型兼容性说明。"
            )
        raise HTTPException(status_code=500, detail=message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "生成失败。请确认所选文件是完整 SDXL checkpoint，LoRA 与 SDXL 兼容，"
                "并查看启动窗口中的详细错误。"
            ),
        ) from exc
    return {"images": generated}
