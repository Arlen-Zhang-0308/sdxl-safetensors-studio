from __future__ import annotations

import gc
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.schemas import GenerateRequest


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    label: str
    cuda_available: bool
    dtype: str


class GenerationEngine:
    def __init__(self) -> None:
        self._torch: Any = None
        self._pipeline: Any = None
        self._model_path: Path | None = None
        self._lora_path: Path | None = None
        self._lock = threading.Lock()
        self.device = self._detect_device()

    def _import_torch(self) -> Any:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _detect_device(self) -> DeviceInfo:
        try:
            torch = self._import_torch()
            if torch.cuda.is_available():
                return DeviceInfo(
                    kind="cuda",
                    label=f"CUDA · {torch.cuda.get_device_name(0)}",
                    cuda_available=True,
                    dtype="float16",
                )
        except (ImportError, RuntimeError):
            pass
        return DeviceInfo(kind="cpu", label="CPU fallback", cuda_available=False, dtype="float32")

    def _unload(self) -> None:
        self._pipeline = None
        self._model_path = None
        self._lora_path = None
        gc.collect()
        if self.device.cuda_available:
            self._import_torch().cuda.empty_cache()

    def _load_pipeline(self, model_path: Path) -> Any:
        if self._pipeline is not None and self._model_path == model_path:
            return self._pipeline

        from diffusers import StableDiffusionXLPipeline

        self._unload()
        torch = self._import_torch()
        dtype = torch.float16 if self.device.cuda_available else torch.float32
        pipeline = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        if self.device.cuda_available:
            pipeline = pipeline.to("cuda")
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
        else:
            pipeline = pipeline.to("cpu")
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        self._model_path = model_path
        return pipeline

    @staticmethod
    def _set_scheduler(pipeline: Any, sampler: str) -> None:
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            UniPCMultistepScheduler,
        )

        schedulers = {
            "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
            "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
            "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            ),
            "UniPC": lambda config: UniPCMultistepScheduler.from_config(config),
        }
        pipeline.scheduler = schedulers[sampler](pipeline.scheduler.config)

    def _set_lora(self, pipeline: Any, lora_path: Path | None, scale: float) -> None:
        if self._lora_path is not None:
            try:
                pipeline.unload_lora_weights()
            except (AttributeError, RuntimeError):
                pass
            self._lora_path = None
        if lora_path is not None:
            pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
            pipeline.set_adapters(["default_0"], adapter_weights=[scale])
            self._lora_path = lora_path

    def generate(
        self,
        request: GenerateRequest,
        model_path: Path,
        lora_path: Path | None,
        on_image: Callable[[Any, int, int], None],
        on_progress: Callable[[int, int, int], None] | None = None,
        init_image: Any | None = None,
    ) -> list[dict[str, int]]:
        with self._lock:
            pipeline = self._load_pipeline(model_path)
            if init_image is not None:
                from diffusers import StableDiffusionXLImg2ImgPipeline

                pipeline = StableDiffusionXLImg2ImgPipeline(**pipeline.components)
                pipeline.set_progress_bar_config(disable=True)
            self._set_scheduler(pipeline, request.sampler)
            self._set_lora(pipeline, lora_path, request.lora_scale)
            torch = self._import_torch()
            seeds: list[dict[str, int]] = []
            steps_per_image = (
                max(1, round(request.steps * request.strength))
                if init_image is not None
                else request.steps
            )
            for index in range(request.batch_size):
                seed = request.seed if request.seed >= 0 else random.SystemRandom().randint(0, 2**63 - 1)
                if request.seed >= 0:
                    seed += index
                generator = torch.Generator(device=self.device.kind).manual_seed(seed)

                def step_callback(
                    _pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]
                ) -> dict[str, Any]:
                    if on_progress is not None:
                        completed = index * steps_per_image + min(step + 1, steps_per_image)
                        on_progress(completed, steps_per_image * request.batch_size, index + 1)
                    return callback_kwargs

                pipeline_kwargs: dict[str, Any] = {}
                if init_image is not None:
                    pipeline_kwargs.update(image=init_image, strength=request.strength)
                with torch.inference_mode():
                    image = pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt or None,
                        width=request.width,
                        height=request.height,
                        guidance_scale=request.cfg,
                        num_inference_steps=request.steps,
                        generator=generator,
                        callback_on_step_end=step_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                        **pipeline_kwargs,
                    ).images[0]
                on_image(image, seed, index)
                seeds.append({"seed": seed, "index": index})
            return seeds
