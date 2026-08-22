from __future__ import annotations

import gc
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.schemas import GenerateRequest
from app.prompt_encoding import build_sdxl_prompt_kwargs, suppress_tokenizer_length_log


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
        self._ip_adapter_path: Path | None = None
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
        self._ip_adapter_path = None
        gc.collect()
        if self.device.cuda_available:
            self._import_torch().cuda.empty_cache()

    def _load_pipeline(self, model_path: Path) -> Any:
        if self._pipeline is not None and self._model_path == model_path:
            return self._pipeline

        from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

        self._unload()
        torch = self._import_torch()
        dtype = torch.float16 if self.device.cuda_available else torch.float32
        if model_path.is_dir():
            pipeline = DiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
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
            DDIMScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LMSDiscreteScheduler,
            PNDMScheduler,
            UniPCMultistepScheduler,
        )

        schedulers = {
            "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
            "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
            "Euler Karras": lambda config: EulerDiscreteScheduler.from_config(
                config, use_karras_sigmas=True
            ),
            "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
            "Heun Karras": lambda config: HeunDiscreteScheduler.from_config(
                config, use_karras_sigmas=True
            ),
            "LMS": lambda config: LMSDiscreteScheduler.from_config(config),
            "LMS Karras": lambda config: LMSDiscreteScheduler.from_config(
                config, use_karras_sigmas=True
            ),
            "DDIM": lambda config: DDIMScheduler.from_config(config),
            "PNDM": lambda config: PNDMScheduler.from_config(config),
            "DPM2": lambda config: KDPM2DiscreteScheduler.from_config(config),
            "DPM2 Karras": lambda config: KDPM2DiscreteScheduler.from_config(
                config, use_karras_sigmas=True
            ),
            "DPM2 a": lambda config: KDPM2AncestralDiscreteScheduler.from_config(config),
            "DPM2 a Karras": lambda config: KDPM2AncestralDiscreteScheduler.from_config(
                config, use_karras_sigmas=True
            ),
            "DPM++ 2M": lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="dpmsolver++"
            ),
            "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            ),
            "DPM++ 2M SDE": lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="sde-dpmsolver++"
            ),
            "DPM++ 2M SDE Karras": lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
            ),
            "DPM++ 2S": lambda config: DPMSolverSinglestepScheduler.from_config(
                config, algorithm_type="dpmsolver++"
            ),
            "DPM++ 2S Karras": lambda config: DPMSolverSinglestepScheduler.from_config(
                config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            ),
            "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
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

    def _set_ip_adapter(self, pipeline: Any, adapter_path: Path | None, scale: float) -> None:
        if self._ip_adapter_path is not None and self._ip_adapter_path != adapter_path:
            pipeline.unload_ip_adapter()
            self._ip_adapter_path = None
        if adapter_path is not None and self._ip_adapter_path != adapter_path:
            pipeline.load_ip_adapter(
                str(adapter_path.parent),
                subfolder="",
                weight_name=adapter_path.name,
                image_encoder_folder="image_encoder",
                local_files_only=True,
            )
            self._ip_adapter_path = adapter_path
        if adapter_path is not None:
            pipeline.set_ip_adapter_scale(scale)

    def generate(
        self,
        request: GenerateRequest,
        model_path: Path,
        lora_path: Path | None,
        ip_adapter_path: Path | None,
        on_image: Callable[[Any, int, int], None],
        on_progress: Callable[[int, int, int], None] | None = None,
        init_image: Any | None = None,
        mask_image: Any | None = None,
        ip_adapter_image: Any | None = None,
    ) -> list[dict[str, int]]:
        with self._lock:
            pipeline = self._load_pipeline(model_path)
            self._set_lora(pipeline, lora_path, request.lora_scale)
            self._set_ip_adapter(pipeline, ip_adapter_path, request.ip_adapter_scale)
            is_sdxl = getattr(pipeline, "text_encoder_2", None) is not None
            if mask_image is not None:
                from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

                pipeline_class = (
                    StableDiffusionXLInpaintPipeline if is_sdxl else StableDiffusionInpaintPipeline
                )
                pipeline = pipeline_class(**pipeline.components)
                pipeline.set_progress_bar_config(disable=True)
            elif init_image is not None:
                from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

                pipeline_class = (
                    StableDiffusionXLImg2ImgPipeline if is_sdxl else StableDiffusionImg2ImgPipeline
                )
                pipeline = pipeline_class(**pipeline.components)
                pipeline.set_progress_bar_config(disable=True)
            self._set_scheduler(pipeline, request.sampler)
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
                if mask_image is not None:
                    pipeline_kwargs["mask_image"] = mask_image
                if ip_adapter_image is not None:
                    pipeline_kwargs["ip_adapter_image"] = ip_adapter_image
                prompt_kwargs = build_sdxl_prompt_kwargs(
                    pipeline, request.prompt, request.negative_prompt, request.clip_skip
                )
                with torch.inference_mode():
                    with suppress_tokenizer_length_log():
                        image = pipeline(
                            width=request.width,
                            height=request.height,
                            guidance_scale=request.cfg,
                            num_inference_steps=request.steps,
                            generator=generator,
                            callback_on_step_end=step_callback,
                            callback_on_step_end_tensor_inputs=["latents"],
                            **prompt_kwargs,
                            **pipeline_kwargs,
                        ).images[0]
                on_image(image, seed, index)
                seeds.append({"seed": seed, "index": index})
            return seeds
