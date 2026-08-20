from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GenerateRequest(BaseModel):
    mode: Literal["txt2img", "img2img"] = "txt2img"
    model: str
    lora: str | None = None
    lora_scale: float = Field(default=1.0, ge=-4.0, le=4.0)
    ip_adapter: str | None = None
    ip_adapter_image: str | None = None
    ip_adapter_scale: float = Field(default=0.6, ge=0.0, le=2.0)
    prompt: str = Field(min_length=1, max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    width: int = Field(default=832, ge=256, le=2048, multiple_of=8)
    height: int = Field(default=1216, ge=256, le=2048, multiple_of=8)
    cfg: float = Field(default=4.0, ge=0.0, le=30.0)
    steps: int = Field(default=30, ge=1, le=150)
    sampler: str = "Euler a"
    clip_skip: int = Field(default=2, ge=0, le=12)
    seed: int = Field(default=-1, ge=-1, le=2**63 - 1)
    batch_size: int = Field(default=1, ge=1, le=4)
    init_image: str | None = None
    mask_image: str | None = None
    strength: float = Field(default=0.65, gt=0.0, le=1.0)

    @field_validator(
        "model", "lora", "ip_adapter", "ip_adapter_image", "init_image", "mask_image"
    )
    @classmethod
    def reject_paths(cls, value: str | None) -> str | None:
        if value is not None and ("/" in value or "\\" in value or value in {".", ".."}):
            raise ValueError("只允许选择已扫描到的文件名")
        return value

    @model_validator(mode="after")
    def require_init_image(self) -> "GenerateRequest":
        if self.mode == "img2img" and not self.init_image:
            raise ValueError("图生图模式必须上传参考图")
        if self.mask_image and self.mode != "img2img":
            raise ValueError("蒙版只能用于图生图模式")
        if bool(self.ip_adapter) != bool(self.ip_adapter_image):
            raise ValueError("IP-Adapter 权重和参考图必须同时选择")
        return self
