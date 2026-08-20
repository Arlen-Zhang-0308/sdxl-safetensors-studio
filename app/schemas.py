from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    model: str
    lora: str | None = None
    lora_scale: float = Field(default=1.0, ge=-4.0, le=4.0)
    prompt: str = Field(min_length=1, max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    width: int = Field(default=832, ge=256, le=2048, multiple_of=8)
    height: int = Field(default=1216, ge=256, le=2048, multiple_of=8)
    cfg: float = Field(default=4.0, ge=0.0, le=30.0)
    steps: int = Field(default=30, ge=1, le=150)
    sampler: str = "Euler a"
    seed: int = Field(default=-1, ge=-1, le=2**63 - 1)
    batch_size: int = Field(default=1, ge=1, le=4)

    @field_validator("model", "lora")
    @classmethod
    def reject_paths(cls, value: str | None) -> str | None:
        if value is not None and ("/" in value or "\\" in value or value in {".", ".."}):
            raise ValueError("只允许选择已扫描到的文件名")
        return value
