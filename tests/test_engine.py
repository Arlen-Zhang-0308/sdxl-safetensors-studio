from pathlib import Path
from types import SimpleNamespace

from app.engine import GenerationEngine


class FakePipeline:
    def __init__(self) -> None:
        self.load_args = None
        self.load_kwargs = None
        self.scale = None

    def load_ip_adapter(self, *args, **kwargs) -> None:
        self.load_args = args
        self.load_kwargs = kwargs

    def set_ip_adapter_scale(self, scale: float) -> None:
        self.scale = scale


class FakeLoadedPipeline:
    def __init__(self) -> None:
        self.device = None
        self.progress_disabled = None

    def to(self, device: str):
        self.device = device
        return self

    def set_progress_bar_config(self, disable: bool) -> None:
        self.progress_disabled = disable


def test_load_local_ip_adapter_package() -> None:
    engine = GenerationEngine.__new__(GenerationEngine)
    engine._ip_adapter_path = None
    pipeline = FakePipeline()
    adapter = Path("ip_adapters/sdxl-plus/ip-adapter-plus_sdxl_vit-h.safetensors")

    engine._set_ip_adapter(pipeline, adapter, 0.65)

    assert pipeline.load_args == (str(adapter.parent),)
    assert pipeline.load_kwargs == {
        "subfolder": "",
        "weight_name": adapter.name,
        "image_encoder_folder": "image_encoder",
        "local_files_only": True,
    }
    assert pipeline.scale == 0.65


def test_load_diffusers_model_directory_with_from_pretrained(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "models" / "directory-model"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")
    loaded = FakeLoadedPipeline()
    calls = {}

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["directory"] = (path, kwargs)
            return loaded

    class FakeSdxlPipeline:
        @classmethod
        def from_single_file(cls, path, **kwargs):
            calls["file"] = (path, kwargs)
            return loaded

    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        SimpleNamespace(
            DiffusionPipeline=FakeDiffusionPipeline,
            StableDiffusionXLPipeline=FakeSdxlPipeline,
        ),
    )
    engine = GenerationEngine.__new__(GenerationEngine)
    engine._pipeline = None
    engine._model_path = None
    engine._lora_path = None
    engine._ip_adapter_path = None
    engine.device = SimpleNamespace(cuda_available=False)
    engine._torch = SimpleNamespace(float16="float16", float32="float32")

    assert engine._load_pipeline(model_path) is loaded
    assert calls == {
        "directory": (
            str(model_path),
            {"torch_dtype": "float32", "local_files_only": True},
        )
    }
    assert loaded.device == "cpu"
    assert loaded.progress_disabled is True
