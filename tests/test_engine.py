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
            {
                "torch_dtype": "float32",
                "local_files_only": True,
                "low_cpu_mem_usage": True,
            },
        )
    }
    assert loaded.device == "cpu"
    assert loaded.progress_disabled is True


def test_load_z_image_without_stable_diffusion_vae_helpers(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "models" / "Z-Image-Turbo"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline"}', encoding="utf-8"
    )
    loaded = FakeLoadedPipeline()
    calls = {}

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["directory"] = (path, kwargs)
            return loaded

    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        SimpleNamespace(
            DiffusionPipeline=FakeDiffusionPipeline,
            StableDiffusionXLPipeline=object,
        ),
    )
    engine = GenerationEngine.__new__(GenerationEngine)
    engine._pipeline = None
    engine._model_path = None
    engine._lora_path = None
    engine._ip_adapter_path = None
    engine.device = SimpleNamespace(cuda_available=True)
    engine._torch = SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )

    assert engine._load_pipeline(model_path) is loaded
    assert calls == {
        "directory": (
            str(model_path),
            {
                "torch_dtype": "bfloat16",
                "local_files_only": True,
                "low_cpu_mem_usage": True,
            },
        )
    }
    assert loaded.device == "cuda"


def test_keep_z_image_native_scheduler() -> None:
    ZImagePipeline = type("ZImagePipeline", (), {})
    pipeline = ZImagePipeline()
    original_scheduler = object()
    pipeline.scheduler = original_scheduler

    GenerationEngine._set_scheduler(pipeline, "Euler a")

    assert pipeline.scheduler is original_scheduler


def test_reject_ip_adapter_when_pipeline_does_not_support_it() -> None:
    engine = GenerationEngine.__new__(GenerationEngine)
    engine._ip_adapter_path = None
    ZImagePipeline = type("ZImagePipeline", (), {})

    try:
        engine._set_ip_adapter(
            ZImagePipeline(), Path("ip_adapters/sdxl/ip-adapter.safetensors"), 0.6
        )
    except RuntimeError as exc:
        assert str(exc) == "ZImagePipeline 不支持 IP-Adapter"
    else:
        raise AssertionError("unsupported IP-Adapter should fail with a clear error")
