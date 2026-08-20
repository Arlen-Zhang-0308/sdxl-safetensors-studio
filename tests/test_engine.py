from pathlib import Path

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
