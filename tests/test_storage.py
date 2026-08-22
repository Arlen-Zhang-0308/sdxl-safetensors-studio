from pathlib import Path

import pytest

from app.storage import Storage


def test_scan_safetensors_and_diffusers_model_directories(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "z.safetensors").touch()
    (storage.models_dir / "A.SAFETENSORS").touch()
    (storage.models_dir / "ignore.ckpt").touch()
    diffusers_model = storage.models_dir / "sd-directory"
    diffusers_model.mkdir()
    (diffusers_model / "model_index.json").write_text("{}", encoding="utf-8")
    incomplete_model = storage.models_dir / "incomplete-directory"
    incomplete_model.mkdir()
    (storage.loras_dir / "detail.safetensors").touch()
    adapter = storage.ip_adapters_dir / "sdxl-plus"
    (adapter / "image_encoder").mkdir(parents=True)
    (adapter / "ip-adapter-plus_sdxl_vit-h.safetensors").touch()
    incomplete = storage.ip_adapters_dir / "incomplete"
    incomplete.mkdir()
    (incomplete / "ip-adapter.safetensors").touch()

    assert storage.scan_weights() == {
        "models": ["A.SAFETENSORS", "sd-directory", "z.safetensors"],
        "loras": ["detail.safetensors"],
        "ip_adapters": ["sdxl-plus"],
    }
    assert storage.model_path("sd-directory") == diffusers_model
    assert storage.ip_adapter_path("sdxl-plus").name == "ip-adapter-plus_sdxl_vit-h.safetensors"


def test_reject_traversal(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.model_path("../secret.safetensors")
    incomplete = storage.models_dir / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        storage.model_path("incomplete")
    with pytest.raises(FileNotFoundError):
        storage.ip_adapter_path("../secret")


def test_history_newest_first(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    for stem, created in (("old", "2025-01-01T00:00:00Z"), ("new", "2026-01-01T00:00:00Z")):
        (storage.history_dir / f"{stem}.png").touch()
        storage.save_metadata(stem, {"created_at": created, "prompt": stem})
        metadata = storage.history_dir / f"{stem}.json"
        text = metadata.read_text(encoding="utf-8").replace(
            metadata.read_text(encoding="utf-8").split('"created_at": "')[1].split('"')[0], created
        )
        metadata.write_text(text, encoding="utf-8")

    assert [item["id"] for item in storage.list_history()] == ["new", "old"]


def test_delete_history_removes_image_and_metadata_only(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    history_id = "a" * 32
    (storage.history_dir / f"{history_id}.png").touch()
    storage.save_metadata(history_id, {"prompt": "delete me", "init_image": "source.png"})
    (storage.inputs_dir / "source.png").touch()

    assert storage.delete_history(history_id) is True
    assert not (storage.history_dir / f"{history_id}.png").exists()
    assert not (storage.history_dir / f"{history_id}.json").exists()
    assert (storage.inputs_dir / "source.png").exists()
    assert storage.delete_history(history_id) is False


def test_delete_history_rejects_invalid_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Storage(tmp_path).delete_history("../history")
