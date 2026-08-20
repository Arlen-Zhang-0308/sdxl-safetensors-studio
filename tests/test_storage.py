from pathlib import Path

import pytest

from app.storage import Storage


def test_scan_only_safetensors(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    (storage.models_dir / "z.safetensors").touch()
    (storage.models_dir / "A.SAFETENSORS").touch()
    (storage.models_dir / "ignore.ckpt").touch()
    (storage.loras_dir / "detail.safetensors").touch()

    assert storage.scan_weights() == {
        "models": ["A.SAFETENSORS", "z.safetensors"],
        "loras": ["detail.safetensors"],
    }


def test_reject_traversal(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.model_path("../secret.safetensors")


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
