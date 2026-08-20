from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_WEIGHT_SUFFIXES = {".safetensors"}


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.models_dir = self.root / "models"
        self.loras_dir = self.root / "loras"
        self.history_dir = self.root / "data" / "history"
        for directory in (self.models_dir, self.loras_dir, self.history_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _scan(directory: Path) -> list[str]:
        return sorted(
            (path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_WEIGHT_SUFFIXES),
            key=str.casefold,
        )

    def scan_weights(self) -> dict[str, list[str]]:
        return {"models": self._scan(self.models_dir), "loras": self._scan(self.loras_dir)}

    def model_path(self, name: str) -> Path:
        return self._safe_weight(self.models_dir, name)

    def lora_path(self, name: str) -> Path:
        return self._safe_weight(self.loras_dir, name)

    @staticmethod
    def _safe_weight(directory: Path, name: str) -> Path:
        candidate = (directory / name).resolve()
        if candidate.parent != directory.resolve() or candidate.suffix.lower() not in SUPPORTED_WEIGHT_SUFFIXES:
            raise FileNotFoundError("非法权重文件名")
        if not candidate.is_file():
            raise FileNotFoundError(f"权重文件不存在：{name}")
        return candidate

    def save_metadata(self, stem: str, metadata: dict[str, Any]) -> None:
        payload = {**metadata, "created_at": datetime.now(timezone.utc).isoformat()}
        (self.history_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list_history(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.history_dir.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                image_name = f"{path.stem}.png"
                if (self.history_dir / image_name).is_file():
                    records.append({**item, "id": path.stem, "image_url": f"/history/{image_name}"})
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)
