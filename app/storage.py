from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_WEIGHT_SUFFIXES = {".safetensors"}
HISTORY_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.models_dir = self.root / "models"
        self.loras_dir = self.root / "loras"
        self.ip_adapters_dir = self.root / "ip_adapters"
        self.history_dir = self.root / "data" / "history"
        self.inputs_dir = self.root / "data" / "inputs"
        for directory in (
            self.models_dir,
            self.loras_dir,
            self.ip_adapters_dir,
            self.history_dir,
            self.inputs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _scan(directory: Path) -> list[str]:
        return sorted(
            (path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_WEIGHT_SUFFIXES),
            key=str.casefold,
        )

    def scan_weights(self) -> dict[str, list[str]]:
        ip_adapters = sorted(
            (
                path.name
                for path in self.ip_adapters_dir.iterdir()
                if path.is_dir()
                and (path / "image_encoder").is_dir()
                and any(child.suffix.lower() == ".safetensors" for child in path.iterdir())
            ),
            key=str.casefold,
        )
        return {
            "models": self._scan(self.models_dir),
            "loras": self._scan(self.loras_dir),
            "ip_adapters": ip_adapters,
        }

    def model_path(self, name: str) -> Path:
        return self._safe_weight(self.models_dir, name)

    def lora_path(self, name: str) -> Path:
        return self._safe_weight(self.loras_dir, name)

    def ip_adapter_path(self, name: str) -> Path:
        package = (self.ip_adapters_dir / name).resolve()
        if package.parent != self.ip_adapters_dir.resolve() or not package.is_dir():
            raise FileNotFoundError("非法 IP-Adapter 包名")
        weights = sorted(
            path for path in package.iterdir() if path.suffix.lower() == ".safetensors"
        )
        if not weights or not (package / "image_encoder").is_dir():
            raise FileNotFoundError(f"IP-Adapter 包不完整：{name}")
        return weights[0]

    def input_image_path(self, name: str) -> Path:
        candidate = (self.inputs_dir / name).resolve()
        if candidate.parent != self.inputs_dir.resolve() or candidate.suffix.lower() != ".png":
            raise FileNotFoundError("非法参考图文件名")
        if not candidate.is_file():
            raise FileNotFoundError(f"参考图不存在：{name}")
        return candidate

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

    def delete_history(self, history_id: str) -> bool:
        if not HISTORY_ID_PATTERN.fullmatch(history_id):
            raise ValueError("非法历史记录 ID")
        image_path = self.history_dir / f"{history_id}.png"
        metadata_path = self.history_dir / f"{history_id}.json"
        if not image_path.is_file() and not metadata_path.is_file():
            return False
        image_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return True

    def list_history(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.history_dir.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                image_name = f"{path.stem}.png"
                if (self.history_dir / image_name).is_file():
                    init_image = item.get("init_image")
                    mask_image = item.get("mask_image")
                    ip_adapter_image = item.get("ip_adapter_image")
                    records.append(
                        {
                            **item,
                            "id": path.stem,
                            "image_url": f"/history/{image_name}",
                            "init_image_url": f"/inputs/{init_image}" if init_image else None,
                            "mask_image_url": f"/inputs/{mask_image}" if mask_image else None,
                            "ip_adapter_image_url": (
                                f"/inputs/{ip_adapter_image}" if ip_adapter_image else None
                            ),
                        }
                    )
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)
