from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class TaskStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, total_steps: int) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._items[task_id] = {
                "id": task_id,
                "status": "queued",
                "progress": 0,
                "current_step": 0,
                "total_steps": total_steps,
                "current_batch": 0,
                "message": "任务已进入队列",
                "images": [],
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        return task_id

    def update(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            if task_id in self._items:
                self._items[task_id].update(changes)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(task_id)
            return deepcopy(item) if item is not None else None
