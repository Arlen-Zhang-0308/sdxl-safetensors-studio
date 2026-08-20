from app.tasks import TaskStore


def test_task_progress_snapshot_is_isolated() -> None:
    store = TaskStore()
    task_id = store.create(60)
    store.update(task_id, status="running", progress=50, current_step=30)

    snapshot = store.get(task_id)
    assert snapshot is not None
    snapshot["progress"] = 99

    current = store.get(task_id)
    assert current is not None
    assert current["progress"] == 50
    assert current["total_steps"] == 60


def test_unknown_task_is_none() -> None:
    assert TaskStore().get("missing") is None
