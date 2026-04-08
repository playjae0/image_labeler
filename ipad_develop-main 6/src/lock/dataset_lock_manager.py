"""Dataset lock manager for concurrent labeling prevention."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.auth.user_store import get_connection

LOCK_TIMEOUT_MINUTES = 30


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _is_expired(locked_at: str) -> bool:
    parsed = _parse_iso(locked_at)
    if parsed is None:
        return True
    return datetime.utcnow() - parsed > timedelta(minutes=LOCK_TIMEOUT_MINUTES)


def check_lock(db_path: str | Path, dataset_key: str) -> dict[str, Any] | None:
    """Return active lock info for dataset, or None if no valid lock."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT dataset_key, employee_id, locked_at FROM dataset_lock WHERE dataset_key = ? LIMIT 1",
            (dataset_key,),
        ).fetchone()
        if row is None:
            return None

        lock_info = dict(row)
        if _is_expired(str(lock_info.get("locked_at", ""))):
            conn.execute("DELETE FROM dataset_lock WHERE dataset_key = ?", (dataset_key,))
            conn.commit()
            return None
        return lock_info


def acquire_lock(db_path: str | Path, dataset_key: str, employee_id: str) -> tuple[bool, dict[str, Any] | None]:
    """Acquire dataset lock; allow refresh if same user owns lock."""
    now = datetime.utcnow().isoformat()
    current = check_lock(db_path, dataset_key)
    if current is None:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO dataset_lock (dataset_key, employee_id, locked_at) VALUES (?, ?, ?)",
                (dataset_key, employee_id, now),
            )
            conn.commit()
        return True, None

    if str(current.get("employee_id", "")) == employee_id:
        with get_connection(db_path) as conn:
            conn.execute(
                "UPDATE dataset_lock SET locked_at = ? WHERE dataset_key = ?",
                (now, dataset_key),
            )
            conn.commit()
        return True, current

    return False, current


def release_lock(db_path: str | Path, dataset_key: str, employee_id: str) -> None:
    """Release lock only when held by the given employee."""
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM dataset_lock WHERE dataset_key = ? AND employee_id = ?",
            (dataset_key, employee_id),
        )
        conn.commit()


def force_unlock(db_path: str | Path, dataset_key: str) -> None:
    """Force-remove lock row regardless of owner."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM dataset_lock WHERE dataset_key = ?", (dataset_key,))
        conn.commit()


def get_active_locks(db_path: str | Path) -> list[dict[str, Any]]:
    """Return active (non-expired) locks latest-first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT dataset_key, employee_id, locked_at FROM dataset_lock ORDER BY locked_at DESC"
        ).fetchall()

    active: list[dict[str, Any]] = []
    for row in rows:
        info = dict(row)
        if _is_expired(str(info.get("locked_at", ""))):
            force_unlock(db_path, str(info.get("dataset_key", "")))
            continue
        active.append(info)
    return active
