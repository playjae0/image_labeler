"""Labeling activity logging helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config import IMAGE_ROOT_DIR
from src.auth.user_store import get_connection
from src.constants import (
    COL_DEFECT_AN_BOT,
    COL_DEFECT_AN_TOP,
    COL_DEFECT_CA_BOT,
    COL_DEFECT_CA_TOP,
)


def count_labeled_cells(df: pd.DataFrame) -> int:
    """Count rows with at least one non-empty defect value."""
    defect_columns = [
        COL_DEFECT_CA_TOP,
        COL_DEFECT_CA_BOT,
        COL_DEFECT_AN_TOP,
        COL_DEFECT_AN_BOT,
    ]
    available = [column for column in defect_columns if column in df.columns]
    if not available:
        return 0

    labeled_mask = df[available].fillna("").astype(str).apply(lambda col: col.str.strip())
    return int((labeled_mask != "").any(axis=1).sum())


def insert_activity_log(
    db_path: str | Path,
    *,
    employee_id: str,
    line: str,
    period: str,
    dataset_path: str,
    labeled_cells: int,
) -> None:
    """Insert one labeling activity row."""
    timestamp = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO labeling_activity_log (
                employee_id, line, period, dataset_path, labeled_cells, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (employee_id, line, period, dataset_path, int(labeled_cells), timestamp),
        )
        conn.commit()


def log_labeling_activity(
    *,
    db_path: str | Path,
    employee_id: str,
    selected_subpath: str,
    df: pd.DataFrame,
) -> None:
    """Parse selected dataset subpath and insert activity log."""
    parts = [part for part in selected_subpath.strip().split("/") if part]
    if len(parts) < 2:
        return

    line, period = parts[0], parts[1]
    dataset_path = str(Path(IMAGE_ROOT_DIR) / line / period)
    labeled_cells = count_labeled_cells(df)

    insert_activity_log(
        db_path,
        employee_id=employee_id,
        line=line,
        period=period,
        dataset_path=dataset_path,
        labeled_cells=labeled_cells,
    )


def get_labeling_activity_logs(db_path: str | Path) -> list[dict[str, object]]:
    """Fetch labeling activity logs ordered by latest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT employee_id, line, period, dataset_path, labeled_cells, timestamp
            FROM labeling_activity_log
            ORDER BY timestamp DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]
