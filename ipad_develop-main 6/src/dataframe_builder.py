"""Build master dataframe (1 row per cell_id) from image_map."""

from __future__ import annotations

import pandas as pd

from src.constants import (
    BASE_DF_COLUMNS,
    COL_CELL_ID,
    DEFECT_COLUMNS,
    POSITION_COLUMNS,
)
from src.image_registry import ImageMap


def build_master_dataframe(image_map: ImageMap) -> pd.DataFrame:
    """Create master dataframe from image_map with required schema.

    - row unit: one row per cell_id
    - position columns: image existence as 1/0
    - defect columns: initialized as empty strings
    """
    rows: list[dict[str, object]] = []

    for cell_id in sorted(image_map.keys()):
        position_dict = image_map.get(cell_id, {})

        row: dict[str, object] = {COL_CELL_ID: cell_id}
        for position in POSITION_COLUMNS:
            row[position] = 1 if position in position_dict else 0
        for defect_col in DEFECT_COLUMNS:
            row[defect_col] = ""

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=BASE_DF_COLUMNS)

    return pd.DataFrame(rows, columns=BASE_DF_COLUMNS)
