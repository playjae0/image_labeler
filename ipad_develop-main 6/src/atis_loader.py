"""ATIS data loading and merge utilities for upload step."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import ATIS_COLUMN_MAPPING, ATIS_FILE_PATH
from src.constants import COL_CELL_ID


def merge_atis_to_master(master_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Merge ATIS data into master dataframe using left join on cell_id.

    Behavior:
    - If ATIS file is missing, return original dataframe unchanged.
    - If mapping is empty, return original dataframe unchanged.
    - Keeps master dataframe row unit as one row per cell_id.
    """
    atis_path = Path(ATIS_FILE_PATH)
    if not atis_path.exists():
        return master_df, "ATIS 파일이 없어 병합 없이 진행합니다."

    if not ATIS_COLUMN_MAPPING:
        return master_df, "ATIS 컬럼 매핑이 비어 있어 병합 없이 진행합니다."

    source_cell_id_col = ATIS_COLUMN_MAPPING.get(COL_CELL_ID, COL_CELL_ID)
    required_source_columns = [source_cell_id_col, *ATIS_COLUMN_MAPPING.values()]

    atis_df = pd.read_excel(atis_path, usecols=lambda c: c in set(required_source_columns))

    if source_cell_id_col not in atis_df.columns:
        return master_df, f"ATIS 파일에 cell_id 원본 컬럼(`{source_cell_id_col}`)이 없어 병합 없이 진행합니다."

    rename_map = {source: target for target, source in ATIS_COLUMN_MAPPING.items()}
    atis_df = atis_df.rename(columns=rename_map)

    if COL_CELL_ID not in atis_df.columns:
        return master_df, "ATIS 매핑 결과에 cell_id 컬럼이 없어 병합 없이 진행합니다."

    # Normalize both IDs to comparable 16-char alnum text (cell_id format requirement).
    atis_df[COL_CELL_ID] = atis_df[COL_CELL_ID].apply(_normalize_cell_id)

    target_columns = [COL_CELL_ID, *[k for k in ATIS_COLUMN_MAPPING.keys() if k != COL_CELL_ID]]
    atis_df = atis_df[[col for col in target_columns if col in atis_df.columns]]
    atis_df = atis_df.drop_duplicates(subset=[COL_CELL_ID])

    master_df = master_df.copy()
    master_df[COL_CELL_ID] = master_df[COL_CELL_ID].apply(_normalize_cell_id)

    merged_df = master_df.merge(atis_df, on=COL_CELL_ID, how="left")
    return merged_df, "ATIS 데이터를 병합했습니다."


def _normalize_cell_id(value: object) -> str:
    """Normalize cell_id to uppercase 16-char alnum string when possible."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]

    normalized = "".join(ch for ch in text if ch.isalnum()).upper()
    return normalized
