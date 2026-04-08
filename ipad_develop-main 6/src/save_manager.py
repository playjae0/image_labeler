"""Saving utilities for CSV export and labeled image export."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from config import CSV_OUTPUT_ROOT_DIR, TIMESTAMP_FORMAT
from src.constants import (
    COL_CELL_ID,
    COL_DEFECT_AN_BOT,
    COL_DEFECT_AN_TOP,
    COL_DEFECT_CA_BOT,
    COL_DEFECT_CA_TOP,
)
from utils.io_utils import dataframe_to_csv_bytes, get_timestamp
from utils.naming_utils import sanitize_token
from utils.path_utils import ensure_directory

POSITION_TO_DEFECT_COLUMN: dict[str, str] = {
    "CA(TOP)": COL_DEFECT_CA_TOP,
    "CA(BOT)": COL_DEFECT_CA_BOT,
    "AN(TOP)": COL_DEFECT_AN_TOP,
    "AN(BOT)": COL_DEFECT_AN_BOT,
}


def build_csv_export_payload(df: pd.DataFrame, base_filename: str) -> tuple[str, bytes]:
    """Build timestamped CSV filename and payload bytes.

    Returns:
        (output_filename, csv_bytes)
    """
    safe_base = sanitize_token(base_filename or "labeling_result", fallback="labeling_result")
    timestamp = get_timestamp(TIMESTAMP_FORMAT)
    output_name = f"{safe_base}_{timestamp}.csv"
    return output_name, dataframe_to_csv_bytes(df)


def build_csv_filename(
    base_filename: str,
    *,
    custom_suffix: str = "",
    timestamp_format: str = "%m%d_%H%M",
) -> str:
    """Build CSV filename with base + time + optional suffix."""
    safe_base = sanitize_token(base_filename or "labeling_result", fallback="labeling_result")
    timestamp = get_timestamp(timestamp_format)
    safe_suffix = sanitize_token(custom_suffix, fallback="") if custom_suffix.strip() else ""
    suffix_part = f"_{safe_suffix}" if safe_suffix else ""
    return f"{safe_base}_{timestamp}{suffix_part}.csv"


def save_csv_to_path(df: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    """Persist CSV to output_dir/filename and return the saved path."""
    target_dir = ensure_directory(output_dir)
    target_path = target_dir / Path(filename).name
    target_path.write_bytes(dataframe_to_csv_bytes(df))
    return target_path


VERSION_PATTERN = re.compile(r"_ver(\d+)\.(\d+)\.csv$", re.IGNORECASE)


def parse_version_from_filename(filename: str) -> tuple[int, int] | None:
    """Parse version tuple (major, minor) from file name suffix `_verX.Y.csv`."""
    match = VERSION_PATTERN.search(filename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def find_latest_csv_version(result_folder: str | Path) -> tuple[int, int]:
    """Find highest CSV version in folder; default to (0, 0) when empty."""
    folder = Path(result_folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        return 0, 0
    versions: list[tuple[int, int]] = []
    for file_path in folder.glob("*.csv"):
        parsed = parse_version_from_filename(file_path.name)
        if parsed is not None:
            versions.append(parsed)

    if not versions:
        return 0, 0
    return max(versions)


def build_next_version(latest: tuple[int, int]) -> tuple[int, int]:
    """Return next version where minor rolls over after 9."""
    major, minor = latest
    if minor < 9:
        return major, minor + 1
    return major + 1, 0


def build_next_version_filename(line: str, period: str, employee_id: str, version: tuple[int, int]) -> str:
    """Build `{line}_{period}_{employee_id}_verX.Y.csv` file name."""
    safe_line = sanitize_token(line, fallback="line")
    safe_period = sanitize_token(period, fallback="period")
    safe_employee = sanitize_token(employee_id, fallback="unknown")
    major, minor = version
    return f"{safe_line}_{safe_period}_{safe_employee}_ver{major}.{minor}.csv"


def ensure_result_folder_from_selected_subpath(csv_root: str | Path, selected_subpath: str) -> Path:
    """Return CSV output folder path for selected `line/period` subpath."""
    clean_subpath = selected_subpath.strip().strip("/")
    return Path(csv_root).expanduser().resolve() / clean_subpath


def export_csv_without_filling_ok(df: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    """Export CSV as-is without auto-filling empty defect values."""
    return save_csv_to_path(df, output_dir, filename)


def create_csv_backup_copy(
    *,
    saved_csv_path: str | Path,
    backup_root: str | Path,
    selected_subpath: str | None = None,
) -> Path:
    """Create one backup copy under backup_root/{line}/{period}/ and return the backup path."""
    source_path = Path(saved_csv_path).expanduser().resolve()
    backup_dir = _resolve_backup_directory(
        saved_csv_path=source_path,
        backup_root=backup_root,
        selected_subpath=selected_subpath,
    )
    ensure_directory(backup_dir)

    timestamp = get_timestamp("%m%d_%H%M%S")
    candidate = backup_dir / f"{timestamp}__{source_path.name}"
    if candidate.exists():
        suffix = 1
        while True:
            candidate = backup_dir / f"{timestamp}_{suffix:02d}__{source_path.name}"
            if not candidate.exists():
                break
            suffix += 1

    shutil.copy2(source_path, candidate)
    return candidate


def _resolve_backup_directory(
    *,
    saved_csv_path: Path,
    backup_root: str | Path,
    selected_subpath: str | None,
) -> Path:
    """Resolve backup directory with dataset structure when available."""
    backup_root_path = Path(backup_root).expanduser().resolve()

    if isinstance(selected_subpath, str):
        parts = [part for part in selected_subpath.split("/") if part]
        if len(parts) >= 2:
            return backup_root_path / parts[0] / parts[1]

    working_root = Path(CSV_OUTPUT_ROOT_DIR).expanduser().resolve()
    try:
        relative_parent = saved_csv_path.parent.relative_to(working_root)
    except ValueError:
        relative_parent = Path("_manual/_manual")

    if relative_parent == Path("."):
        relative_parent = Path("_manual/_manual")

    return backup_root_path / relative_parent


def find_latest_csv_file(result_folder: str | Path) -> Path | None:
    """Return latest versioned CSV path in folder, if present."""
    folder = Path(result_folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        return None
    candidates: list[tuple[tuple[int, int], Path]] = []
    for file_path in folder.glob("*.csv"):
        parsed = parse_version_from_filename(file_path.name)
        if parsed is not None:
            candidates.append((parsed, file_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def extract_employee_and_version_from_filename(filename: str) -> tuple[str, str] | None:
    """Extract `(employee_id, vX.Y)` from `{line}_{period}_{employee}_verX.Y.csv`."""
    parsed = parse_version_from_filename(filename)
    if parsed is None:
        return None

    stem = Path(filename).stem
    stem_no_ver = re.sub(r"_ver\d+\.\d+$", "", stem, flags=re.IGNORECASE)
    parts = stem_no_ver.split("_")
    if len(parts) < 3:
        return None

    employee_id = parts[-1]
    major, minor = parsed
    return employee_id, f"v{major}.{minor}"


def load_previous_defect_values(csv_path: str | Path) -> pd.DataFrame:
    """Load previous CSV and return cell_id + defect + ATIS-like columns."""
    loaded_df = pd.read_csv(csv_path)
    required_columns = [
        COL_CELL_ID,
        COL_DEFECT_CA_TOP,
        COL_DEFECT_CA_BOT,
        COL_DEFECT_AN_TOP,
        COL_DEFECT_AN_BOT,
    ]
    atis_columns = [
        column
        for column in loaded_df.columns
        if column.lower().startswith("atis")
    ]
    available_columns = [column for column in required_columns if column in loaded_df.columns]
    available_columns.extend([column for column in atis_columns if column not in available_columns])
    return loaded_df[available_columns].copy()


def apply_loaded_defect_values(current_df: pd.DataFrame, loaded_df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite defect columns and changed ATIS columns on current dataframe by `cell_id` match."""
    if COL_CELL_ID not in loaded_df.columns:
        return current_df

    defect_columns = [
        COL_DEFECT_CA_TOP,
        COL_DEFECT_CA_BOT,
        COL_DEFECT_AN_TOP,
        COL_DEFECT_AN_BOT,
    ]
    target_defect_columns = [column for column in defect_columns if column in loaded_df.columns]
    atis_columns = [
        column
        for column in loaded_df.columns
        if column.lower().startswith("atis") and column in current_df.columns
    ]
    if not target_defect_columns and not atis_columns:
        return current_df

    merged = current_df.copy()
    indexed_loaded = loaded_df.set_index(COL_CELL_ID)
    for idx, row in merged.iterrows():
        cell_id = row[COL_CELL_ID]
        if cell_id not in indexed_loaded.index:
            continue
        for column in target_defect_columns:
            merged.at[idx, column] = indexed_loaded.at[cell_id, column]
        for column in atis_columns:
            loaded_value = indexed_loaded.at[cell_id, column]
            current_value = merged.at[idx, column]
            loaded_text = "" if pd.isna(loaded_value) else str(loaded_value).strip()
            current_text = "" if pd.isna(current_value) else str(current_value).strip()
            if loaded_text and loaded_text != current_text:
                merged.at[idx, column] = loaded_value
    return merged


def save_defect_images(
    *,
    df: pd.DataFrame,
    image_map: dict[str, dict[str, Any]],
    save_root: str | Path,
    employee_id: str,
    custom_folder: str | None = None,
) -> dict[str, int]:
    """Save non-empty and non-ok defect images into structured folders.

    Save structure:
        [custom_folder/][position]/[defect]/[position]_[defect]_[cell_id]_[employee_id].jpg
    """
    export_root = Path(save_root)
    if custom_folder and custom_folder.strip():
        export_root = export_root / sanitize_token(custom_folder, fallback="custom")
    export_root = ensure_directory(export_root)

    safe_employee = sanitize_token(employee_id or "unknown", fallback="unknown")

    saved_count = 0
    skipped_count = 0

    for _, row in df.iterrows():
        cell_id = str(row[COL_CELL_ID])

        for position, defect_column in POSITION_TO_DEFECT_COLUMN.items():
            defect_raw = str(row.get(defect_column, "") or "").strip()
            if not defect_raw:
                skipped_count += 1
                continue
            if defect_raw.lower() == "ok":
                skipped_count += 1
                continue

            image_ref = image_map.get(cell_id, {}).get(position)
            if image_ref is None:
                skipped_count += 1
                continue

            try:
                safe_position = sanitize_token(position)
                safe_defect = sanitize_token(defect_raw)
                safe_cell_id = sanitize_token(cell_id)
                file_name = f"{safe_position}_{safe_defect}_{safe_cell_id}_{safe_employee}.jpg"
                defect_dir = ensure_directory(
                    export_root / safe_position / safe_defect
                )
                output_path = defect_dir / file_name
                output_path.write_bytes(_read_image_bytes(image_ref))
                saved_count += 1
            except Exception:
                skipped_count += 1

    return {"saved": saved_count, "skipped": skipped_count}


def _read_image_bytes(image_ref: Any) -> bytes:
    """Read image bytes from path-like or uploaded-file-like object."""
    if isinstance(image_ref, Path):
        return image_ref.read_bytes()

    if isinstance(image_ref, str):
        return Path(image_ref).read_bytes()

    if hasattr(image_ref, "getvalue"):
        data = image_ref.getvalue()
        return data if isinstance(data, bytes) else bytes(data)

    if hasattr(image_ref, "read"):
        data = image_ref.read()
        if hasattr(image_ref, "seek"):
            image_ref.seek(0)
        return data if isinstance(data, bytes) else bytes(data)

    raise ValueError("Unsupported image reference type for saving.")
