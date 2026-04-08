"""Naming helpers for defect image export."""

from __future__ import annotations

import re
from pathlib import Path


_SANITIZE_PATTERN = re.compile(r"[^\w가-힣()_-]+", re.UNICODE)


def sanitize_token(raw: str, *, fallback: str = "unknown") -> str:
    """Sanitize position/defect-like text for path-safe naming."""
    cleaned = _SANITIZE_PATTERN.sub("_", raw.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def build_labeled_image_filename(
    position: str,
    defect: str,
    cell_id: str,
    extension: str,
) -> str:
    """Build export filename: [position]_[defect]_[cell_id].[ext]."""
    safe_position = sanitize_token(position)
    safe_defect = sanitize_token(defect)
    safe_cell_id = sanitize_token(cell_id)

    ext = extension.lower().lstrip(".")
    if not ext:
        raise ValueError("extension must not be empty.")

    return f"{safe_position}_{safe_defect}_{safe_cell_id}.{ext}"


def infer_extension_from_path(file_name_or_path: str) -> str:
    """Infer file extension from a file name/path without dot."""
    suffix = Path(file_name_or_path).suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError("Could not infer extension from input path.")
    return suffix
