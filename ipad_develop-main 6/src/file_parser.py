"""Filename parsing utilities for uploaded defect images."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import POSITIONS


TAB_CELL_ID_PATTERN = re.compile(r"TAB_([A-Za-z0-9]{16})")
CRACK_POSITION_PATTERN = re.compile(r"CRACK\s+(.{7})")


@dataclass(frozen=True)
class FileParseResult:
    """Result model for a filename parsing attempt."""

    filename: str
    cell_id: str | None
    position: str | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.cell_id is not None and self.position is not None


def parse_filename(file_reference: Any) -> FileParseResult:
    """Parse cell_id and position from image file name.

    Rules:
    - cell_id: first 16-char alphanumeric value after `TAB_`
    - position: first 7 chars after `CRACK` + spaces

    Args:
        file_reference: Uploaded file object or file path-like object.

    Returns:
        FileParseResult including error message on failure.
    """
    filename = _extract_filename(file_reference)

    cell_match = TAB_CELL_ID_PATTERN.search(filename)
    if cell_match is None:
        return FileParseResult(
            filename=filename,
            cell_id=None,
            position=None,
            error="Failed to parse cell_id: missing `TAB_` + 16 alphanumeric pattern.",
        )
    cell_id = cell_match.group(1)

    position_match = CRACK_POSITION_PATTERN.search(filename)
    if position_match is None:
        return FileParseResult(
            filename=filename,
            cell_id=cell_id,
            position=None,
            error="Failed to parse position: missing `CRACK <7 chars>` pattern.",
        )

    position = position_match.group(1)
    if position not in POSITIONS:
        return FileParseResult(
            filename=filename,
            cell_id=cell_id,
            position=None,
            error=f"Invalid position `{position}`. Allowed positions: {POSITIONS}.",
        )

    return FileParseResult(filename=filename, cell_id=cell_id, position=position, error=None)


def _extract_filename(file_reference: Any) -> str:
    """Extract filename text from a path or uploaded file-like object."""
    if hasattr(file_reference, "name"):
        return Path(str(file_reference.name)).name
    return Path(str(file_reference)).name
