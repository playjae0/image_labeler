"""Validation helpers for upload inputs and parsing results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_COUNT
from src.file_parser import FileParseResult, parse_filename


def validate_file_extensions(file_references: list[Any]) -> list[str]:
    """Return filenames with unsupported extensions."""
    invalid_files: list[str] = []

    for file_ref in file_references:
        file_name = _extract_filename(file_ref)
        extension = Path(file_name).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            invalid_files.append(file_name)

    return invalid_files


def validate_file_count(file_references: list[Any]) -> str | None:
    """Validate max upload count and return error message if invalid."""
    if len(file_references) > MAX_UPLOAD_COUNT:
        return f"Upload limit exceeded: {len(file_references)} files (max: {MAX_UPLOAD_COUNT})."
    return None


def parse_files_with_results(file_references: list[Any]) -> list[tuple[FileParseResult, Any]]:
    """Parse all file references and keep pair of parse result + original reference."""
    return [(parse_filename(file_ref), file_ref) for file_ref in file_references]


def extract_parse_failures(parsed_pairs: list[tuple[FileParseResult, Any]]) -> list[dict[str, str]]:
    """Collect parsing failure information for UI error reporting."""
    failures: list[dict[str, str]] = []

    for parse_result, _ in parsed_pairs:
        if not parse_result.is_valid:
            failures.append(
                {
                    "filename": parse_result.filename,
                    "error": parse_result.error or "Unknown parsing error.",
                }
            )

    return failures


def _extract_filename(file_reference: Any) -> str:
    """Extract filename text from uploaded file-like object or path string."""
    if hasattr(file_reference, "name"):
        return Path(str(file_reference.name)).name
    return Path(str(file_reference)).name
