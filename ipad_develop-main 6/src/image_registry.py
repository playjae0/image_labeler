"""Image registry utilities to keep file references outside dataframe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.file_parser import FileParseResult


ImageMap = dict[str, dict[str, Any]]


def register_image(image_map: ImageMap, parse_result: FileParseResult, file_reference: Any) -> None:
    """Register a parsed image reference into image_map.

    Structure:
        image_map[cell_id][position] = file_reference
    """
    if not parse_result.is_valid or parse_result.cell_id is None or parse_result.position is None:
        raise ValueError("Cannot register image: parse_result is invalid.")

    if parse_result.cell_id not in image_map:
        image_map[parse_result.cell_id] = {}

    image_map[parse_result.cell_id][parse_result.position] = file_reference


def build_image_map(parsed_pairs: list[tuple[FileParseResult, Any]]) -> ImageMap:
    """Build image_map from a list of `(FileParseResult, file_reference)` pairs.

    Invalid parse entries are ignored by design and should be handled in validation.
    """
    image_map: ImageMap = {}
    for parse_result, file_reference in parsed_pairs:
        if parse_result.is_valid:
            register_image(image_map=image_map, parse_result=parse_result, file_reference=file_reference)
    return image_map


def load_image_bytes(file_reference: Any) -> bytes:
    """Load raw image bytes from a stored image reference."""
    if isinstance(file_reference, bytes):
        return file_reference

    if isinstance(file_reference, bytearray):
        return bytes(file_reference)

    if isinstance(file_reference, Path):
        return file_reference.read_bytes()

    if isinstance(file_reference, str):
        return Path(file_reference).read_bytes()

    if hasattr(file_reference, "getvalue"):
        data = file_reference.getvalue()
        return data if isinstance(data, bytes) else bytes(data)

    if hasattr(file_reference, "read"):
        data = file_reference.read()
        if hasattr(file_reference, "seek"):
            file_reference.seek(0)
        return data if isinstance(data, bytes) else bytes(data)

    raise ValueError("Unsupported image reference type for byte loading.")
