"""Path and directory helper functions."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create directory if missing and return the resolved Path.

    Args:
        path: Directory path to create.

    Returns:
        Resolved directory Path object.
    """
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_session_save_path(save_root: str | Path, session_name: str) -> Path:
    """Create and return a save path under save_root for a session folder."""
    if not session_name.strip():
        raise ValueError("session_name must not be empty.")

    target_dir = ensure_directory(Path(save_root) / session_name.strip())
    return target_dir


def list_subdirectories(root_path: str | Path) -> list[str]:
    """Return sorted direct child folder names under root_path.

    If the root path does not exist, returns an empty list.
    """
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    subdirs = [item.name for item in root.iterdir() if item.is_dir()]
    return sorted(subdirs)


def list_subdirectories_relative(root_path: str | Path) -> list[str]:
    """Return sorted relative subdirectory paths under root_path."""
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    relative_dirs: list[str] = []
    for directory in root.rglob("*"):
        if directory.is_dir():
            relative_dirs.append(directory.relative_to(root).as_posix())
    return sorted(relative_dirs)


def collect_files_with_extensions(directory: str | Path, extensions: tuple[str, ...]) -> list[Path]:
    """Collect files in a directory filtered by allowed extensions.

    Returns an empty list when directory does not exist.
    """
    target = Path(directory).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        return []

    normalized = {ext.lower().lstrip(".") for ext in extensions}
    files = [
        file_path
        for file_path in target.iterdir()
        if file_path.is_file() and file_path.suffix.lower().lstrip(".") in normalized
    ]
    return sorted(files, key=lambda x: x.name)
