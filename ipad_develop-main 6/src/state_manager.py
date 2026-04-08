"""Session-state management utilities for the Streamlit app (Step 1)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.constants import (
    KEY_CURRENT_CELL_INDEX,
    KEY_EAGER_THRESHOLD,
    KEY_IMAGE_MAP,
    KEY_IMAGE_LOADING_MODE,
    KEY_LABEL_SYNC_TOKEN,
    KEY_MASTER_DF,
    KEY_PRELOAD_BACKWARD_COUNT,
    KEY_PRELOAD_FORWARD_COUNT,
    KEY_RESOLVED_LOADING_STRATEGY,
    KEY_SELECTED_CELL_ID,
    KEY_UPLOAD_COMPLETED,
    SESSION_DEFAULTS,
)


def initialize_session_state() -> None:
    """Initialize required session_state keys with stable defaults."""
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            # Copy mutable defaults to avoid accidental cross-reference updates.
            st.session_state[key] = default_value.copy() if isinstance(default_value, dict) else default_value


def set_master_dataframe(df: pd.DataFrame | None) -> None:
    """Store the master dataframe in session state."""
    st.session_state[KEY_MASTER_DF] = df


def get_master_dataframe() -> pd.DataFrame | None:
    """Return the stored master dataframe, if available."""
    value: Any = st.session_state.get(KEY_MASTER_DF)
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    raise TypeError("Session state value for master dataframe is not a pandas DataFrame.")


def set_image_map(image_map: dict[str, dict[str, Any]]) -> None:
    """Store image references outside dataframe as image_map."""
    st.session_state[KEY_IMAGE_MAP] = image_map


def get_image_map() -> dict[str, dict[str, Any]]:
    """Get image_map from session state."""
    value: Any = st.session_state.get(KEY_IMAGE_MAP, {})
    if isinstance(value, dict):
        return value
    raise TypeError("Session state value for image_map is not a dictionary.")


def set_current_cell_index(index: int) -> None:
    """Store selected cell index for navigation persistence."""
    if index < 0:
        raise ValueError("Current cell index cannot be negative.")
    st.session_state[KEY_CURRENT_CELL_INDEX] = index


def get_current_cell_index() -> int:
    """Retrieve current selected cell index."""
    value: Any = st.session_state.get(KEY_CURRENT_CELL_INDEX, 0)
    if isinstance(value, int):
        return value
    raise TypeError("Session state value for current cell index is not an integer.")


def set_selected_cell_id(cell_id: str | None) -> None:
    """Store selected cell_id as stable selection source."""
    st.session_state[KEY_SELECTED_CELL_ID] = cell_id


def get_selected_cell_id() -> str | None:
    """Get selected cell_id from session state."""
    value: Any = st.session_state.get(KEY_SELECTED_CELL_ID)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def set_upload_completed(is_completed: bool) -> None:
    """Update upload completion status."""
    st.session_state[KEY_UPLOAD_COMPLETED] = is_completed


def is_upload_completed() -> bool:
    """Return upload completion status."""
    value: Any = st.session_state.get(KEY_UPLOAD_COMPLETED, False)
    if isinstance(value, bool):
        return value
    raise TypeError("Session state value for upload completion is not a boolean.")


def touch_label_sync_token() -> int:
    """Increment and return a sync token to trigger labeling UI refresh logic."""
    current = st.session_state.get(KEY_LABEL_SYNC_TOKEN, 0)
    if not isinstance(current, int):
        current = 0
    current += 1
    st.session_state[KEY_LABEL_SYNC_TOKEN] = current
    return current


def get_label_sync_token() -> int:
    """Get the current label sync token value."""
    value: Any = st.session_state.get(KEY_LABEL_SYNC_TOKEN, 0)
    if isinstance(value, int):
        return value
    raise TypeError("Session state value for label sync token is not an integer.")


def set_image_loading_settings(
    *,
    image_loading_mode: str,
    eager_threshold: int,
    preload_forward_count: int,
    preload_backward_count: int,
) -> None:
    """Store image loading strategy inputs in session state."""
    st.session_state[KEY_IMAGE_LOADING_MODE] = image_loading_mode
    st.session_state[KEY_EAGER_THRESHOLD] = eager_threshold
    st.session_state[KEY_PRELOAD_FORWARD_COUNT] = preload_forward_count
    st.session_state[KEY_PRELOAD_BACKWARD_COUNT] = preload_backward_count


def get_image_loading_settings() -> dict[str, int | str]:
    """Get image loading strategy inputs from session state."""
    return {
        "image_loading_mode": st.session_state.get(KEY_IMAGE_LOADING_MODE, "auto"),
        "eager_threshold": int(st.session_state.get(KEY_EAGER_THRESHOLD, 200)),
        "preload_forward_count": int(st.session_state.get(KEY_PRELOAD_FORWARD_COUNT, 2)),
        "preload_backward_count": int(st.session_state.get(KEY_PRELOAD_BACKWARD_COUNT, 1)),
    }


def set_resolved_loading_strategy(strategy: str) -> None:
    """Store resolved image loading strategy in session state."""
    st.session_state[KEY_RESOLVED_LOADING_STRATEGY] = strategy


def get_resolved_loading_strategy() -> str:
    """Get resolved image loading strategy from session state."""
    return str(st.session_state.get(KEY_RESOLVED_LOADING_STRATEGY, "auto"))


def set_sidebar_cell_index(index: int) -> None:
    """Store sidebar cell radio widget index for sync with current cell."""
    st.session_state["sidebar_cell_index"] = index


def get_sidebar_cell_index(default: int = 0) -> int:
    """Get sidebar cell radio widget index."""
    value = st.session_state.get("sidebar_cell_index", default)
    return int(value) if isinstance(value, int) else default
