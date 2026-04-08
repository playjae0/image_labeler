"""Sidebar cell list UI for labeling page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.constants import COL_CELL_ID, DEFECT_COLUMNS, POSITION_COLUMNS


SIDEBAR_CELL_INDEX_KEY = "sidebar_cell_index"
SIDEBAR_FORCE_SYNC_KEY = "sidebar_force_sync"
SIDEBAR_SELECTION_CHANGED_KEY = "sidebar_selection_changed"
SIDEBAR_FILTER_POSITION_KEY = "sidebar_filter_position"
SIDEBAR_FILTER_TOP_KEY = "sidebar_filter_top"
SIDEBAR_FILTER_SUB_KEY = "sidebar_filter_sub"

POSITION_TO_DEFECT_COLUMN = {
    "CA(TOP)": "Defect_CA(TOP)",
    "CA(BOT)": "Defect_CA(BOT)",
    "AN(TOP)": "Defect_AN(TOP)",
    "AN(BOT)": "Defect_AN(BOT)",
}

POSITION_TO_ATIS_COLUMN = {
    position: f"ATIS_{position}" for position in POSITION_COLUMNS
}


def render_sidebar_cell_list(df: pd.DataFrame, current_index: int) -> int:
    """Render sorted cell list with current index mirrored into sidebar widget state.

    Returns updated selected index.
    """
    sorted_df = df.sort_values(COL_CELL_ID).reset_index(drop=True)
    filters = _render_sidebar_filters(sorted_df)
    visible_indices = _filter_sidebar_indices(sorted_df, filters)

    st.sidebar.subheader("Cell List")

    if not visible_indices:
        st.sidebar.info("조건에 맞는 cell이 없습니다.")
        st.session_state[SIDEBAR_FORCE_SYNC_KEY] = False
        st.session_state[SIDEBAR_SELECTION_CHANGED_KEY] = False
        return int(current_index)

    max_index = max(len(sorted_df) - 1, 0)
    safe_index = min(max(current_index, 0), max_index)
    current_visible = safe_index in visible_indices

    force_sync = bool(st.session_state.get(SIDEBAR_FORCE_SYNC_KEY, False))
    selected_widget_index = st.session_state.get(SIDEBAR_CELL_INDEX_KEY)
    if force_sync:
        if current_visible:
            selected_widget_index = safe_index
        elif selected_widget_index not in visible_indices:
            selected_widget_index = visible_indices[0]
    elif SIDEBAR_CELL_INDEX_KEY not in st.session_state or selected_widget_index not in visible_indices:
        selected_widget_index = safe_index if current_visible else visible_indices[0]

    st.session_state[SIDEBAR_CELL_INDEX_KEY] = int(selected_widget_index)
    st.session_state[SIDEBAR_FORCE_SYNC_KEY] = False

    selected_index = st.sidebar.radio(
        "cell_id 목록",
        options=visible_indices,
        key=SIDEBAR_CELL_INDEX_KEY,
        format_func=lambda idx: _build_cell_summary(sorted_df, idx),
        on_change=_mark_sidebar_selection_changed,
    )

    user_changed = bool(st.session_state.get(SIDEBAR_SELECTION_CHANGED_KEY, False))
    st.session_state[SIDEBAR_SELECTION_CHANGED_KEY] = False
    if user_changed:
        return int(selected_index)
    return int(safe_index)


def _build_cell_summary(df: pd.DataFrame, index: int) -> str:
    """Build display text with cell_id + 4 defect values (no column names)."""
    row = df.iloc[index]
    values = [str(row[col]).strip() for col in DEFECT_COLUMNS]
    normalized = [value if value else "-" for value in values]
    return f"{row[COL_CELL_ID]} | {' / '.join(normalized)}"


def _render_sidebar_filters(df: pd.DataFrame) -> dict[str, str]:
    """Render sidebar selectbox filters for visible cell list only."""
    all_option = "전체"
    st.sidebar.subheader("Cell Filter")

    position_options = [all_option, *POSITION_COLUMNS]
    _ensure_selectbox_value(SIDEBAR_FILTER_POSITION_KEY, position_options, all_option)
    selected_position = st.sidebar.selectbox(
        "position",
        options=position_options,
        key=SIDEBAR_FILTER_POSITION_KEY,
    )

    top_options = [all_option, *_collect_top_defect_options(df, selected_position)]
    _ensure_selectbox_value(SIDEBAR_FILTER_TOP_KEY, top_options, all_option)
    selected_top = st.sidebar.selectbox(
        "top defect",
        options=top_options,
        key=SIDEBAR_FILTER_TOP_KEY,
    )

    sub_options = [all_option, *_collect_sub_defect_options(df, selected_position)]
    _ensure_selectbox_value(SIDEBAR_FILTER_SUB_KEY, sub_options, all_option)
    selected_sub = st.sidebar.selectbox(
        "sub defect",
        options=sub_options,
        key=SIDEBAR_FILTER_SUB_KEY,
    )

    return {
        "position": selected_position,
        "top_defect": selected_top,
        "sub_defect": selected_sub,
    }


def _ensure_selectbox_value(key: str, options: list[str], fallback: str) -> None:
    """Keep selectbox state valid when dependent option lists change."""
    current_value = st.session_state.get(key)
    if current_value not in options:
        st.session_state[key] = fallback


def _collect_top_defect_options(df: pd.DataFrame, position: str) -> list[str]:
    """Collect top-defect options from ATIS columns for current position filter."""
    positions = POSITION_COLUMNS if position == "전체" else [position]
    values: set[str] = set()
    for current_position in positions:
        column = POSITION_TO_ATIS_COLUMN[current_position]
        if column not in df.columns:
            continue
        series = df[column].fillna("").astype(str).map(_normalize_top_defect)
        values.update(
            value for value in series.tolist()
            if value and value.upper() != "OK"
        )
    return sorted(values)


def _collect_sub_defect_options(df: pd.DataFrame, position: str) -> list[str]:
    """Collect sub-defect options from Defect columns for current position filter."""
    positions = POSITION_COLUMNS if position == "전체" else [position]
    values: set[str] = set()
    for current_position in positions:
        column = POSITION_TO_DEFECT_COLUMN[current_position]
        if column not in df.columns:
            continue
        series = df[column].fillna("").astype(str).str.strip()
        values.update(
            value for value in series.tolist()
            if value and value.upper() != "OK"
        )
    return sorted(values)


def _filter_sidebar_indices(df: pd.DataFrame, filters: dict[str, str]) -> list[int]:
    """Filter visible sidebar rows only, without changing underlying selection model."""
    position = filters["position"]
    top_defect = filters["top_defect"]
    sub_defect = filters["sub_defect"]

    if position == "전체":
        positions = POSITION_COLUMNS
    else:
        positions = [position]

    matched_indices: list[int] = []
    for index, row in df.iterrows():
        if _row_matches_filters(row=row, positions=positions, top_defect=top_defect, sub_defect=sub_defect):
            matched_indices.append(int(index))
    return matched_indices


def _row_matches_filters(
    *,
    row: pd.Series,
    positions: list[str],
    top_defect: str,
    sub_defect: str,
) -> bool:
    """Apply AND logic across filters, with OR across positions when needed."""
    for position in positions:
        top_value = _normalize_top_defect(str(row.get(POSITION_TO_ATIS_COLUMN[position], "")).strip())
        sub_value = str(row.get(POSITION_TO_DEFECT_COLUMN[position], "")).strip()

        top_ok = top_defect == "전체" or top_value == top_defect
        sub_ok = sub_defect == "전체" or sub_value == sub_defect
        if top_ok and sub_ok:
            return True
    return False


def _normalize_top_defect(value: str) -> str:
    """Normalize ATIS text to the effective top-defect value."""
    text = value.strip()
    if not text or text.lower() == "nan":
        return ""
    if "/" in text:
        _, latest = text.split("/", 1)
        return latest.strip()
    return text


def _mark_sidebar_selection_changed() -> None:
    """Track explicit sidebar radio interaction separately from filter reruns."""
    st.session_state[SIDEBAR_SELECTION_CHANGED_KEY] = True
