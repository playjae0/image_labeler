"""Defect controls for each position image block."""

from __future__ import annotations

import streamlit as st

from config import ATIS_SUB_LABELS, ATIS_TOP_LEVELS


def render_defect_selector(
    *,
    current_top: str,
    current_sub: str,
    widget_key_prefix: str,
) -> tuple[str, str, bool, bool]:
    """Render top/sub defect selector and return (top, sub, changed, interacted)."""
    top_options = list(ATIS_TOP_LEVELS)
    default_top_index = top_options.index(current_top) if current_top in top_options else 0
    selected_top = st.radio(
        "상위 분류",
        options=top_options,
        index=default_top_index,
        key=f"{widget_key_prefix}_top",
        horizontal=True,
    )

    next_top = current_top
    next_sub = current_sub
    changed = False
    interacted = False

    if selected_top != current_top:
        next_top = selected_top
        next_sub = ""
        changed = True

    sub_options = list(ATIS_SUB_LABELS.get(selected_top, []))

    if not sub_options:
        st.caption("선택 가능한 하위 불량이 없습니다.")
        next_sub = "" if selected_top != current_top else next_sub
    else:
        button_labels = list(sub_options)
        button_values = list(sub_options)

        for start in range(0, len(button_labels), 4):
            cols = st.columns(4)
            for idx, label in enumerate(button_labels[start : start + 4]):
                value = button_values[start + idx]
                key = f"{widget_key_prefix}_sub_{value or 'empty'}"
                is_active = next_sub == value
                button_type = "primary" if is_active else "secondary"
                if cols[idx].button(label, key=key, use_container_width=True, type=button_type):
                    if value != current_sub or selected_top != current_top:
                        next_top = selected_top
                        next_sub = value
                        changed = True
                        interacted = True

        st.caption(f"현재 하위 라벨: {next_sub if next_sub else '-'}")

    return next_top, next_sub, changed, interacted
