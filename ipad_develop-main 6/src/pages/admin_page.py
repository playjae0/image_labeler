"""Admin page for access history visibility."""

from __future__ import annotations

import secrets
import sqlite3

import pandas as pd
import streamlit as st

from config import (
    AUTH_DB_PATH,
    EAGER_THRESHOLD_DEFAULT,
    IMAGE_LOADING_MODE_DEFAULT,
    PRELOAD_BACKWARD_COUNT_DEFAULT,
    PRELOAD_FORWARD_COUNT_DEFAULT,
)
from src.auth.password_utils import hash_password
from src.lock.dataset_lock_manager import force_unlock, get_active_locks
from src.auth.user_store import (
    deactivate_user,
    list_dataset_access_history,
    list_successful_login_history,
    list_users,
    reset_user_password,
    update_user_status,
)
from src.logging.activity_logger import get_labeling_activity_logs


def render_admin_page() -> None:
    """Render admin-only history dashboards."""
    st.title("관리자 페이지")

    st.subheader("User Management")
    _render_user_management_section()

    st.subheader("로그인 이력")
    login_rows = list_successful_login_history(AUTH_DB_PATH)
    if login_rows:
        st.dataframe(pd.DataFrame(login_rows), use_container_width=True)
    else:
        st.info("로그인 이력이 없습니다.")

    st.subheader("데이터셋 접근 이력")
    access_rows = list_dataset_access_history(AUTH_DB_PATH)
    if access_rows:
        st.dataframe(pd.DataFrame(access_rows), use_container_width=True)
    else:
        st.info("데이터셋 접근 이력이 없습니다.")

    st.subheader("Labeling Activity Log")
    activity_rows = get_labeling_activity_logs(AUTH_DB_PATH)
    if activity_rows:
        activity_df = pd.DataFrame(activity_rows)[
            ["employee_id", "line", "period", "labeled_cells", "timestamp"]
        ]
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("라벨링 활동 로그가 없습니다.")

    st.subheader("Dataset Locks")
    _render_dataset_lock_section()

    st.subheader("Image Loading Strategy")
    _render_image_loading_strategy_section()


def _render_user_management_section() -> None:
    """Render user table and admin action controls."""
    users = list_users(AUTH_DB_PATH)
    if users:
        user_df = pd.DataFrame(users)[
            ["employee_id", "full_name", "email", "status", "is_active", "created_at"]
        ]
        st.dataframe(user_df, use_container_width=True)
    else:
        st.info("사용자가 없습니다.")

    employee_ids = [str(user.get("employee_id", "")) for user in users]
    if not employee_ids:
        return

    selected_employee = st.selectbox("관리 대상 사용자", options=employee_ids)
    selected_user = next((user for user in users if user.get("employee_id") == selected_employee), None)
    if selected_user is None:
        return

    col_approve, col_reject, col_deactivate, col_reset = st.columns(4)

    with col_approve:
        if st.button("승인", use_container_width=True):
            if str(selected_user.get("status", "")).lower() == "pending":
                update_user_status(AUTH_DB_PATH, selected_employee, "approved")
                st.success("승인 완료")
                st.rerun()
            else:
                st.warning("pending 상태 사용자만 승인할 수 있습니다.")

    with col_reject:
        if st.button("반려", use_container_width=True):
            if str(selected_user.get("status", "")).lower() == "pending":
                update_user_status(AUTH_DB_PATH, selected_employee, "rejected")
                st.success("반려 완료")
                st.rerun()
            else:
                st.warning("pending 상태 사용자만 반려할 수 있습니다.")

    with col_deactivate:
        if st.button("비활성화", use_container_width=True):
            deactivate_user(AUTH_DB_PATH, selected_employee)
            st.success("비활성화 완료")
            st.rerun()

    with col_reset:
        if st.button("임시 비밀번호 발급", use_container_width=True):
            temporary_password = secrets.token_urlsafe(8)
            reset_user_password(AUTH_DB_PATH, selected_employee, hash_password(temporary_password))
            st.success("비밀번호 재설정 완료")
            st.code(f"임시 비밀번호: {temporary_password}")


def _render_dataset_lock_section() -> None:
    """Render active lock monitor and force-unlock control."""
    locks = get_active_locks(AUTH_DB_PATH)
    if locks:
        lock_df = pd.DataFrame(locks)[["dataset_key", "employee_id", "locked_at"]]
        st.dataframe(lock_df, use_container_width=True)

        keys = [str(row.get("dataset_key", "")) for row in locks]
        selected_key = st.selectbox("강제 해제 대상 데이터셋", options=keys)
        if st.button("Force Unlock"):
            force_unlock(AUTH_DB_PATH, selected_key)
            st.success("잠금 강제 해제 완료")
            st.rerun()
    else:
        st.info("활성 잠금이 없습니다.")


def _render_image_loading_strategy_section() -> None:
    """Render and save image loading strategy settings for labeling page."""
    current = _load_image_loading_settings()

    mode = st.selectbox(
        "image_loading_mode",
        options=["auto", "eager", "lazy_cache"],
        index=["auto", "eager", "lazy_cache"].index(str(current["image_loading_mode"])),
        help="auto: 이미지 수 기준 자동 선택 / eager: 전체 선로딩 / lazy_cache: 표시 범위+캐시",
    )
    eager_threshold = st.number_input(
        "eager_threshold",
        min_value=1,
        step=1,
        value=int(current["eager_threshold"]),
    )
    preload_forward_count = st.number_input(
        "preload_forward_count",
        min_value=0,
        step=1,
        value=int(current["preload_forward_count"]),
    )
    preload_backward_count = st.number_input(
        "preload_backward_count",
        min_value=0,
        step=1,
        value=int(current["preload_backward_count"]),
    )

    if st.button("이미지 로딩 전략 저장", use_container_width=True):
        _save_image_loading_settings(
            image_loading_mode=mode,
            eager_threshold=int(eager_threshold),
            preload_forward_count=int(preload_forward_count),
            preload_backward_count=int(preload_backward_count),
        )
        st.success("이미지 로딩 전략 설정을 저장했습니다.")


def _load_image_loading_settings() -> dict[str, int | str]:
    """Load strategy settings from sqlite key-value table."""
    defaults: dict[str, int | str] = {
        "image_loading_mode": IMAGE_LOADING_MODE_DEFAULT,
        "eager_threshold": EAGER_THRESHOLD_DEFAULT,
        "preload_forward_count": PRELOAD_FORWARD_COUNT_DEFAULT,
        "preload_backward_count": PRELOAD_BACKWARD_COUNT_DEFAULT,
    }

    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
            """
        )
        rows = conn.execute(
            "SELECT setting_key, setting_value FROM app_settings WHERE setting_key IN (?, ?, ?, ?)",
            (
                "image_loading_mode",
                "eager_threshold",
                "preload_forward_count",
                "preload_backward_count",
            ),
        ).fetchall()

    for key, value in rows:
        if key in {"eager_threshold", "preload_forward_count", "preload_backward_count"}:
            try:
                defaults[key] = int(value)
            except ValueError:
                continue
        else:
            defaults[key] = value
    return defaults


def _save_image_loading_settings(
    *,
    image_loading_mode: str,
    eager_threshold: int,
    preload_forward_count: int,
    preload_backward_count: int,
) -> None:
    """Persist image loading strategy settings to sqlite key-value table."""
    with sqlite3.connect(str(AUTH_DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT OR REPLACE INTO app_settings (setting_key, setting_value) VALUES (?, ?)",
            [
                ("image_loading_mode", image_loading_mode),
                ("eager_threshold", str(max(1, eager_threshold))),
                ("preload_forward_count", str(max(0, preload_forward_count))),
                ("preload_backward_count", str(max(0, preload_backward_count))),
            ],
        )
        conn.commit()
