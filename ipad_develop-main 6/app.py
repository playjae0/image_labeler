"""Streamlit app entrypoint with authentication-gated routing."""

from __future__ import annotations

import streamlit as st

st.set_page_config(layout="wide", page_title="Defect Labeling")

from config import (
    ADMIN_EMAIL,
    ADMIN_EMPLOYEE_ID,
    ADMIN_FULL_NAME,
    ADMIN_PASSWORD,
    AUTH_DB_PATH,
)
from src.auth.user_store import find_user_by_employee_id
from src.auth.db_init import initialize_auth_db
from src.constants import PAGE_ADMIN, PAGE_DEFECT_STATS, PAGE_LABELING, PAGE_UPLOAD
from src.pages.admin_page import render_admin_page
from src.pages.defect_statistics_page import render_defect_statistics_page
from src.pages.labeling_page import render_labeling_page
from src.pages.login_page import render_login_page
from src.pages.signup_page import render_signup_page
from src.pages.upload_page import render_upload_page
from src.state_manager import initialize_session_state


AUTH_PAGE_LOGIN = "login"
AUTH_PAGE_SIGNUP = "signup"


def _initialize_auth_state() -> None:
    if "auth_logged_in" not in st.session_state:
        st.session_state["auth_logged_in"] = False
    if "auth_employee_id" not in st.session_state:
        st.session_state["auth_employee_id"] = ""
    if "auth_page" not in st.session_state:
        st.session_state["auth_page"] = AUTH_PAGE_LOGIN


def _render_unauthenticated_router() -> None:
    st.title("Defect Labeling")
    st.caption("로그인 후 업로드/라벨링 페이지를 사용할 수 있습니다.")

    auth_page = st.radio(
        "인증 페이지",
        options=[AUTH_PAGE_LOGIN, AUTH_PAGE_SIGNUP],
        format_func=lambda value: "Login" if value == AUTH_PAGE_LOGIN else "Signup",
        horizontal=True,
        index=0 if st.session_state.get("auth_page") != AUTH_PAGE_SIGNUP else 1,
        key="auth_page_selector",
    )
    st.session_state["auth_page"] = auth_page

    with st.sidebar:
        st.header("Authentication")
        st.caption("메인 화면에서 Login/Signup 탭을 선택하세요.")

    if auth_page == AUTH_PAGE_SIGNUP:
        render_signup_page(str(AUTH_DB_PATH))
    else:
        render_login_page(str(AUTH_DB_PATH))


def _render_authenticated_router() -> None:
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = PAGE_UPLOAD

    employee_id = st.session_state.get("auth_employee_id", "")
    user = find_user_by_employee_id(str(AUTH_DB_PATH), str(employee_id)) if employee_id else None
    is_admin = bool(user and user.get("is_admin"))

    with st.sidebar:
        st.header("Navigation")
        st.button("Upload", on_click=lambda: st.session_state.__setitem__("current_page", PAGE_UPLOAD))
        st.button("Labeling", on_click=lambda: st.session_state.__setitem__("current_page", PAGE_LABELING))
        st.button("Defect Stats", on_click=lambda: st.session_state.__setitem__("current_page", PAGE_DEFECT_STATS))
        if is_admin:
            st.button("Admin", on_click=lambda: st.session_state.__setitem__("current_page", PAGE_ADMIN))
        st.divider()
        st.caption(f"로그인: {st.session_state.get('auth_employee_id', '')}")
        if st.button("Logout"):
            st.session_state["auth_logged_in"] = False
            st.session_state["auth_employee_id"] = ""
            st.session_state["auth_page"] = AUTH_PAGE_LOGIN
            st.rerun()

    if st.session_state["current_page"] == PAGE_UPLOAD:
        render_upload_page()
    elif st.session_state["current_page"] == PAGE_DEFECT_STATS:
        render_defect_statistics_page()
    elif st.session_state["current_page"] == PAGE_ADMIN and is_admin:
        render_admin_page()
    else:
        render_labeling_page()


def main() -> None:
    """Application bootstrap and auth-gated page router."""
    initialize_session_state()
    _initialize_auth_state()

    initialize_auth_db(
        db_path=AUTH_DB_PATH,
        admin_employee_id=ADMIN_EMPLOYEE_ID,
        admin_full_name=ADMIN_FULL_NAME,
        admin_email=ADMIN_EMAIL,
        admin_password=ADMIN_PASSWORD,
    )

    if not st.session_state["auth_logged_in"]:
        _render_unauthenticated_router()
        return

    _render_authenticated_router()


if __name__ == "__main__":
    main()
