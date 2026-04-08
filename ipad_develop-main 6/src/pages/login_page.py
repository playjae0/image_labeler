"""Login page for authentication."""

from __future__ import annotations

import streamlit as st

from src.auth.auth_manager import verify_login


def render_login_page(db_path: str) -> None:
    """Render login page and update auth session state on success."""
    st.title("로그인")

    with st.form("login_form"):
        employee_id = st.text_input("사번")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")

    if not submitted:
        return

    success, message = verify_login(
        db_path=db_path,
        employee_id=employee_id.strip(),
        password=password,
    )
    if not success:
        st.error(message)
        return

    st.session_state["auth_logged_in"] = True
    st.session_state["auth_employee_id"] = employee_id.strip()
    st.session_state["auth_page"] = "upload"
    st.success("로그인 성공")
    st.rerun()
