"""Signup page for authentication."""

from __future__ import annotations

import sqlite3

import streamlit as st

from src.auth.password_utils import hash_password
from src.auth.user_store import create_user, find_user_by_email, find_user_by_employee_id


def render_signup_page(db_path: str) -> None:
    """Render signup page and create pending users."""
    st.title("회원가입")

    with st.form("signup_form"):
        employee_id = st.text_input("사번")
        full_name = st.text_input("성명")
        email = st.text_input("email")
        password = st.text_input("password", type="password")
        password_confirm = st.text_input("password_confirm", type="password")
        submitted = st.form_submit_button("가입 요청")

    if not submitted:
        return

    employee_id = employee_id.strip()
    full_name = full_name.strip()
    email = email.strip()

    if password != password_confirm:
        st.error("비밀번호 확인이 일치하지 않습니다.")
        return

    if find_user_by_employee_id(db_path, employee_id) is not None:
        st.error("이미 사용 중인 사번입니다.")
        return

    if find_user_by_email(db_path, email) is not None:
        st.error("이미 사용 중인 이메일입니다.")
        return

    try:
        create_user(
            db_path,
            employee_id=employee_id,
            full_name=full_name,
            email=email,
            password_hash=hash_password(password),
            status="pending",
            is_admin=False,
            is_active=True,
        )
    except sqlite3.IntegrityError:
        st.error("중복된 계정 정보가 있어 가입할 수 없습니다.")
        return

    st.success("가입 요청이 완료되었습니다. 관리자 승인을 기다려주세요.")
