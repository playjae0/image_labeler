"""Authentication manager for login verification and status checks."""

from __future__ import annotations

from pathlib import Path

from src.auth.password_utils import verify_password
from src.auth.user_store import find_user_by_employee_id, record_login_history


def verify_login(
    *,
    db_path: str | Path,
    employee_id: str,
    password: str,
) -> tuple[bool, str]:
    """Verify login credentials and account status.

    Returns:
        (success, message)
    """
    user = find_user_by_employee_id(db_path, employee_id)
    if user is None:
        return False, "사번 또는 비밀번호가 올바르지 않습니다."

    if not bool(user.get("is_active", 0)):
        return False, "비활성화된 계정입니다. 관리자에게 문의하세요."

    status = str(user.get("status", "")).lower()
    if status == "pending":
        return False, "관리자 승인 대기 중입니다."
    if status == "rejected":
        return False, "가입이 반려되었습니다. 관리자에게 문의하세요."
    if status != "approved":
        return False, "사번 또는 비밀번호가 올바르지 않습니다."

    if not verify_password(password, str(user.get("password_hash", ""))):
        return False, "사번 또는 비밀번호가 올바르지 않습니다."

    # Spec note: only successful logins are stored.
    record_login_history(db_path, employee_id, success=True)
    return True, "로그인 성공"
