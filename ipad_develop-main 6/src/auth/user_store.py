"""SQLite-backed user storage for authentication."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Return sqlite connection with row_factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_user(
    db_path: str | Path,
    *,
    employee_id: str,
    full_name: str,
    email: str,
    password_hash: str,
    status: str = "pending",
    is_admin: bool = False,
    is_active: bool = True,
) -> None:
    """Create a user row with uniqueness on employee_id/email."""
    created_at = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (
                employee_id, full_name, email, password_hash,
                status, is_admin, is_active, created_at, approved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee_id,
                full_name,
                email,
                password_hash,
                status,
                int(is_admin),
                int(is_active),
                created_at,
                None,
            ),
        )
        conn.commit()


def find_user_by_employee_id(db_path: str | Path, employee_id: str) -> dict[str, Any] | None:
    """Find one user by employee_id."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE employee_id = ? LIMIT 1",
            (employee_id,),
        ).fetchone()
    return dict(row) if row else None


def find_user_by_email(db_path: str | Path, email: str) -> dict[str, Any] | None:
    """Find one user by email."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? LIMIT 1",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def update_user_status(db_path: str | Path, employee_id: str, status: str) -> None:
    """Update user approval status and approved_at when approved."""
    approved_at = datetime.utcnow().isoformat() if status == "approved" else None
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE users SET status = ?, approved_at = ? WHERE employee_id = ?",
            (status, approved_at, employee_id),
        )
        conn.commit()


def deactivate_user(db_path: str | Path, employee_id: str) -> None:
    """Soft-delete user by setting is_active to False."""
    with get_connection(db_path) as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE employee_id = ?", (employee_id,))
        conn.commit()


def list_users(db_path: str | Path) -> list[dict[str, Any]]:
    """List all users ordered by creation time DESC."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def record_login_history(db_path: str | Path, employee_id: str, success: bool) -> None:
    """Insert login history row (intended for successful login logs)."""
    login_at = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO login_history (employee_id, login_at, success) VALUES (?, ?, ?)",
            (employee_id, login_at, int(success)),
        )
        conn.commit()


def reset_user_password(db_path: str | Path, employee_id: str, password_hash: str) -> None:
    """Reset user password hash."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE employee_id = ?",
            (password_hash, employee_id),
        )
        conn.commit()


def list_successful_login_history(db_path: str | Path) -> list[dict[str, Any]]:
    """List successful login history rows."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT employee_id, login_at FROM login_history WHERE success = 1 ORDER BY login_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def log_dataset_access(db_path: str | Path, employee_id: str, folder_name: str) -> None:
    """Record dataset folder access history for auditing."""
    access_time = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO dataset_access_history (employee_id, folder_name, access_time)
            VALUES (?, ?, ?)
            """,
            (employee_id, folder_name, access_time),
        )
        conn.commit()


def list_dataset_access_history(db_path: str | Path) -> list[dict[str, Any]]:
    """List dataset access history rows ordered by latest access."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT employee_id, folder_name, access_time
            FROM dataset_access_history
            ORDER BY access_time DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]
