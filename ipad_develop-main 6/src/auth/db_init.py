"""Database initialization for authentication tables and admin bootstrap."""

from __future__ import annotations

from pathlib import Path

from src.auth.password_utils import hash_password
from src.auth.user_store import create_user, find_user_by_employee_id, get_connection


def initialize_auth_db(
    *,
    db_path: str | Path,
    admin_employee_id: str,
    admin_full_name: str,
    admin_email: str,
    admin_password: str,
) -> None:
    """Create tables (if missing) and ensure initial admin account exists."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                employee_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                approved_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS login_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                login_at TEXT NOT NULL,
                success INTEGER NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES users(employee_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_access_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                folder_name TEXT NOT NULL,
                access_time TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES users(employee_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS labeling_activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                line TEXT NOT NULL,
                period TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                labeled_cells INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES users(employee_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_lock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_key TEXT NOT NULL UNIQUE,
                employee_id TEXT NOT NULL,
                locked_at TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES users(employee_id)
            )
            """
        )
        conn.commit()

    admin_exists = find_user_by_employee_id(db_file, admin_employee_id)
    if admin_exists is None:
        create_user(
            db_file,
            employee_id=admin_employee_id,
            full_name=admin_full_name,
            email=admin_email,
            password_hash=hash_password(admin_password),
            status="approved",
            is_admin=True,
            is_active=True,
        )
