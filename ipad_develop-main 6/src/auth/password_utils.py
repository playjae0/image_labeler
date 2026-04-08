"""Password hashing and verification utilities."""

from __future__ import annotations

import hashlib
import hmac
import os

PBKDF2_ITERATIONS = 120_000
SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Hash plaintext password using PBKDF2-HMAC-SHA256.

    Stored format: pbkdf2_sha256$<iterations>$<salt_hex>$<digest_hex>
    """
    if not password:
        raise ValueError("Password must not be empty.")

    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify plaintext password against PBKDF2 hash string."""
    if not password or not password_hash:
        return False

    try:
        algorithm, iterations_text, salt_hex, digest_hex = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)
