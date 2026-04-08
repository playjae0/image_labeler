"""Application configuration for the Streamlit defect labeling platform (Step 1)."""

from __future__ import annotations

from pathlib import Path

# Upload settings
ALLOWED_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png")
MAX_UPLOAD_COUNT: int = 4000

# Label settings
DEFAULT_DEFECT_LIST: list[str] = [
    "Crack",
    "Damage",
    "Scrap",
    "Dust",
    "Contamination",
    "Wrinkle",
    "Scratch",
    "Etc",
]
MAX_DEFECT_COUNT: int = 16

# ATIS-driven defect hierarchy settings
#
# - Top level is selected first and can override ATIS source value.
# - Defect_* columns only store sub labels (not top-level text).
ATIS_TOP_LEVELS: list[str] = ["OK", "Damage", "Crack", "Scrap"]
ATIS_SUB_LABELS: dict[str, list[str]] = {
    "OK": ["OK"],
    "Damage": ["Damage", "Dust", "Contamination", "Wrinkle", "Etc"],
    "Crack": ["Crack", "Scratch", "Etc"],
    "Scrap": ["Scrap", "Etc"],
}

# Position settings (must stay fixed)
POSITIONS: list[str] = [
    "CA(TOP)",
    "CA(BOT)",
    "AN(TOP)",
    "AN(BOT)",
]

# Data source / output paths
IMAGE_ROOT_PATH: Path = Path("./data/images")
ATIS_FILE_PATH: Path = Path("./data/atis/atis.xlsx")
DEFAULT_SAVE_ROOT: Path = Path("./outputs")

# Path aliases/extensions for upload/export controls
IMAGE_ROOT_DIR: Path = IMAGE_ROOT_PATH
CSV_OUTPUT_ROOT_DIR: Path = Path("./output/result")
CSV_BACKUP_ROOT_DIR: Path = Path("./output/backup")
IMAGE_EXPORT_ROOT_DIR: Path = DEFAULT_SAVE_ROOT

# ATIS mapping placeholder for future merge rules
# key: target output column name, value: source ATIS column name
# Example:
# ATIS_COLUMN_MAPPING = {
#     "cell_id": "Cell ID",
#     "ATIS_CA(TOP)": "ENG Top Cathode",
#     "ATIS_CA(BOT)": "ENG Bottom Cathode",
#     "ATIS_AN(TOP)": "ENG Top Anode",
#     "ATIS_AN(BOT)": "ENG Bottom Anode",
# }
ATIS_COLUMN_MAPPING: dict[str, str] = {}

# Timestamp formatting
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"

# Authentication settings
AUTH_DB_PATH: Path = Path("./data/auth/auth.db")
ADMIN_EMPLOYEE_ID: str = "ADMIN000000000001"
ADMIN_FULL_NAME: str = "System Admin"
ADMIN_EMAIL: str = "admin@example.com"
ADMIN_PASSWORD: str = "ChangeMe!123"

# Image loading strategy defaults (Step 1)
IMAGE_LOADING_MODE_DEFAULT: str = "auto"
EAGER_THRESHOLD_DEFAULT: int = 200
PRELOAD_FORWARD_COUNT_DEFAULT: int = 2
PRELOAD_BACKWARD_COUNT_DEFAULT: int = 1
