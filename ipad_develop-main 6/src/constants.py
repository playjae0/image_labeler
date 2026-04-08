"""Shared constants for dataframe schema, session keys, and page names."""

from __future__ import annotations

# Dataframe columns (must remain stable)
COL_CELL_ID = "cell_id"

COL_CA_TOP = "CA(TOP)"
COL_CA_BOT = "CA(BOT)"
COL_AN_TOP = "AN(TOP)"
COL_AN_BOT = "AN(BOT)"
POSITION_COLUMNS = [COL_CA_TOP, COL_CA_BOT, COL_AN_TOP, COL_AN_BOT]

COL_DEFECT_CA_TOP = "Defect_CA(TOP)"
COL_DEFECT_CA_BOT = "Defect_CA(BOT)"
COL_DEFECT_AN_TOP = "Defect_AN(TOP)"
COL_DEFECT_AN_BOT = "Defect_AN(BOT)"
DEFECT_COLUMNS = [
    COL_DEFECT_CA_TOP,
    COL_DEFECT_CA_BOT,
    COL_DEFECT_AN_TOP,
    COL_DEFECT_AN_BOT,
]

BASE_DF_COLUMNS = [COL_CELL_ID, *POSITION_COLUMNS, *DEFECT_COLUMNS]

# Session state keys
KEY_MASTER_DF = "master_df"
KEY_IMAGE_MAP = "image_map"
KEY_CURRENT_CELL_INDEX = "current_cell_index"
KEY_SELECTED_CELL_ID = "selected_cell_id"
KEY_UPLOAD_COMPLETED = "upload_completed"
KEY_LABEL_SYNC_TOKEN = "label_sync_token"
KEY_SELECTED_FOLDER_INFO = "selected_folder_info"
KEY_UPLOAD_SOURCE_TYPE = "upload_source_type"
KEY_SELECTED_IMAGE_SUBPATH = "selected_image_subpath"
KEY_IMAGE_LOADING_MODE = "image_loading_mode"
KEY_RESOLVED_LOADING_STRATEGY = "resolved_loading_strategy"
KEY_EAGER_THRESHOLD = "eager_threshold"
KEY_PRELOAD_FORWARD_COUNT = "preload_forward_count"
KEY_PRELOAD_BACKWARD_COUNT = "preload_backward_count"

SESSION_DEFAULTS = {
    KEY_MASTER_DF: None,
    KEY_IMAGE_MAP: {},
    KEY_CURRENT_CELL_INDEX: 0,
    KEY_SELECTED_CELL_ID: None,
    KEY_UPLOAD_COMPLETED: False,
    KEY_LABEL_SYNC_TOKEN: 0,
    KEY_SELECTED_FOLDER_INFO: "",
    KEY_UPLOAD_SOURCE_TYPE: "drag_upload",
    KEY_SELECTED_IMAGE_SUBPATH: None,
    KEY_IMAGE_LOADING_MODE: "auto",
    KEY_RESOLVED_LOADING_STRATEGY: "auto",
    KEY_EAGER_THRESHOLD: 200,
    KEY_PRELOAD_FORWARD_COUNT: 2,
    KEY_PRELOAD_BACKWARD_COUNT: 1,
}

# Page names (v1, two-page structure)
PAGE_UPLOAD = "upload"
PAGE_LABELING = "labeling"
PAGE_DEFECT_STATS = "defect_statistics"
PAGE_ADMIN = "admin"
PAGES = [PAGE_UPLOAD, PAGE_LABELING, PAGE_DEFECT_STATS, PAGE_ADMIN]
