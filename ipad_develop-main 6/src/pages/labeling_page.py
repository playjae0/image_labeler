"""Labeling page implementation for Step 5."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import sqlite3

import pandas as pd
import streamlit as st

from config import CSV_BACKUP_ROOT_DIR, CSV_OUTPUT_ROOT_DIR, IMAGE_EXPORT_ROOT_DIR
from config import (
    AUTH_DB_PATH,
    EAGER_THRESHOLD_DEFAULT,
    IMAGE_LOADING_MODE_DEFAULT,
    PRELOAD_BACKWARD_COUNT_DEFAULT,
    PRELOAD_FORWARD_COUNT_DEFAULT,
)
from src.image_registry import load_image_bytes
from src.lock.dataset_lock_manager import acquire_lock, release_lock
from src.logging.activity_logger import log_labeling_activity
from src.constants import (
    COL_CELL_ID,
    DEFECT_COLUMNS,
    POSITION_COLUMNS,
    KEY_SELECTED_FOLDER_INFO,
    KEY_SELECTED_IMAGE_SUBPATH,
    KEY_UPLOAD_SOURCE_TYPE,
    PAGE_UPLOAD,
)
from src.save_manager import save_defect_images
from src.save_manager import (
    apply_loaded_defect_values,
    build_next_version,
    build_next_version_filename,
    create_csv_backup_copy,
    ensure_result_folder_from_selected_subpath,
    export_csv_without_filling_ok,
    find_latest_csv_file,
    find_latest_csv_version,
    load_previous_defect_values,
    parse_version_from_filename,
)
from src.state_manager import (
    get_current_cell_index,
    get_image_loading_settings,
    get_image_map,
    get_master_dataframe,
    get_resolved_loading_strategy,
    is_upload_completed,
    set_current_cell_index,
    set_image_loading_settings,
    set_master_dataframe,
    set_resolved_loading_strategy,
    touch_label_sync_token,
)
from src.ui.image_grid import render_image_grid
from src.ui.sidebar_list import render_sidebar_cell_list
from src.ui.status_panel import render_status_panel
from utils.naming_utils import sanitize_token
from utils.path_utils import ensure_directory, list_subdirectories_relative


EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Za-z]{2}\d{5}$")
SIDEBAR_FORCE_SYNC_KEY = "sidebar_force_sync"
SIDEBAR_CELL_INDEX_KEY = "sidebar_cell_index"
DEFECT_SUMMARY_DATA_KEY = "labeling_defect_summary_data"
DEFECT_SUMMARY_SOURCE_KEY = "labeling_defect_summary_source"
CSV_IMPORT_PREVIEW_KEY = "labeling_csv_import_preview"
CSV_IMPORT_SOURCE_KEY = "labeling_csv_import_source"


def render_labeling_page() -> None:
    """Render labeling page with sidebar list, image grid, and controls."""
    st.title("라벨링 페이지")

    if not is_upload_completed():
        st.warning("업로드가 완료되지 않았습니다. 업로드 페이지에서 먼저 데이터를 준비해주세요.")
        if st.button("업로드 페이지로 이동"):
            st.session_state["current_page"] = PAGE_UPLOAD
            st.rerun()
        return

    master_df = get_master_dataframe()
    image_map = get_image_map()

    if master_df is None or master_df.empty:
        st.warning("라벨링할 데이터가 없습니다. 업로드 페이지에서 데이터를 생성해주세요.")
        if st.button("업로드 페이지로 이동"):
            st.session_state["current_page"] = PAGE_UPLOAD
            st.rerun()
        return

    _resolve_and_store_image_loading_strategy(master_df=master_df, image_map=image_map)

    if not _ensure_dataset_lock():
        return

    if not _render_auto_previous_csv_prompt():
        return

    _render_sidebar_source_info()
    _render_sidebar_previous_csv_loader()

    sorted_df = master_df.sort_values(COL_CELL_ID).reset_index(drop=True)
    current_index = _safe_index(get_current_cell_index(), len(sorted_df))

    selected_index = render_sidebar_cell_list(sorted_df, current_index)
    if selected_index != current_index:
        set_current_cell_index(selected_index)
        current_index = selected_index

    render_status_panel(sorted_df, current_index)
    _render_cell_progress_summary(sorted_df)
    _render_navigation_buttons(current_index, len(sorted_df))

    runtime_image_map = _build_runtime_image_map(
        sorted_df=sorted_df,
        image_map=image_map,
        current_index=current_index,
    )

    changed, interacted = render_image_grid(df=sorted_df, image_map=runtime_image_map, row_index=current_index)
    if changed:
        set_master_dataframe(sorted_df)
        touch_label_sync_token()
        set_current_cell_index(current_index)
        st.session_state[SIDEBAR_FORCE_SYNC_KEY] = True
        st.rerun()

    _render_csv_import_section()
    _render_save_section(sorted_df, image_map)


def _resolve_and_store_image_loading_strategy(*, master_df: pd.DataFrame, image_map: dict[str, dict[str, object]]) -> None:
    """Resolve image loading strategy (Step 1) and store settings/state for Step 2 hooks."""
    settings = _load_image_loading_settings_from_db()
    set_image_loading_settings(
        image_loading_mode=str(settings["image_loading_mode"]),
        eager_threshold=int(settings["eager_threshold"]),
        preload_forward_count=int(settings["preload_forward_count"]),
        preload_backward_count=int(settings["preload_backward_count"]),
    )

    image_count = _count_dataset_images(master_df=master_df, image_map=image_map)
    resolved = _resolve_strategy_from_settings(image_count=image_count, settings=settings)
    set_resolved_loading_strategy(resolved)

    session_settings = get_image_loading_settings()
    st.caption(
        "Image loading strategy "
        f"(mode={session_settings['image_loading_mode']}, resolved={get_resolved_loading_strategy()}, "
        f"images={image_count}, eager_threshold={session_settings['eager_threshold']}, "
        f"forward={session_settings['preload_forward_count']}, backward={session_settings['preload_backward_count']})"
    )


def _load_image_loading_settings_from_db() -> dict[str, int | str]:
    """Load strategy settings from sqlite key-value table with safe defaults."""
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
        elif key == "image_loading_mode" and value in {"auto", "eager", "lazy_cache"}:
            defaults[key] = value

    return defaults


def _count_dataset_images(*, master_df: pd.DataFrame, image_map: dict[str, dict[str, object]]) -> int:
    """Count dataset image size for strategy resolution without changing existing loading flow."""
    if image_map:
        return int(sum(len(position_map) for position_map in image_map.values()))

    image_columns = [column for column in POSITION_COLUMNS if column in master_df.columns]
    if not image_columns:
        return 0

    numeric_images = master_df[image_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return int(numeric_images.sum().sum())


def _resolve_strategy_from_settings(*, image_count: int, settings: dict[str, int | str]) -> str:
    """Resolve final strategy: eager/lazy_cache based on mode + threshold."""
    mode = str(settings.get("image_loading_mode", IMAGE_LOADING_MODE_DEFAULT))
    if mode == "eager":
        return "eager"
    if mode == "lazy_cache":
        return "lazy_cache"

    eager_threshold = int(settings.get("eager_threshold", EAGER_THRESHOLD_DEFAULT))
    return "eager" if image_count <= eager_threshold else "lazy_cache"


def _safe_index(index: int, length: int) -> int:
    """Clamp index into dataframe bounds."""
    if length <= 0:
        return 0
    return min(max(index, 0), length - 1)


def _build_runtime_image_map(
    *,
    sorted_df: pd.DataFrame,
    image_map: dict[str, dict[str, object]],
    current_index: int,
) -> dict[str, dict[str, object]]:
    """Build render-time image map using eager/lazy image-data loading."""
    cache = st.session_state.setdefault("image_data_cache", {})
    dataset_key = _get_dataset_cache_key(sorted_df)

    if st.session_state.get("image_data_cache_dataset_key") != dataset_key:
        cache.clear()
        st.session_state["image_data_cache_dataset_key"] = dataset_key

    cell_ids = [str(value) for value in sorted_df[COL_CELL_ID].tolist()]
    if not cell_ids:
        return image_map

    settings = get_image_loading_settings()
    resolved = get_resolved_loading_strategy()

    if resolved == "eager":
        for cell_id in cell_ids:
            _load_visible_images(image_map=image_map, cache=cache, cell_id=cell_id)
    else:
        _load_visible_images(image_map=image_map, cache=cache, cell_id=cell_ids[current_index])
        _preload_neighbor_images(
            cell_ids=cell_ids,
            image_map=image_map,
            cache=cache,
            current_index=current_index,
            preload_forward_count=int(settings["preload_forward_count"]),
            preload_backward_count=int(settings["preload_backward_count"]),
        )
        _evict_old_cached_images(
            cache=cache,
            cell_ids=cell_ids,
            current_index=current_index,
            preload_forward_count=int(settings["preload_forward_count"]),
            preload_backward_count=int(settings["preload_backward_count"]),
        )

    runtime_image_map: dict[str, dict[str, object]] = {}
    for cell_id, position_map in image_map.items():
        runtime_image_map[cell_id] = {}
        for position, image_ref in position_map.items():
            runtime_image_map[cell_id][position] = _get_cached_image(
                cache=cache,
                cell_id=cell_id,
                position=position,
                fallback=image_ref,
            )

    return runtime_image_map


def _get_dataset_cache_key(sorted_df: pd.DataFrame) -> str:
    """Build a dataset identity key from ordered cell ids."""
    return "|".join(str(value) for value in sorted_df[COL_CELL_ID].tolist())


def _load_visible_images(
    *,
    image_map: dict[str, dict[str, object]],
    cache: dict[tuple[str, str], object],
    cell_id: str,
) -> None:
    """Load current cell images into cache."""
    for position, image_ref in image_map.get(cell_id, {}).items():
        key = (cell_id, position)
        if key not in cache:
            try:
                cache[key] = load_image_bytes(image_ref)
            except Exception:
                cache[key] = image_ref


def _preload_neighbor_images(
    *,
    cell_ids: list[str],
    image_map: dict[str, dict[str, object]],
    cache: dict[tuple[str, str], object],
    current_index: int,
    preload_forward_count: int,
    preload_backward_count: int,
) -> None:
    """Preload images for neighboring cells within rolling window."""
    start = max(0, current_index - preload_backward_count)
    end = min(len(cell_ids) - 1, current_index + preload_forward_count)
    for idx in range(start, end + 1):
        _load_visible_images(image_map=image_map, cache=cache, cell_id=cell_ids[idx])


def _get_cached_image(
    *,
    cache: dict[tuple[str, str], object],
    cell_id: str,
    position: str,
    fallback: object,
) -> object:
    """Return cached image payload when available."""
    return cache.get((cell_id, position), fallback)


def _evict_old_cached_images(
    *,
    cache: dict[tuple[str, str], object],
    cell_ids: list[str],
    current_index: int,
    preload_forward_count: int,
    preload_backward_count: int,
) -> None:
    """Evict image cache entries outside rolling window."""
    start = max(0, current_index - preload_backward_count)
    end = min(len(cell_ids) - 1, current_index + preload_forward_count)
    keep_cell_ids = set(cell_ids[start : end + 1])

    stale_keys = [key for key in cache if key[0] not in keep_cell_ids]
    for key in stale_keys:
        cache.pop(key, None)


def _render_sidebar_source_info() -> None:
    """Show selected source folder information in sidebar when available."""
    folder_info = str(st.session_state.get(KEY_SELECTED_FOLDER_INFO, "") or "").strip()
    if folder_info:
        st.sidebar.caption(f"선택 폴더(상위/하위): {folder_info}")


def _render_sidebar_previous_csv_loader() -> None:
    """Render manual previous CSV load flow for folder-select mode."""
    upload_source_type = st.session_state.get(KEY_UPLOAD_SOURCE_TYPE, "drag_upload")
    selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)
    if upload_source_type != "folder_select" or not isinstance(selected_subpath, str) or not selected_subpath.strip():
        return

    result_dir = ensure_result_folder_from_selected_subpath(CSV_OUTPUT_ROOT_DIR, selected_subpath)
    csv_files = _list_versioned_csv_files(result_dir)

    st.sidebar.divider()
    st.sidebar.caption("이전 CSV 불러오기")
    if not csv_files:
        st.sidebar.caption("불러올 CSV가 없습니다.")
        return

    selected_file_name = st.sidebar.selectbox(
        "불러올 CSV 파일",
        options=[file_path.name for file_path in csv_files],
        index=0,
    )
    selected_csv = next((path for path in csv_files if path.name == selected_file_name), None)
    if selected_csv is None:
        return

    if st.sidebar.button("이전값 불러오기"):
        st.session_state["confirm_load_previous_csv"] = True

    if st.session_state.get("confirm_load_previous_csv", False):
        st.sidebar.warning("진짜 불러오겠습니까? 현재 작업 내용이 덮어 씌워질 수 있습니다.")
        if st.sidebar.button("불러오기 확인"):
            _load_previous_values_into_current_df(selected_csv)
            st.session_state["confirm_load_previous_csv"] = False
            st.sidebar.success(f"불러오기 완료: {selected_csv.name}")
            st.rerun()


def _render_auto_previous_csv_prompt() -> bool:
    """Prompt once to load latest CSV when dataset is opened in folder-select mode."""
    upload_source_type = st.session_state.get(KEY_UPLOAD_SOURCE_TYPE, "drag_upload")
    selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)
    if upload_source_type != "folder_select" or not isinstance(selected_subpath, str) or not selected_subpath.strip():
        return True

    result_dir = ensure_result_folder_from_selected_subpath(CSV_OUTPUT_ROOT_DIR, selected_subpath)
    latest_csv = find_latest_csv_file(result_dir)
    if latest_csv is None:
        return True

    prompt_key = f"auto_load_prompt_done::{selected_subpath}"
    if st.session_state.get(prompt_key, False):
        return True

    st.info("이전 파일이 있습니다. 불러오겠습니까?")
    st.caption(f"대상 파일: {latest_csv.name}")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("불러오기", key=f"auto_load_yes::{selected_subpath}"):
            _load_previous_values_into_current_df(latest_csv)
            st.session_state[prompt_key] = True
            st.rerun()
    with col_no:
        if st.button("건너뛰기", key=f"auto_load_no::{selected_subpath}"):
            st.session_state[prompt_key] = True
            st.rerun()
    return False


def _list_versioned_csv_files(result_dir: Path) -> list[Path]:
    """List versioned CSV files sorted by latest version first."""
    files = [path for path in result_dir.glob("*.csv") if parse_version_from_filename(path.name) is not None]
    files.sort(key=lambda path: parse_version_from_filename(path.name) or (0, 0), reverse=True)
    return files


def _load_previous_values_into_current_df(csv_path: Path) -> None:
    """Load selected CSV and merge defect/ATIS columns into current dataframe."""
    current_df = get_master_dataframe()
    if current_df is None or current_df.empty:
        st.warning("현재 데이터프레임이 비어 있어 불러올 수 없습니다.")
        return

    loaded_df = load_previous_defect_values(csv_path)
    merged_df = apply_loaded_defect_values(current_df, loaded_df)
    _reset_labeling_widget_state()
    set_master_dataframe(merged_df)
    touch_label_sync_token()


def _render_csv_import_section() -> None:
    """Render two-step CSV import flow for Defect columns only."""
    st.divider()
    st.subheader("CSV 가져오기")
    st.caption("외부에서 수정한 CSV의 Defect 값만 검증 후 반영합니다.")

    current_df = get_master_dataframe()
    if current_df is None or current_df.empty:
        st.info("현재 데이터가 없어 CSV를 가져올 수 없습니다.")
        return

    uploaded_file = st.file_uploader(
        "가져올 CSV 업로드",
        type=["csv"],
        key="labeling_csv_import_file",
        help="cell_id 기준으로 Defect 4개 컬럼만 반영합니다.",
    )
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    source_key = f"{uploaded_file.name}:{hashlib.md5(file_bytes).hexdigest()}"

    col_validate, col_apply = st.columns(2)
    with col_validate:
        if st.button("업로드 검증", key="labeling_csv_import_validate"):
            preview = _build_csv_import_preview(file_bytes=file_bytes, current_df=current_df)
            st.session_state[CSV_IMPORT_PREVIEW_KEY] = preview
            st.session_state[CSV_IMPORT_SOURCE_KEY] = source_key
            st.rerun()

    preview = st.session_state.get(CSV_IMPORT_PREVIEW_KEY)
    preview_source = st.session_state.get(CSV_IMPORT_SOURCE_KEY)
    if not isinstance(preview, dict) or preview_source != source_key:
        return

    _render_csv_import_preview(preview)
    if not preview.get("is_valid", False):
        return

    with col_apply:
        if st.button("Defect 값 적용", key="labeling_csv_import_apply"):
            updated_df = _apply_uploaded_defect_values(
                current_df=current_df,
                uploaded_df=preview["normalized_df"],
            )
            _reset_labeling_widget_state()
            set_master_dataframe(updated_df)
            touch_label_sync_token()
            st.session_state.pop(CSV_IMPORT_PREVIEW_KEY, None)
            st.session_state.pop(CSV_IMPORT_SOURCE_KEY, None)
            st.success("CSV Defect 값 반영이 완료되었습니다.")
            st.rerun()


def _build_csv_import_preview(*, file_bytes: bytes, current_df: pd.DataFrame) -> dict[str, object]:
    """Parse uploaded CSV, validate structure, and build compact preview summary."""
    preview: dict[str, object] = {
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "summary_rows": [],
    }

    try:
        uploaded_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        preview["errors"] = [f"CSV를 읽을 수 없습니다: {exc}"]
        return preview

    errors: list[str] = []
    warnings: list[str] = []

    if COL_CELL_ID not in uploaded_df.columns:
        errors.append("필수 컬럼 cell_id 가 없습니다.")
        preview["errors"] = errors
        preview["warnings"] = warnings
        return preview

    duplicate_mask = uploaded_df[COL_CELL_ID].astype(str).duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_values = uploaded_df.loc[duplicate_mask, COL_CELL_ID].astype(str).tolist()
        errors.append(f"중복 cell_id 가 있습니다: {', '.join(duplicate_values[:10])}")

    invalid_defect_columns = [
        column for column in uploaded_df.columns
        if column.startswith("Defect") and column not in DEFECT_COLUMNS
    ]
    if invalid_defect_columns:
        warnings.append(
            "지원하지 않는 Defect 컬럼은 무시됩니다: "
            + ", ".join(sorted(invalid_defect_columns))
        )

    supported_columns = [column for column in DEFECT_COLUMNS if column in uploaded_df.columns]
    if not supported_columns:
        warnings.append("가져올 Defect 컬럼이 없습니다. 반영 가능한 변경이 없습니다.")

    normalized_df = uploaded_df[[COL_CELL_ID, *supported_columns]].copy()
    normalized_df[COL_CELL_ID] = normalized_df[COL_CELL_ID].astype(str)
    for column in supported_columns:
        normalized_df[column] = normalized_df[column].fillna("").astype(str).str.strip()

    current_cell_ids = {str(value) for value in current_df[COL_CELL_ID].tolist()}
    uploaded_cell_ids = normalized_df[COL_CELL_ID].tolist()
    matched_cell_ids = {cell_id for cell_id in uploaded_cell_ids if cell_id in current_cell_ids}
    unmatched_count = sum(1 for cell_id in uploaded_cell_ids if cell_id not in current_cell_ids)

    non_empty_update_count = 0
    if matched_cell_ids and supported_columns:
        candidate_df = normalized_df[normalized_df[COL_CELL_ID].isin(matched_cell_ids)]
        for column in supported_columns:
            non_empty_update_count += int((candidate_df[column] != "").sum())

    preview["is_valid"] = len(errors) == 0
    preview["errors"] = errors
    preview["warnings"] = warnings
    preview["normalized_df"] = normalized_df
    preview["summary_rows"] = [
        {"항목": "업로드 row 수", "값": str(len(uploaded_df))},
        {"항목": "매칭 cell_id 수", "값": str(len(matched_cell_ids))},
        {"항목": "미매칭 cell_id 수", "값": str(unmatched_count)},
        {"항목": "반영 예정 Defect 값 수", "값": str(non_empty_update_count)},
    ]
    return preview


def _render_csv_import_preview(preview: dict[str, object]) -> None:
    """Render validation result and compact preview summary."""
    errors = preview.get("errors", [])
    warnings = preview.get("warnings", [])

    for error in errors if isinstance(errors, list) else []:
        st.error(error)
    for warning in warnings if isinstance(warnings, list) else []:
        st.warning(warning)

    summary_rows = preview.get("summary_rows", [])
    if isinstance(summary_rows, list) and summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _apply_uploaded_defect_values(*, current_df: pd.DataFrame, uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Apply uploaded non-empty Defect values to matching current rows by cell_id only."""
    if uploaded_df.empty:
        return current_df

    updated_df = current_df.copy()
    cell_index_map = {
        str(cell_id): index for index, cell_id in updated_df[COL_CELL_ID].astype(str).items()
    }

    for _, uploaded_row in uploaded_df.iterrows():
        row_cell_id = str(uploaded_row[COL_CELL_ID])
        target_index = cell_index_map.get(row_cell_id)
        if target_index is None:
            continue

        for defect_column in DEFECT_COLUMNS:
            if defect_column not in uploaded_df.columns:
                continue
            new_value = str(uploaded_row.get(defect_column, "")).strip()
            if not new_value:
                continue
            updated_df.at[target_index, defect_column] = new_value

    return updated_df


def _reset_labeling_widget_state() -> None:
    """Clear widget state that can become stale after programmatic dataframe updates."""
    stale_keys = [
        key
        for key in list(st.session_state.keys())
        if key.startswith("defect_") or key.startswith("labeling_csv_import_")
    ]
    for key in stale_keys:
        st.session_state.pop(key, None)


def _render_save_section(df: pd.DataFrame, image_map: dict[str, dict[str, object]]) -> None:
    """Render CSV and image save controls."""
    st.divider()
    st.subheader("저장")

    employee_id = str(st.session_state.get("auth_employee_id", "")).strip() or "unknown"

    separate_image_path = st.checkbox("별도 경로 생성하기", value=False)
    custom_image_root = ""
    if separate_image_path:
        custom_image_root = st.text_input("이미지 저장 상위 폴더명", value="")

    image_save_root = IMAGE_EXPORT_ROOT_DIR
    custom_folder = sanitize_token(custom_image_root.strip(), fallback="custom") if custom_image_root.strip() else None

    st.caption("각 이미지는 '이미지 측정 위치 / 불량 Type' 폴더로 저장됨. ex) CA(TOP)/융착/{이미지}")
    if st.button("라벨링 종료"):
        _release_current_dataset_lock()
        st.session_state["current_page"] = PAGE_UPLOAD
        st.rerun()

    if st.button("이미지 저장"):
        result = save_defect_images(
            df=df,
            image_map=image_map,
            save_root=image_save_root,
            employee_id=employee_id,
            custom_folder=custom_folder if separate_image_path else None,
        )
        st.success(
            f"이미지 저장 완료 - saved: {result['saved']}, skipped: {result['skipped']}"
        )

    csv_output_dir, csv_filename = _resolve_csv_output_dir_and_filename(employee_id)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

    st.caption(f"CSV 저장 경로: {csv_output_dir}")
    if st.button("CSV 파일 저장"):
        if not EMPLOYEE_ID_PATTERN.match(employee_id):
            st.error("CSV 저장 실패: employee_id 형식이 올바르지 않습니다. (예: so12345)")
            return
        saved_path = export_csv_without_filling_ok(df, csv_output_dir, csv_filename)
        selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)
        create_csv_backup_copy(
            saved_csv_path=saved_path,
            backup_root=CSV_BACKUP_ROOT_DIR,
            selected_subpath=selected_subpath if isinstance(selected_subpath, str) else None,
        )
        if isinstance(selected_subpath, str) and selected_subpath.strip():
            log_labeling_activity(
                db_path=AUTH_DB_PATH,
                employee_id=employee_id,
                selected_subpath=selected_subpath,
                df=df,
            )
            _release_current_dataset_lock()
        st.success(f"CSV 저장 완료: {saved_path}")

    st.download_button(
        "CSV 저장",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv",
    )

    st.caption("CSV 추출 대상 데이터프레임")
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("불량 요약 통계")
    current_summary_source = _build_defect_summary_source_key(df)
    if st.button("불량 요약 테이블 생성", key="build_defect_summary_table"):
        st.session_state[DEFECT_SUMMARY_DATA_KEY] = _build_defect_summary_table(df)
        st.session_state[DEFECT_SUMMARY_SOURCE_KEY] = current_summary_source
        st.rerun()

    summary_df = st.session_state.get(DEFECT_SUMMARY_DATA_KEY)
    summary_source = st.session_state.get(DEFECT_SUMMARY_SOURCE_KEY)
    if isinstance(summary_df, pd.DataFrame) and summary_source == current_summary_source:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _resolve_csv_output_dir_and_filename(employee_id: str) -> tuple[Path, str]:
    """Resolve matched output folder and next versioned file name."""
    upload_source_type = st.session_state.get(KEY_UPLOAD_SOURCE_TYPE, "drag_upload")
    selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)

    if not EMPLOYEE_ID_PATTERN.match(employee_id):
        st.warning("employee_id 형식은 영문 2자 + 숫자 5자여야 합니다. (예: so12345)")

    if upload_source_type == "folder_select" and isinstance(selected_subpath, str) and selected_subpath.strip():
        parts = [part for part in selected_subpath.split("/") if part]
        if len(parts) >= 2:
            line, period = parts[0], parts[1]
            target_dir = ensure_result_folder_from_selected_subpath(CSV_OUTPUT_ROOT_DIR, f"{line}/{period}")
            latest = find_latest_csv_version(target_dir)
            next_version = build_next_version(latest)
            file_name = build_next_version_filename(line, period, employee_id, next_version)
            return target_dir, file_name

    root_dir = Path(CSV_OUTPUT_ROOT_DIR).expanduser().resolve()
    existing_subdirs = ["(root)", *list_subdirectories_relative(root_dir)]
    selected_existing = st.selectbox("CSV 저장 하위 폴더 선택", options=existing_subdirs)
    chosen_dir = root_dir if selected_existing == "(root)" else root_dir / selected_existing
    latest = find_latest_csv_version(chosen_dir)
    next_version = build_next_version(latest)
    file_name = build_next_version_filename("manual", "manual", employee_id, next_version)
    return chosen_dir, file_name


def _render_navigation_buttons(current_index: int, total_count: int) -> None:
    """Render previous/next navigation controls."""
    col_prev, col_next = st.columns(2)

    with col_prev:
        if st.button("이전 cell", disabled=(current_index <= 0)):
            set_current_cell_index(current_index - 1)
            st.session_state[SIDEBAR_FORCE_SYNC_KEY] = True
            st.rerun()

    with col_next:
        if st.button("다음 cell", disabled=(current_index >= total_count - 1)):
            set_current_cell_index(current_index + 1)
            st.session_state[SIDEBAR_FORCE_SYNC_KEY] = True
            st.rerun()


def _build_defect_summary_source_key(df: pd.DataFrame) -> str:
    """Build a stable dataset identity key for the current labeling dataframe."""
    cell_ids = [str(value) for value in df[COL_CELL_ID].tolist()] if COL_CELL_ID in df.columns else []
    return "|".join(cell_ids)


def _build_defect_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build read-only summary table for current dataframe defect columns."""
    rows: list[dict[str, str]] = []
    top_columns = [f"Top{rank}" for rank in range(1, 6)]

    for defect_column in DEFECT_COLUMNS:
        if defect_column not in df.columns:
            row = {"Position": defect_column, "Total Count": "0ea"}
            row.update({column: "-" for column in top_columns})
            rows.append(row)
            continue

        series = df[defect_column].fillna("").astype(str).str.strip()
        filtered = series[(series != "") & (series.str.upper() != "OK")]
        counts = filtered.value_counts()

        row = {
            "Position": defect_column,
            "Total Count": f"{int(counts.sum())}ea",
        }

        top_items = list(counts.items())[:5]
        for index, column in enumerate(top_columns):
            if index < len(top_items):
                label, count = top_items[index]
                row[column] = f"{label} ({int(count)}ea)"
            else:
                row[column] = "-"
        rows.append(row)

    return pd.DataFrame(rows)


def _render_cell_progress_summary(df: pd.DataFrame) -> None:
    """Render cell-based progress summary aligned with upload-page logic."""
    total_cells = len(df)
    if total_cells <= 0:
        st.caption("전체 cell: 0")
        st.caption("처리된 cell: 0")
        st.caption("남은 cell: 0")
        st.caption("진행률: 0%")
        st.caption("총 이미지 수: 0")
        return

    defect_columns = [column for column in DEFECT_COLUMNS if column in df.columns]
    if defect_columns:
        defect_values = df[defect_columns].fillna("").astype(str).apply(lambda col: col.str.strip())
        processed_cells = int((defect_values != "").any(axis=1).sum())
    else:
        processed_cells = 0

    remaining_cells = total_cells - processed_cells
    progress_percent = int(round((processed_cells / total_cells) * 100))

    image_columns = [column for column in POSITION_COLUMNS if column in df.columns]
    total_images = 0
    if image_columns:
        numeric_images = df[image_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        total_images = int(numeric_images.sum().sum())

    st.caption(f"전체 cell: {total_cells}")
    st.caption(f"처리된 cell: {processed_cells}")
    st.caption(f"남은 cell: {remaining_cells}")
    st.caption(f"진행률: {progress_percent}%")
    st.caption(f"총 이미지 수: {total_images}")


def _ensure_dataset_lock() -> bool:
    """Ensure current dataset lock is acquired for this user."""
    selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)
    employee_id = str(st.session_state.get("auth_employee_id", "")).strip()
    if not isinstance(selected_subpath, str) or not selected_subpath.strip() or not employee_id:
        return True

    acquired, current = acquire_lock(
        db_path=AUTH_DB_PATH,
        dataset_key=selected_subpath,
        employee_id=employee_id,
    )
    if acquired:
        return True

    lock_owner = str((current or {}).get("employee_id", "알 수 없음"))
    lock_time = str((current or {}).get("locked_at", ""))
    st.error("현재 이 데이터셋은 다른 사용자가 작업 중입니다.")
    st.warning(f"현재 {lock_owner} 사용자가 작업 중입니다.\n잠금 시간: {lock_time}")
    if st.button("업로드 페이지로 이동"):
        st.session_state["current_page"] = PAGE_UPLOAD
        st.rerun()
    return False


def _release_current_dataset_lock() -> None:
    """Release current dataset lock for logged-in user."""
    selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)
    employee_id = str(st.session_state.get("auth_employee_id", "")).strip()
    if isinstance(selected_subpath, str) and selected_subpath.strip() and employee_id:
        release_lock(
            db_path=AUTH_DB_PATH,
            dataset_key=selected_subpath,
            employee_id=employee_id,
        )
