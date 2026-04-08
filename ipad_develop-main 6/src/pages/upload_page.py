"""Upload page UI for Step 4.

This page handles:
- drag-and-drop upload
- folder selection upload (period/line two-level)
- upload validation
- filename parsing
- image_map construction
- master dataframe construction
- missing count summary
- dataframe preview
- session state save + move to labeling page
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import ALLOWED_EXTENSIONS, CSV_OUTPUT_ROOT_DIR, IMAGE_ROOT_DIR, IMAGE_ROOT_PATH
from config import AUTH_DB_PATH
from src.lock.dataset_lock_manager import release_lock
from src.atis_loader import merge_atis_to_master
from src.auth.user_store import log_dataset_access
from src.constants import DEFECT_COLUMNS
from src.lock.dataset_lock_manager import get_active_locks
from src.save_manager import find_latest_csv_file
from src.constants import (
    COL_CELL_ID,
    KEY_SELECTED_FOLDER_INFO,
    KEY_SELECTED_IMAGE_SUBPATH,
    KEY_UPLOAD_SOURCE_TYPE,
    PAGE_LABELING,
    POSITION_COLUMNS,
)
from src.dataframe_builder import build_master_dataframe
from src.image_registry import build_image_map
from src.state_manager import (
    initialize_session_state,
    set_image_map,
    set_master_dataframe,
    set_current_cell_index,
    set_upload_completed,
)
from src.validation import (
    extract_parse_failures,
    parse_files_with_results,
    validate_file_count,
    validate_file_extensions,
)
from utils.path_utils import collect_files_with_extensions, ensure_directory, list_subdirectories


EXPECTED_LINES = ["G8", "G9", "GA", "GB", "GC"]
EXPECTED_PERIODS = ["1주차", "2주차", "3주차", "4주차"]


def render_upload_page() -> None:
    """Render upload page and process uploaded images."""
    initialize_session_state()
    _release_lock_if_any()

    st.title("이미지 업로드")
    st.caption("이미지를 업로드한 뒤, cell_id 기준 master dataframe을 생성합니다.")
    _render_csv_status_overview_table()

    current_upload_source_type = st.session_state.get(KEY_UPLOAD_SOURCE_TYPE, "drag_upload")
    upload_source = st.radio(
        "업로드 방식 선택",
        options=["드래그 업로드", "폴더 선택 업로드"],
        index=1 if current_upload_source_type == "folder_select" else 0,
        horizontal=True,
    )

    if upload_source == "드래그 업로드":
        uploaded_files = st.file_uploader(
            "이미지 파일을 업로드하세요 (jpg, jpeg, png)",
            type=list(ALLOWED_EXTENSIONS),
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("업로드할 이미지를 선택해주세요.")
            return

        _render_uploaded_files_save_section(uploaded_files)
        _render_validation_result(
            uploaded_files,
            upload_source_type="drag_upload",
            selected_folder_info="",
            selected_image_subpath=None,
        )
        return

    folder_files, folder_info = _render_folder_selector()
    if folder_files is None:
        return

    _render_validation_result(
        folder_files,
        upload_source_type="folder_select",
        selected_folder_info=folder_info,
        selected_image_subpath=folder_info,
    )


def _render_uploaded_files_save_section(uploaded_files: list[Any]) -> None:
    """Optionally save drag-uploaded original files under IMAGE_ROOT_PATH."""
    st.subheader("업로드 원본 저장")
    st.caption(f"기본 저장 경로: {IMAGE_ROOT_PATH}")

    selected_line = st.selectbox("저장 Line", options=EXPECTED_LINES, key="raw_save_line")
    selected_period = st.selectbox("저장 Period", options=EXPECTED_PERIODS, key="raw_save_period")
    st.caption(f"저장 대상: {IMAGE_ROOT_PATH / selected_line / selected_period}")

    if st.button("업로드한 원본 이미지 저장"):
        try:
            save_root = ensure_directory(IMAGE_ROOT_PATH / selected_line / selected_period)

            saved_count = 0
            for uploaded_file in uploaded_files:
                output_path = save_root / Path(str(uploaded_file.name)).name
                output_path.write_bytes(uploaded_file.getvalue())
                saved_count += 1

            st.success(f"원본 이미지 저장 완료: {saved_count}개 ({save_root})")
        except Exception as error:  # pragma: no cover - UI safety fallback
            st.error(f"원본 저장 중 오류가 발생했습니다: {error}")


def _render_folder_selector() -> tuple[list[Path] | None, str]:
    """Render two-level folder selector and return collected image file paths."""
    st.subheader("폴더 선택 업로드")
    st.caption(f"루트 폴더: {IMAGE_ROOT_DIR}")

    lines = list_subdirectories(IMAGE_ROOT_DIR)
    if not lines:
        st.warning("루트 폴더에 라인 하위 폴더가 없습니다.")
        return None, ""

    selected_line = st.selectbox("1단계: 라인", options=lines)
    line_path = IMAGE_ROOT_DIR / selected_line

    periods = list_subdirectories(line_path)
    if not periods:
        st.warning("선택한 라인 폴더에 기간 하위 폴더가 없습니다.")
        return None, ""

    selected_period = st.selectbox("2단계: 기간", options=periods)
    period_path = line_path / selected_period

    collected_files = collect_files_with_extensions(period_path, ALLOWED_EXTENSIONS)
    if not collected_files:
        st.warning("선택한 라인 폴더에 업로드 가능한 이미지 파일이 없습니다.")
        return None, ""

    st.success(f"선택된 폴더에서 {len(collected_files)}개 이미지를 찾았습니다.")
    folder_info = f"{selected_line}/{selected_period}"
    return collected_files, folder_info


def _render_validation_result(
    uploaded_files: list[Any],
    *,
    upload_source_type: str,
    selected_folder_info: str,
    selected_image_subpath: str | None,
) -> None:
    """Validate uploads, build artifacts, and expose save/navigation actions."""
    count_error = validate_file_count(uploaded_files)
    if count_error:
        st.error(count_error)
        return

    invalid_extensions = validate_file_extensions(uploaded_files)
    if invalid_extensions:
        st.error("지원하지 않는 확장자가 포함되어 있습니다.")
        st.write(invalid_extensions)
        return

    parsed_pairs = parse_files_with_results(uploaded_files)
    parse_failures = extract_parse_failures(parsed_pairs)
    if parse_failures:
        st.warning("일부 파일은 파싱에 실패하여 제외됩니다.")
        st.dataframe(parse_failures, use_container_width=True)

    image_map = build_image_map(parsed_pairs)
    master_df = build_master_dataframe(image_map)

    if master_df.empty:
        st.error("유효한 파일이 없어 master dataframe을 생성할 수 없습니다.")
        return

    master_df, atis_message = merge_atis_to_master(master_df)
    st.info(atis_message)

    _render_missing_counts(master_df)

    st.subheader("Master DataFrame Preview")
    st.dataframe(master_df, use_container_width=True)

    if st.button("업로드 결과 저장 후 라벨링 페이지로 이동", type="primary"):
        employee_id = str(st.session_state.get("auth_employee_id", "")).strip()
        if employee_id and selected_folder_info:
            log_dataset_access(
                db_path=AUTH_DB_PATH,
                employee_id=employee_id,
                folder_name=selected_folder_info,
            )

        st.session_state[KEY_UPLOAD_SOURCE_TYPE] = upload_source_type
        st.session_state[KEY_SELECTED_FOLDER_INFO] = selected_folder_info
        st.session_state[KEY_SELECTED_IMAGE_SUBPATH] = selected_image_subpath
        set_master_dataframe(master_df)
        set_image_map(image_map)
        set_current_cell_index(0)
        st.session_state["sidebar_force_sync"] = True
        set_upload_completed(True)
        st.session_state["current_page"] = PAGE_LABELING
        st.success("세션 상태 저장 완료. 라벨링 페이지로 이동합니다.")
        st.rerun()


def _render_missing_counts(master_df: Any) -> None:
    """Show missing image counts by position.

    Missing count = number of cells where position column value is 0.
    """
    st.subheader("위치별 누락 수량")
    columns = st.columns(len(POSITION_COLUMNS))

    for idx, position in enumerate(POSITION_COLUMNS):
        missing_count = int((master_df[position] == 0).sum())
        columns[idx].metric(label=position, value=f"{missing_count} cells")

    st.caption(f"전체 cell 수: {master_df[COL_CELL_ID].nunique()}")


def _render_csv_status_overview_table() -> None:
    """Render dataset progress board with CSV-first and image-workload fallback."""
    st.subheader("데이터셋 진행 현황")

    lock_map = _get_active_lock_map(str(AUTH_DB_PATH))
    rows: list[dict[str, str]] = []
    for line in EXPECTED_LINES:
        row: dict[str, str] = {"라인": line}
        for period in EXPECTED_PERIODS:
            dataset_key = f"{line}/{period}"
            progress_text = _calculate_progress_text_for_dataset(
                csv_root=str(CSV_OUTPUT_ROOT_DIR),
                line=line,
                period=period,
            )
            lock_employee = lock_map.get(dataset_key)
            if progress_text != "-" and lock_employee:
                progress_text = f"{progress_text} - {lock_employee} 작업중"
            row[period] = progress_text
        rows.append(row)

    progress_df = pd.DataFrame(rows).set_index("라인")
    st.dataframe(progress_df, use_container_width=True)


@st.cache_data(ttl=60)
def _get_active_lock_map(db_path: str) -> dict[str, str]:
    """Return dataset_key -> employee_id map for active locks."""
    locks = get_active_locks(db_path)
    return {
        str(lock.get("dataset_key", "")): str(lock.get("employee_id", ""))
        for lock in locks
        if lock.get("dataset_key") and lock.get("employee_id")
    }


@st.cache_data(ttl=60)
def _calculate_progress_text_for_dataset(*, csv_root: str, line: str, period: str) -> str:
    """Calculate dataset progress text using latest CSV; fallback to image workload."""
    latest_csv = find_latest_csv_file(Path(csv_root) / line / period)
    if latest_csv is None:
        image_count = _count_dataset_images_in_folder(line=line, period=period)
        return f"미작업 ({image_count} imgs)"

    try:
        df = pd.read_csv(latest_csv)
    except Exception:
        image_count = _count_dataset_images_in_folder(line=line, period=period)
        return f"미작업 ({image_count} imgs)"

    total_cells = len(df)
    if total_cells <= 0:
        image_count = _count_dataset_images_in_folder(line=line, period=period)
        return f"미작업 ({image_count} imgs)"

    defect_columns = [column for column in DEFECT_COLUMNS if column in df.columns]
    if not defect_columns:
        image_count = _count_dataset_images_in_folder(line=line, period=period)
        return f"미작업 ({image_count} imgs)"

    defect_values = df[defect_columns].fillna("").astype(str).apply(lambda col: col.str.strip())
    processed_cells = int((defect_values != "").any(axis=1).sum())
    progress_percent = int(round((processed_cells / total_cells) * 100))

    image_columns = [column for column in POSITION_COLUMNS if column in df.columns]
    total_images = 0
    if image_columns:
        numeric_images = df[image_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        total_images = int(numeric_images.sum().sum())

    if progress_percent >= 100:
        return f"100% 완료 ({processed_cells}/{total_cells} cells, {total_images} imgs)"
    return f"{progress_percent}% ({processed_cells}/{total_cells} cells, {total_images} imgs)"


@st.cache_data(ttl=120)
def _count_dataset_images_in_folder(*, line: str, period: str) -> int:
    """Count image files by allowed extensions under `data/images/{line}/{period}`."""
    dataset_dir = Path(IMAGE_ROOT_DIR) / line / period
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return 0
    files = collect_files_with_extensions(dataset_dir, ALLOWED_EXTENSIONS)
    return len(files)


def _release_lock_if_any() -> None:
    """Best-effort lock release when user leaves labeling to upload."""
    selected_subpath = st.session_state.get(KEY_SELECTED_IMAGE_SUBPATH)
    employee_id = str(st.session_state.get("auth_employee_id", "")).strip()
    if isinstance(selected_subpath, str) and selected_subpath.strip() and employee_id:
        release_lock(
            db_path=AUTH_DB_PATH,
            dataset_key=selected_subpath,
            employee_id=employee_id,
        )
