"""Read-only defect statistics page based on saved latest CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import ALLOWED_EXTENSIONS, CSV_OUTPUT_ROOT_DIR, IMAGE_ROOT_DIR
from src.constants import POSITION_COLUMNS
from src.file_parser import parse_filename
from src.save_manager import find_latest_csv_file


POSITION_TO_DEFECT_COLUMN = {
    "CA(TOP)": "Defect_CA(TOP)",
    "CA(BOT)": "Defect_CA(BOT)",
    "AN(TOP)": "Defect_AN(TOP)",
    "AN(BOT)": "Defect_AN(BOT)",
}

POSITION_TO_ATIS_COLUMN = {
    position: f"ATIS_{position}" for position in POSITION_COLUMNS
}

IMAGE_MATCHES_KEY = "defect_stats_image_matches"
IMAGE_FILTER_KEY = "defect_stats_image_filter"
IMAGE_PAGE_KEY = "defect_stats_image_page"

FILTER_FIELDS = ["line", "period", "position", "top_defect", "sub_defect"]
FILTER_LABELS = {
    "line": "line",
    "period": "period",
    "position": "position",
    "top_defect": "top defect",
    "sub_defect": "sub defect",
}
FILTER_MODE_ALL_HIDE = "전체선택(구분X)"
FILTER_MODE_ALL_SHOW = "전체선택(구분O)"
FILTER_MODE_DIRECT = "직접선택"
FILTER_MODES = [FILTER_MODE_ALL_HIDE, FILTER_MODE_ALL_SHOW, FILTER_MODE_DIRECT]


def render_defect_statistics_page() -> None:
    """Render read-only defect statistics page from saved latest CSV files."""
    st.title("불량 통계")
    records_df = _build_record_dataframe(CSV_OUTPUT_ROOT_DIR)
    raw_merged_df = _build_raw_merged_dataframe(CSV_OUTPUT_ROOT_DIR)

    if records_df.empty:
        st.info("표시할 통계 데이터가 없습니다. 저장된 CSV를 확인해주세요.")
        return

    filters = _render_filters(records_df)
    summary_df = _build_summary_table(records_df, filters)
    if summary_df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    _render_download_buttons(
        summary_df=summary_df,
        raw_merged_df=raw_merged_df,
        records_df=records_df,
        filters=filters,
    )
    _render_image_viewer(records_df, filters)


def _render_filters(records_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Render mode-based filters with optional multiselect values."""
    filters: dict[str, dict[str, object]] = {}
    columns = st.columns(5)

    for column_container, field in zip(columns, FILTER_FIELDS):
        options = sorted(records_df[field].astype(str).unique().tolist())
        mode_key = f"defect_stats_mode::{field}"
        values_key = f"defect_stats_values::{field}"

        with column_container:
            mode = st.selectbox(
                f"{FILTER_LABELS[field]} mode",
                options=FILTER_MODES,
                key=mode_key,
            )
            selected_values: list[str] = []
            if mode == FILTER_MODE_DIRECT:
                current_values = st.session_state.get(values_key, [])
                if not isinstance(current_values, list):
                    current_values = []
                st.session_state[values_key] = [value for value in current_values if value in options]
                selected_values = st.multiselect(
                    FILTER_LABELS[field],
                    options=options,
                    key=values_key,
                )

        filters[field] = {
            "mode": mode,
            "values": selected_values,
        }

    return filters


def _build_record_dataframe(csv_root: str | Path) -> pd.DataFrame:
    """Build raw defect records from the latest CSV of each line/period."""
    csv_root_path = Path(csv_root).expanduser().resolve()
    if not csv_root_path.exists() or not csv_root_path.is_dir():
        return pd.DataFrame(columns=["line", "period", "cell_id", "position", "top_defect", "sub_defect"])

    records: list[dict[str, str]] = []
    for line_dir in sorted(path for path in csv_root_path.iterdir() if path.is_dir()):
        for period_dir in sorted(path for path in line_dir.iterdir() if path.is_dir()):
            latest_csv = find_latest_csv_file(period_dir)
            if latest_csv is None:
                continue
            try:
                df = pd.read_csv(latest_csv)
            except Exception:
                continue
            records.extend(
                _extract_statistics_records(
                    df=df,
                    line=line_dir.name,
                    period=period_dir.name,
                )
            )

    if not records:
        return pd.DataFrame(columns=["line", "period", "cell_id", "position", "top_defect", "sub_defect"])

    return pd.DataFrame(records)


def _build_raw_merged_dataframe(csv_root: str | Path) -> pd.DataFrame:
    """Build merged raw dataframe from latest CSVs with line/period context."""
    csv_root_path = Path(csv_root).expanduser().resolve()
    base_columns = ["line", "period", "cell_id", *POSITION_TO_ATIS_COLUMN.values(), *POSITION_TO_DEFECT_COLUMN.values()]
    if not csv_root_path.exists() or not csv_root_path.is_dir():
        return pd.DataFrame(columns=base_columns)

    merged_frames: list[pd.DataFrame] = []
    for line_dir in sorted(path for path in csv_root_path.iterdir() if path.is_dir()):
        for period_dir in sorted(path for path in line_dir.iterdir() if path.is_dir()):
            latest_csv = find_latest_csv_file(period_dir)
            if latest_csv is None:
                continue
            try:
                df = pd.read_csv(latest_csv)
            except Exception:
                continue

            available_columns = [
                column for column in ["cell_id", *POSITION_TO_ATIS_COLUMN.values(), *POSITION_TO_DEFECT_COLUMN.values()]
                if column in df.columns
            ]
            if "cell_id" not in available_columns:
                continue

            merged_df = df[available_columns].copy()
            merged_df.insert(0, "period", period_dir.name)
            merged_df.insert(0, "line", line_dir.name)
            merged_frames.append(merged_df)

    if not merged_frames:
        return pd.DataFrame(columns=base_columns)

    return pd.concat(merged_frames, ignore_index=True)


def _extract_statistics_records(*, df: pd.DataFrame, line: str, period: str) -> list[dict[str, str]]:
    """Extract row-wise defect records from one CSV dataframe."""
    records: list[dict[str, str]] = []
    if "cell_id" not in df.columns:
        return records

    for position in POSITION_COLUMNS:
        defect_column = POSITION_TO_DEFECT_COLUMN[position]
        atis_column = POSITION_TO_ATIS_COLUMN[position]
        if defect_column not in df.columns:
            continue

        sub_series = df[defect_column].fillna("").astype(str).str.strip()
        top_series = (
            df[atis_column].fillna("").astype(str).str.strip()
            if atis_column in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        cell_series = df["cell_id"].fillna("").astype(str).str.strip()

        for cell_id, top_value, sub_value in zip(cell_series.tolist(), top_series.tolist(), sub_series.tolist()):
            normalized_sub = sub_value.strip()
            normalized_top = _normalize_top_defect(top_value)
            if not normalized_top:
                continue
            if "/" not in normalized_top and normalized_top.upper() == "OK":
                continue
            if (not normalized_sub or normalized_sub.upper() == "OK") and "/" not in normalized_top:
                continue

            records.append(
                {
                    "line": line,
                    "period": period,
                    "cell_id": cell_id,
                    "position": position,
                    "top_defect": normalized_top,
                    "sub_defect": normalized_sub,
                }
            )

    return records


def _build_summary_table(records_df: pd.DataFrame, filters: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Build filtered + dynamic-grouped summary table."""
    filtered_df = _apply_filters(records_df, filters)
    if filtered_df.empty:
        return pd.DataFrame()

    groupby_columns = _build_groupby_columns(filters)
    denominator_df = _apply_line_period_scope(records_df, filters)
    denominator = len(denominator_df) if not denominator_df.empty else len(filtered_df)

    if not groupby_columns:
        grouped = filtered_df.groupby(["line"], dropna=False).size().reset_index(name="count_value")
    else:
        grouped = filtered_df.groupby(groupby_columns, dropna=False).size().reset_index(name="count_value")

    grouped = grouped.sort_values("count_value", ascending=False).reset_index(drop=True)
    total_row = _build_total_summary_row(
        grouped=grouped,
        groupby_columns=groupby_columns if groupby_columns else ["line"],
        denominator=denominator,
    )
    grouped["count"] = grouped["count_value"].map(
        lambda value: f"{int(value)} ({(value / denominator * 100):.1f}%)"
    )
    display_df = grouped.drop(columns=["count_value"])
    return pd.concat([pd.DataFrame([total_row]), display_df], ignore_index=True)


def _apply_filters(records_df: pd.DataFrame, filters: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Apply include filters for all fields."""
    filtered_df = records_df.copy()
    for field in FILTER_FIELDS:
        mode = str(filters[field]["mode"])
        selected_values = [str(value) for value in filters[field]["values"]]
        if mode != FILTER_MODE_DIRECT:
            continue
        if not selected_values:
            return filtered_df.iloc[0:0].copy()
        filtered_df = filtered_df[filtered_df[field].isin(selected_values)]
    return filtered_df.reset_index(drop=True)


def _build_total_summary_row(
    *,
    grouped: pd.DataFrame,
    groupby_columns: list[str],
    denominator: int,
) -> dict[str, str]:
    """Build top total row for the current summary table."""
    total_count = int(grouped["count_value"].sum()) if "count_value" in grouped.columns else 0
    total_ratio = (total_count / denominator * 100) if denominator > 0 else 0.0

    row = {column: "" for column in groupby_columns}
    first_column = groupby_columns[0] if groupby_columns else "line"
    row[first_column] = "전체합계"
    row["count"] = f"{total_count} ({total_ratio:.1f}%)"
    return row


def _build_groupby_columns(filters: dict[str, dict[str, object]]) -> list[str]:
    """Build dynamic groupby columns from filter mode configuration."""
    return [
        field
        for field in FILTER_FIELDS
        if str(filters[field]["mode"]) in {FILTER_MODE_ALL_SHOW, FILTER_MODE_DIRECT}
    ]


def _apply_line_period_scope(records_df: pd.DataFrame, filters: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Build denominator scope using line/period include rules only."""
    scoped_df = records_df.copy()
    for field in ["line", "period"]:
        mode = str(filters[field]["mode"])
        selected_values = [str(value) for value in filters[field]["values"]]
        if mode != FILTER_MODE_DIRECT:
            continue
        if not selected_values:
            return scoped_df.iloc[0:0].copy()
        scoped_df = scoped_df[scoped_df[field].isin(selected_values)]
    return scoped_df.reset_index(drop=True)


def _normalize_top_defect(value: str) -> str:
    """Normalize ATIS top defect text for stats/filtering.

    Keeps the latest overridden value for defect-to-defect changes like `Damage/Scrap`,
    but preserves the full transition text for defect-to-OK changes like `Scrap/OK`
    so they remain visible and filterable in the statistics UI.
    """
    text = value.strip()
    if not text or text.lower() == "nan":
        return ""
    if "/" in text:
        original, latest = text.split("/", 1)
        original = original.strip()
        latest = latest.strip()
        if latest.upper() == "OK":
            return f"{original}/{latest}" if original else latest
        return latest
    return text


def _render_download_buttons(
    *,
    summary_df: pd.DataFrame,
    raw_merged_df: pd.DataFrame,
    records_df: pd.DataFrame,
    filters: dict[str, dict[str, object]],
) -> None:
    """Render CSV download buttons for aggregated and filtered raw data."""
    aggregated_download_df = _build_aggregated_download_df(summary_df)
    raw_download_df = _build_filtered_raw_download_df(raw_merged_df, records_df, filters)

    col_agg, col_raw = st.columns(2)
    with col_agg:
        st.download_button(
            "집계 결과 다운로드",
            data=aggregated_download_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="defect_stats_aggregated.csv",
            mime="text/csv",
            key="defect_stats_download_aggregated",
        )
    with col_raw:
        st.download_button(
            "원본 데이터 다운로드",
            data=raw_download_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="defect_stats_raw_filtered.csv",
            mime="text/csv",
            key="defect_stats_download_raw",
        )


def _build_aggregated_download_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build download dataframe for the currently displayed aggregated table."""
    download_df = summary_df.copy()
    if "count" not in download_df.columns:
        return download_df

    parsed = download_df["count"].astype(str).str.extract(r"^\s*(?P<count>\d+)\s*\((?P<ratio>[\d.]+)%\)\s*$")
    if not parsed.empty:
        download_df["count"] = parsed["count"].fillna(download_df["count"])
        download_df["ratio (%)"] = parsed["ratio"].fillna("")
    return download_df


def _build_filtered_raw_download_df(
    raw_merged_df: pd.DataFrame,
    records_df: pd.DataFrame,
    filters: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Build filtered raw merged dataframe for current filter context."""
    if raw_merged_df.empty:
        return raw_merged_df.copy()

    filtered_records_df = _apply_filters(records_df, filters)
    if filtered_records_df.empty:
        return raw_merged_df.iloc[0:0].copy()

    keys = {
        (str(row["line"]), str(row["period"]), str(row["cell_id"]))
        for _, row in filtered_records_df.iterrows()
    }

    key_series = list(
        zip(
            raw_merged_df["line"].astype(str),
            raw_merged_df["period"].astype(str),
            raw_merged_df["cell_id"].astype(str),
        )
    )
    mask = [key in keys for key in key_series]
    return raw_merged_df.loc[mask].reset_index(drop=True)


def _render_image_viewer(records_df: pd.DataFrame, filters: dict[str, dict[str, object]]) -> None:
    """Render on-demand image viewer for current filters."""
    st.divider()
    st.subheader("이미지 보기")
    st.caption("필터를 선택한 뒤 버튼을 눌렀을 때만 이미지를 조회합니다.")

    position_filter = filters["position"]
    if position_filter["mode"] != FILTER_MODE_DIRECT or len(position_filter["values"]) != 1:
        st.info("이미지 보기를 위해 position을 직접선택으로 1개 선택해주세요.")
        return

    filter_key = _build_image_filter_key(filters)
    if st.button("이미지 보기", key="defect_stats_view_images"):
        st.session_state[IMAGE_MATCHES_KEY] = _build_image_matches(records_df, filters)
        st.session_state[IMAGE_FILTER_KEY] = filter_key
        st.session_state[IMAGE_PAGE_KEY] = 1
        st.rerun()

    matches = st.session_state.get(IMAGE_MATCHES_KEY)
    active_filter_key = st.session_state.get(IMAGE_FILTER_KEY)
    if not isinstance(matches, list) or active_filter_key != filter_key:
        return

    if not matches:
        st.warning("선택한 조건에 해당하는 이미지가 없습니다.")
        return

    total_pages = max(1, (len(matches) + 9) // 10)
    current_page = int(st.session_state.get(IMAGE_PAGE_KEY, 1))
    current_page = min(max(current_page, 1), total_pages)
    st.session_state[IMAGE_PAGE_KEY] = current_page

    col_prev, col_input, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("이전 페이지", disabled=(current_page <= 1), key="defect_stats_prev_page"):
            st.session_state[IMAGE_PAGE_KEY] = current_page - 1
            st.rerun()
    with col_input:
        requested_page = st.number_input(
            "페이지 번호",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
            key="defect_stats_page_input",
        )
        if int(requested_page) != current_page:
            st.session_state[IMAGE_PAGE_KEY] = int(requested_page)
            st.rerun()
    with col_next:
        if st.button("다음 페이지", disabled=(current_page >= total_pages), key="defect_stats_next_page"):
            st.session_state[IMAGE_PAGE_KEY] = current_page + 1
            st.rerun()

    st.caption(f"{current_page} / {total_pages} 페이지")
    start = (current_page - 1) * 10
    page_items = matches[start : start + 10]

    for row_start in range(0, len(page_items), 5):
        cols = st.columns(5)
        for idx, item in enumerate(page_items[row_start : row_start + 5]):
            with cols[idx]:
                st.image(str(item["image_path"]), use_container_width=True)
                st.caption(f"{item['cell_id']} | {item['position']}")


def _build_image_filter_key(filters: dict[str, dict[str, object]]) -> str:
    """Build stable key for the current image-view filters."""
    parts: list[str] = []
    for field in FILTER_FIELDS:
        mode = str(filters[field]["mode"])
        values = ",".join(sorted(str(value) for value in filters[field]["values"]))
        parts.append(f"{field}:{mode}:{values}")
    return "|".join(parts)


def _build_image_matches(records_df: pd.DataFrame, filters: dict[str, dict[str, object]]) -> list[dict[str, str | Path]]:
    """Build image match list from current filtered raw records."""
    image_root_path = Path(IMAGE_ROOT_DIR).expanduser().resolve()
    if not image_root_path.exists() or not image_root_path.is_dir():
        return []

    filtered_df = _apply_filters(records_df, filters)
    if filtered_df.empty:
        return []

    target_position = str(filters["position"]["values"][0])
    filtered_df = filtered_df[filtered_df["position"] == target_position]
    if filtered_df.empty:
        return []

    matches: list[dict[str, str | Path]] = []
    cell_keys = {
        (str(row["line"]), str(row["period"]), str(row["cell_id"]))
        for _, row in filtered_df.iterrows()
    }
    datasets = sorted({(str(row["line"]), str(row["period"])) for _, row in filtered_df.iterrows()})

    for line_name, period_name in datasets:
        image_dir = image_root_path / line_name / period_name
        if not image_dir.exists() or not image_dir.is_dir():
            continue

        for image_path in _collect_image_paths(image_dir):
            parse_result = parse_filename(image_path)
            if not parse_result.is_valid or parse_result.position != target_position:
                continue

            key = (line_name, period_name, str(parse_result.cell_id))
            if key not in cell_keys:
                continue

            matches.append(
                {
                    "line": line_name,
                    "period": period_name,
                    "cell_id": str(parse_result.cell_id),
                    "position": target_position,
                    "image_path": image_path,
                }
            )

    matches.sort(key=lambda item: str(item["cell_id"]))
    return matches


def _collect_image_paths(image_dir: Path) -> list[Path]:
    """Collect image files under existing dataset image folder."""
    allowed_suffixes = {f".{ext.lower().lstrip('.')}" for ext in ALLOWED_EXTENSIONS}
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
