"""Detect paraclinical triggers and clinical bleeding events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config
from src.parse_text import normalize, parse_vietnamese_date_fragment, split_clinical_notes

# ---------------------------------------------------------------------------
# Bleeding event NLP inputs — edit here to tune detection
# ---------------------------------------------------------------------------

BLEEDING_KEYWORDS: tuple[str, ...] = (
    "chảy máu",
    "xuất huyết",
    "phân đen",
    "dịch dẫn lưu hồng",
  # Active bleeding phrases (Vietnamese clinical narrative)
    "ra máu",
    "nôn ra máu",
    "nôn máu",
    "tiểu máu",
    "thấm máu",
    "đi ngoài ra máu",
    "mảng máu",
)

BLEEDING_NEGATION_PATTERNS: tuple[str, ...] = (
    r"không xuất huyết",
    r"không chảy máu",
    r"không sưng nề chảy máu",
    r"chưa phát hiện điểm chảy máu bất thường",
    r"không ra máu",
    r"không thấm máu",
    r"không nôn ra máu",
    r"không nôn máu",
    r"không có dấu hiệu[^.]{0,40}xuất huyết",
    r"không có dấu hiệu[^.]{0,40}chảy máu",
    r"không có dấu hiệu[^.]{0,40}ra máu",
    r"mục tiêu:\s*không có dấu hiệu",
    # r"không\s+cầm máu",
)

BLEEDING_EXCLUDE_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"nguy cơ/rủi ro khác",
    r"nguy cơ\s*(?:/rủi ro)?[^.]{0,40}(?:chảy máu|xuất huyết)",
    r"chẩn đoán\s+\d[^.]{0,120}(?:chảy máu|xuất huyết)",
    r"mục tiêu[^.]{0,100}(?:chảy máu|xuất huyết)",
    r"chuyển dạng\s+(?:chảy máu|xuất huyết)",
    r"xuất huyết chuyển dạng",
    r"chảy máu chuyển dạng",
    r"chuyển dạng\s+hi\d",
    r"dạng\s+chảy máu",
    r"(?:hình ảnh|trên phim|phim chụp|chụp phim)[^.]{0,100}(?:chảy máu|xuất huyết)",
    r"(?:chảy máu|xuất huyết)[^.]{0,40}(?:trên phim|hình ảnh|phim)",
    r"phản ứng thuốc nguy cơ",
    r"tai biến trong\s+(?:phẫu thuật|mổ)",
    r"dụng cụ tim mạch",
    r"cơn đau thắt ngực.{0,80}stent",
    r"phiếu chăm sóc",
    r"chăm sóc cấp",
    r"can thiệp giáo dục sức khỏe",
    r"đột qu[ỵị][^.;]{0,40}không\s+xác\s*định[^.;]{0,80}(?:xuất huyết|chảy máu|nhồi máu)",
    r"không\s+xác\s*định\s+do (?:xuất huyết|chảy máu) hay nhồi máu",
    r"không\s+xác\s*định\s+nhồi máu hay (?:xuất huyết|chảy máu)",
)

BLEEDING_NARRATIVE_CUTOFF_MARKERS: tuple[str, ...] = (
    "stt tên thuốc/hàm lượng",
    "ngày sử dụng",
    "phiếu chăm sóc",
    "chăm sóc cấp",
    "phiếu theo dõi",
    "bộ y tế phiếu chăm sóc",
    "iii. chẩn đoán",
    "iv. tình trạng ra viện",
    "v. tiên lượng",
    "vi. hướng điều trị",
    "phiếu khám bệnh vào viện",
    "kết quả điều trị:",
)

BLEEDING_INTERVENTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"kẹp.{0,40}clip", "kẹp clip"),
    (r"mổ cầm máu", "mổ cầm máu"),
    (r"(?<!mổ\s)cầm máu", "cầm máu"),
)

BLEEDING_PROCEDURE_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"chọc\s+(?:động\s+)?mạch",
    r"chọc\s+mạch",
    r"đặt\s+catheter",
    r"vị\s+trí\s+(?:chọc|can\s+thiệp)",
    r"đường\s+vào\s+(?:mạch|can\s+thiệp)",
)

BLEEDING_PROCEDURE_KEYWORDS: frozenset[str] = frozenset(
    {"thấm máu", "ra máu", "chảy máu"}
)

EARLY_TREATMENT_BLOCK_LIMIT = 5
EARLY_ADMISSION_DAYS = 2
PLAUSIBLE_CALENDAR_YEAR_MIN = 1990
PLAUSIBLE_CALENDAR_YEAR_MAX = 2035

HISTORY_ZONE_MARKERS: tuple[str, ...] = (
    "tiền sử",
    "bệnh sử",
    "ngày qua",
    "bệnh viện tuyến dưới",
)

BLEEDING_CONTEXT_RADIUS = 200
BLEEDING_NEGATION_RADIUS = 120
HISTORY_YEAR_RADIUS = 120

# ---------------------------------------------------------------------------
# Internal regex (clinical sections, DOAC stop, etc.)
# ---------------------------------------------------------------------------

RE_DIEN_BIEN_SECTION = re.compile(
    r"(?i)(?=ngày\s+giờ\s+diễn\s+biến\s+bệnh)",
)
RE_SECTION_DATE = re.compile(
    r"(?i)(?:ngày\s+sử\s+dụng[:\s]*|ngày\s+y\s+lệnh[:\s]*)"
    r"(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
)
RE_NEARBY_DATE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
)
RE_DOAC_NAME = re.compile(
    r"(?i)(lixiana|edoxaban|rivacryst|xarelto|rivaroxaban|pradaxa|dabigatran|eliquis|apixaban)",
)
RE_DOAC_STOP = re.compile(
    r"(?i)(ngừng|dừng|tạm\s+dừng|đình\s+chỉ|thu\s+hồi)",
)
RE_DISCHARGE_BOILERPLATE = re.compile(
    r"(?i)(ii\.thông tin đơn thuốc|đơn thuốc có giá trị mua|lời dặn bác sĩ|hướng dẫn sau ra viện)",
)

DOAC_STOP_WINDOW = 100

_BLEEDING_NEGATION_RE = [
    re.compile(pattern, re.IGNORECASE) for pattern in BLEEDING_NEGATION_PATTERNS
]
_BLEEDING_EXCLUDE_RE = [
    re.compile(pattern, re.IGNORECASE) for pattern in BLEEDING_EXCLUDE_CONTEXT_PATTERNS
]
_BLEEDING_INTERVENTION_RE = [
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in BLEEDING_INTERVENTION_PATTERNS
]
_BLEEDING_PROCEDURE_RE = [
    re.compile(pattern, re.IGNORECASE) for pattern in BLEEDING_PROCEDURE_CONTEXT_PATTERNS
]
RE_CONSULTATION_BLOCK = re.compile(r"(?i)phòng\s+hội\s+chẩn")
RE_HISTORICAL_YEAR = re.compile(r"(\d{4})")


@dataclass
class ClinicalSection:
    text: str
    kind: str  # "to_dieu_tri" | "dien_bien"
    block_index: int  # 0-based among tờ điều trị splits; -1 for diễn biến


def _normalize_patient_record(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return str(value)
    return digits.zfill(12)


def detect_bleeding_and_triggers(text_dir: Path | None = None) -> None:
    """Write trigger, bleeding checkpoint CSVs, and discordance analysis table."""
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_EXPORT.mkdir(parents=True, exist_ok=True)
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    patients_df = _load_patients()
    labs_df = _load_labs()
    doac_start_dates = _load_doac_start_dates()

    trigger_rows = _detect_lab_triggers(labs_df)
    bleeding_rows: list[dict] = []

    for _, patient in patients_df.iterrows():
        source_file = str(patient["source_file"])
        patient_record = _normalize_patient_record(patient["patient_record"])
        date_admission = patient.get("date_admission")
        path = text_dir / source_file
        if not path.exists():
            continue

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        trigger_rows.extend(
            _detect_doac_stop_triggers(raw_text, patient_record, source_file)
        )
        bleeding_rows.extend(
            _detect_bleeding_events(
                raw_text,
                patient_record,
                source_file,
                date_admission,
                doac_start_dates,
            )
        )

    triggers_df = pd.DataFrame(trigger_rows)
    if triggers_df.empty:
        triggers_df = pd.DataFrame(
            columns=[
                "patient_record",
                "trigger_type",
                "date",
                "value",
                "baseline",
                "source_file",
            ]
        )
    else:
        triggers_df = triggers_df.drop_duplicates(
            subset=["patient_record", "trigger_type", "date", "value", "baseline"]
        )

    bleeding_df = _finalize_bleeding_events(bleeding_rows)

    triggers_df.to_csv(config.CHECKPOINT_TRIGGERS_CSV, index=False)
    bleeding_df.to_csv(config.CHECKPOINT_BLEEDING_CSV, index=False)

    analysis_df = _build_discordance_analysis(patients_df, triggers_df, bleeding_df)
    analysis_df["patient_record"] = analysis_df["patient_record"].map(_normalize_patient_record)
    if "patient_id" in analysis_df.columns:
        analysis_df["patient_id"] = analysis_df["patient_id"].apply(
            lambda value: re.sub(r"\D", "", str(value)).zfill(10)
            if re.sub(r"\D", "", str(value))
            else str(value)
        )
    analysis_df.to_csv(config.BLEEDING_DISCORDANCE_CSV, index=False)

    dose_discordance_df, dose_discordance_summary = _build_dose_discordance_exports(
        patients_df
    )
    dose_discordance_df.to_csv(config.CHECKPOINT_DOSE_DISCORDANCE_CSV, index=False)
    _write_dose_discordance_checkpoint(dose_discordance_summary)

    summary = _build_cohort_summary(triggers_df, bleeding_df, analysis_df)
    _write_cohort_checkpoint(summary)
    _print_summary(summary, dose_discordance_summary)


def _load_patients() -> pd.DataFrame:
    if not config.PATIENTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {config.PATIENTS_CSV}; run metadata extraction first."
        )
    return pd.read_csv(config.PATIENTS_CSV, dtype=str)


def _load_labs() -> pd.DataFrame:
    if not config.LABS_LONG_CSV.exists():
        raise FileNotFoundError(
            f"Missing {config.LABS_LONG_CSV}; run lab extraction first."
        )
    labs = pd.read_csv(config.LABS_LONG_CSV)
    labs["date_return"] = pd.to_datetime(labs["date_return"], errors="coerce")
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    return labs


def _load_doac_start_dates() -> dict[str, pd.Timestamp]:
    """First active DOAC date per patient from medications_long.csv."""
    if not config.MEDICATIONS_LONG_CSV.exists():
        return {}

    meds = pd.read_csv(config.MEDICATIONS_LONG_CSV, dtype=str)
    meds["patient_record"] = meds["patient_record"].map(_normalize_patient_record)
    meds["date_active"] = pd.to_datetime(meds["date_active"], errors="coerce")
    doac = meds[meds["drug_generic"].isin(config.DOAC_DRUGS)].copy()
    if doac.empty:
        return {}

    not_stopped = doac[doac["is_stopped"].str.lower() != "yes"]
    source = not_stopped if not not_stopped.empty else doac
    grouped = source.groupby("patient_record")["date_active"].min()
    return {
        patient: date
        for patient, date in grouped.items()
        if pd.notna(date)
    }


def _is_consultation_block(text: str) -> bool:
    return bool(RE_CONSULTATION_BLOCK.search(text[:300]))


def _split_bleeding_search_sections(raw_text: str) -> list[ClinicalSection]:
    """Clinical progress-note blocks: tờ điều trị and ngày giờ diễn biến bệnh y lệnh."""
    sections: list[ClinicalSection] = []
    block_index = 0
    for part in split_clinical_notes(raw_text):
        part = part.strip()
        if not part or _is_consultation_block(part):
            continue
        sections.append(
            ClinicalSection(text=part, kind="to_dieu_tri", block_index=block_index)
        )
        block_index += 1

    for part in re.split(RE_DIEN_BIEN_SECTION, raw_text):
        part = part.strip()
        if not part or _is_consultation_block(part):
            continue
        head = part[:200].lower()
        if "ngày giờ diễn biến bệnh" in head or "diễn biến bệnh y lệnh" in head:
            sections.append(
                ClinicalSection(text=part, kind="dien_bien", block_index=-1)
            )

    return sections


def _parse_section_date(section: str) -> str | None:
    match = RE_SECTION_DATE.search(section)
    if match:
        parsed = parse_vietnamese_date_fragment(match.group(1))
        if parsed:
            return parsed.isoformat()

    for match in RE_NEARBY_DATE.finditer(section[:500]):
        parsed = parse_vietnamese_date_fragment(match.group(1))
        if parsed:
            return parsed.isoformat()
    return None


def _detect_lab_triggers(labs_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []

    inr = labs_df[labs_df["lab_type"] == "pt_inr"]
    for _, row in inr[inr["value"] > config.TRIGGER_INR_THRESHOLD].iterrows():
        rows.append(
            {
                "patient_record": _normalize_patient_record(row["patient_record"]),
                "trigger_type": "inr_gt_5",
                "date": _format_date(row["date_return"]),
                "value": row["value"],
                "baseline": pd.NA,
                "source_file": row.get("source_file", pd.NA),
            }
        )

    aptt = labs_df[labs_df["lab_type"] == "aptt"]
    for _, row in aptt[aptt["value"] > config.TRIGGER_APTT_THRESHOLD_SEC].iterrows():
        rows.append(
            {
                "patient_record": _normalize_patient_record(row["patient_record"]),
                "trigger_type": "aptt_gt_100",
                "date": _format_date(row["date_return"]),
                "value": row["value"],
                "baseline": pd.NA,
                "source_file": row.get("source_file", pd.NA),
            }
        )

    for lab_type in ("hct", "hgb"):
        subset = labs_df[labs_df["lab_type"] == lab_type].copy()
        subset["patient_record"] = subset["patient_record"].map(_normalize_patient_record)
        for patient_record, group in subset.groupby("patient_record"):
            ordered = group.sort_values("date_return")
            for idx in range(1, len(ordered)):
                baseline_val = ordered.iloc[idx - 1]["value"]
                current_val = ordered.iloc[idx]["value"]
                if pd.isna(baseline_val) or pd.isna(current_val) or baseline_val <= 0:
                    continue
                drop_fraction = (baseline_val - current_val) / baseline_val
                if drop_fraction >= config.TRIGGER_HCT_HGB_DROP_FRACTION:
                    trigger_name = f"{lab_type}_drop_ge_25pct"
                    rows.append(
                        {
                            "patient_record": patient_record,
                            "trigger_type": trigger_name,
                            "date": _format_date(ordered.iloc[idx]["date_return"]),
                            "value": current_val,
                            "baseline": baseline_val,
                            "source_file": ordered.iloc[idx].get("source_file", pd.NA),
                        }
                    )

    return rows


def _detect_doac_stop_triggers(
    raw_text: str,
    patient_record: str,
    source_file: str,
) -> list[dict]:
    rows: list[dict] = []
    for clinical_section in _split_bleeding_search_sections(raw_text):
        section = clinical_section.text
        if RE_DISCHARGE_BOILERPLATE.search(section):
            continue

        norm_section = normalize(section.replace("\n", " "))
        for doac_match in RE_DOAC_NAME.finditer(norm_section):
            start = max(0, doac_match.start() - DOAC_STOP_WINDOW)
            end = min(len(norm_section), doac_match.end() + DOAC_STOP_WINDOW)
            window = norm_section[start:end]
            stop_match = RE_DOAC_STOP.search(window)
            if not stop_match:
                continue

            section_date = _parse_section_date(section)
            rows.append(
                {
                    "patient_record": patient_record,
                    "trigger_type": "doac_sudden_stop",
                    "date": section_date or pd.NA,
                    "value": doac_match.group(1),
                    "baseline": pd.NA,
                    "source_file": source_file,
                }
            )
            break

    return rows


def _apply_doac_timing_filter(
    event_class: str,
    history_reason: str,
    date_encounter: str | None,
    patient_record: str,
    doac_start_dates: dict[str, pd.Timestamp],
) -> tuple[str, str]:
    if event_class != "bleeding_event":
        return event_class, history_reason
    doac_start = doac_start_dates.get(patient_record)
    if doac_start is None or not date_encounter:
        return event_class, history_reason
    encounter = pd.to_datetime(date_encounter, errors="coerce")
    if pd.isna(encounter) or encounter >= doac_start:
        return event_class, history_reason
    return "history_bleeding", "pre_doac"


def _detect_bleeding_events(
    raw_text: str,
    patient_record: str,
    source_file: str,
    date_admission: str | None,
    doac_start_dates: dict[str, pd.Timestamp] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    fallback_date = str(date_admission) if date_admission and str(date_admission) != "nan" else None
    doac_start_dates = doac_start_dates or {}

    for clinical_section in _split_bleeding_search_sections(raw_text):
        section = clinical_section.text
        section_date = _parse_section_date(section) or fallback_date
        narrative = _clinical_narrative_portion(section)
        if not narrative.strip():
            continue

        searchable = narrative.casefold()

        for keyword in BLEEDING_KEYWORDS:
            _collect_keyword_events(
                rows,
                narrative,
                searchable,
                keyword,
                patient_record,
                source_file,
                section_date,
                clinical_section,
                date_admission,
                doac_start_dates,
            )

        for pattern, label in _BLEEDING_INTERVENTION_RE:
            for match in pattern.finditer(searchable):
                if _should_skip_bleeding_match(searchable, match.start(), match.end()):
                    continue
                event_class, history_reason = _classify_bleeding_hit(
                    searchable,
                    match.start(),
                    match.end(),
                    label,
                    clinical_section,
                    section_date,
                    date_admission,
                )
                event_class, history_reason = _apply_doac_timing_filter(
                    event_class,
                    history_reason,
                    section_date,
                    patient_record,
                    doac_start_dates,
                )
                rows.append(
                    _bleeding_event_row(
                        patient_record,
                        source_file,
                        section_date,
                        label,
                        narrative,
                        match.start(),
                        len(match.group(0)),
                        event_class=event_class,
                        history_reason=history_reason,
                    )
                )

    return rows


def _clinical_narrative_portion(section: str) -> str:
    """Diễn biến clinical text before y lệnh tables and nursing care-plan blocks."""
    lower = section.casefold()
    cut = len(section)
    for marker in BLEEDING_NARRATIVE_CUTOFF_MARKERS:
        pos = lower.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return section[:cut]


def _collect_keyword_events(
    rows: list[dict],
    raw_section: str,
    searchable: str,
    keyword: str,
    patient_record: str,
    source_file: str,
    section_date: str | None,
    clinical_section: ClinicalSection,
    date_admission: str | None,
    doac_start_dates: dict[str, pd.Timestamp],
) -> None:
    start = 0
    key = keyword.casefold()
    while True:
        pos = searchable.find(key, start)
        if pos == -1:
            break
        if not _should_skip_bleeding_match(searchable, pos, pos + len(key)):
            event_class, history_reason = _classify_bleeding_hit(
                searchable,
                pos,
                pos + len(key),
                keyword,
                clinical_section,
                section_date,
                date_admission,
            )
            event_class, history_reason = _apply_doac_timing_filter(
                event_class,
                history_reason,
                section_date,
                patient_record,
                doac_start_dates,
            )
            rows.append(
                _bleeding_event_row(
                    patient_record,
                    source_file,
                    section_date,
                    keyword,
                    raw_section,
                    pos,
                    len(key),
                    event_class=event_class,
                    history_reason=history_reason,
                )
            )
        start = pos + len(key)


def _is_procedure_context(searchable: str, start: int, end: int) -> bool:
    window_start = max(0, start - BLEEDING_CONTEXT_RADIUS)
    window_end = min(len(searchable), end + BLEEDING_CONTEXT_RADIUS)
    window = searchable[window_start:window_end]
    return any(pattern.search(window) for pattern in _BLEEDING_PROCEDURE_RE)


def _is_early_section(
    clinical_section: ClinicalSection,
    section_date: str | None,
    date_admission: str | None,
) -> bool:
    if (
        clinical_section.kind == "to_dieu_tri"
        and clinical_section.block_index >= 0
        and clinical_section.block_index < EARLY_TREATMENT_BLOCK_LIMIT
    ):
        return True
    if not section_date or not date_admission:
        return False
    parsed_section = pd.to_datetime(section_date, errors="coerce")
    parsed_admission = pd.to_datetime(date_admission, errors="coerce")
    if pd.isna(parsed_section) or pd.isna(parsed_admission):
        return False
    return (parsed_section - parsed_admission).days <= EARLY_ADMISSION_DAYS


def _history_zone_reason(
    searchable: str,
    match_start: int,
    keyword: str,
) -> str | None:
    last_marker: str | None = None
    last_pos = -1
    for marker in HISTORY_ZONE_MARKERS:
        pos = searchable.rfind(marker, 0, match_start)
        if pos > last_pos:
            last_pos = pos
            last_marker = marker

    vao_vien_pos = searchable.rfind("vào viện vì", 0, match_start)
    if vao_vien_pos >= 0:
        span_end = match_start
        for marker in HISTORY_ZONE_MARKERS:
            next_pos = searchable.find(marker, vao_vien_pos + len("vào viện vì"))
            if next_pos != -1 and next_pos < span_end:
                span_end = next_pos
        admission_span = searchable[vao_vien_pos:span_end]
        if keyword.casefold() in admission_span and vao_vien_pos > last_pos:
            last_pos = vao_vien_pos
            last_marker = "vào viện vì"

    if last_marker is None or last_pos < 0:
        return None

    marker_key = last_marker.replace(" ", "_")
    return f"history_zone:{marker_key}"


def _parse_admission_date(date_admission: str | None):
    if not date_admission or str(date_admission) == "nan":
        return None
    parsed = pd.to_datetime(date_admission, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _historical_before_admission_reason(
    searchable: str,
    start: int,
    end: int,
    date_admission: str | None,
) -> str | None:
    admission = _parse_admission_date(date_admission)
    if admission is None:
        return None

    window_start = max(0, start - HISTORY_YEAR_RADIUS)
    window_end = min(len(searchable), end + HISTORY_YEAR_RADIUS)
    window = searchable[window_start:window_end]

    date_spans: list[tuple[int, int]] = []
    for match in RE_NEARBY_DATE.finditer(window):
        date_spans.append((match.start(), match.end()))
        parsed = parse_vietnamese_date_fragment(match.group(1))
        if parsed is not None and parsed < admission:
            return f"before_admission:{parsed.isoformat()}"

    for match in RE_HISTORICAL_YEAR.finditer(window):
        span_start, span_end = match.start(), match.end()
        if any(span_start < date_end and span_end > date_start for date_start, date_end in date_spans):
            continue
        year = int(match.group(1))
        if not (PLAUSIBLE_CALENDAR_YEAR_MIN <= year <= PLAUSIBLE_CALENDAR_YEAR_MAX):
            continue
        if year < admission.year:
            return f"year_{year}"
    return None


def _classify_bleeding_hit(
    searchable: str,
    start: int,
    end: int,
    trigger_type: str,
    clinical_section: ClinicalSection,
    section_date: str | None,
    date_admission: str | None,
) -> tuple[str, str]:
    if (
        trigger_type in BLEEDING_PROCEDURE_KEYWORDS
        and _is_procedure_context(searchable, start, end)
    ):
        return "procedure", ""

    history_reason = _historical_before_admission_reason(
        searchable, start, end, date_admission
    )
    if history_reason:
        return "history_bleeding", history_reason

    if _is_early_section(clinical_section, section_date, date_admission):
        zone_reason = _history_zone_reason(searchable, start, trigger_type)
        if zone_reason:
            return "history_bleeding", zone_reason

    return "bleeding_event", ""


def _should_skip_bleeding_match(searchable: str, start: int, end: int) -> bool:
    window_start = max(0, start - BLEEDING_NEGATION_RADIUS)
    window_end = min(len(searchable), end + BLEEDING_NEGATION_RADIUS)
    window = searchable[window_start:window_end]

    for pattern in _BLEEDING_NEGATION_RE:
        if pattern.search(window):
            return True
    for pattern in _BLEEDING_EXCLUDE_RE:
        if pattern.search(window):
            return True
    return False


def _bleeding_event_row(
    patient_record: str,
    source_file: str,
    date_encounter: str | None,
    trigger_type: str,
    raw_section: str,
    match_start: int,
    match_len: int,
    *,
    event_class: str = "bleeding_event",
    history_reason: str = "",
) -> dict:
    context = _extract_raw_context(raw_section, match_start, match_len)
    return {
        "patient_record": patient_record,
        "source_file": source_file,
        "date_encounter": date_encounter or pd.NA,
        "trigger_type": trigger_type,
        "source_context": context,
        "event_class": event_class,
        "history_reason": history_reason or pd.NA,
    }


def _extract_raw_context(raw_section: str, match_start: int, match_len: int) -> str:
    """Map casefold match offsets to raw text and return ±BLEEDING_CONTEXT_RADIUS."""
    raw_lower = raw_section.casefold()
    if len(raw_lower) != len(raw_section):
        # Fallback when case mapping shifts length (unlikely for Vietnamese text).
        start = max(0, match_start - BLEEDING_CONTEXT_RADIUS)
        end = min(len(raw_section), match_start + match_len + BLEEDING_CONTEXT_RADIUS)
        return re.sub(r"\s+", " ", raw_section[start:end]).strip()

    start = max(0, match_start - BLEEDING_CONTEXT_RADIUS)
    end = min(len(raw_section), match_start + match_len + BLEEDING_CONTEXT_RADIUS)
    return re.sub(r"\s+", " ", raw_section[start:end]).strip()


def _normalize_bleeding_context(context: object) -> str:
    text = str(context).casefold()
    text = re.sub(r"<<[^>]+>>", "", text)
    text = re.sub(config.RE_PDF_ARTIFACT, "", text)
    return re.sub(r"\s+", " ", text).strip()


def _finalize_bleeding_events(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "patient_record",
        "source_file",
        "date_encounter",
        "trigger_type",
        "source_context",
        "event_class",
        "history_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["_context_norm"] = df["source_context"].map(_normalize_bleeding_context)

    df = df.drop_duplicates(
        subset=["source_file", "trigger_type", "_context_norm", "event_class"]
    )

    return df.drop(columns=["_context_norm"]).reindex(columns=columns)


def _build_discordance_analysis(
    patients_df: pd.DataFrame,
    triggers_df: pd.DataFrame,
    bleeding_df: pd.DataFrame,
) -> pd.DataFrame:
    kidney_df = _load_kidney_summary()

    trigger_patients = {
        _normalize_patient_record(v) for v in triggers_df["patient_record"].astype(str)
    }
    acute_bleeding = bleeding_df[bleeding_df["event_class"] == "bleeding_event"]
    bleeding_patients = {
        _normalize_patient_record(v) for v in acute_bleeding["patient_record"].astype(str)
    }

    rows: list[dict] = []
    for _, patient in patients_df.iterrows():
        patient_record = _normalize_patient_record(patient["patient_record"])
        kidney = kidney_df.get(patient_record, {})
        ecrcl = kidney.get("ecrcl_cg_ml_min", pd.NA)
        egfr_indexed = kidney.get("egfr_2009_indexed", pd.NA)
        egfr_deindexed = kidney.get("egfr_2009_absolute", pd.NA)
        distance_indexed = pd.NA
        distance_deindexed = pd.NA
        if pd.notna(ecrcl) and pd.notna(egfr_indexed):
            distance_indexed = float(ecrcl) - float(egfr_indexed)
        if pd.notna(ecrcl) and pd.notna(egfr_deindexed):
            distance_deindexed = float(ecrcl) - float(egfr_deindexed)

        rows.append(
            {
                "patient_record": patient_record,
                "patient_id": patient.get("patient_id", pd.NA),
                "source_file": patient.get("source_file", pd.NA),
                "bleeding_event": (
                    "yes" if patient_record in bleeding_patients else "no"
                ),
                "high_risk_trigger": (
                    "yes" if patient_record in trigger_patients else "no"
                ),
                "ecrcl_cg_ml_min": ecrcl,
                "egfr_2009_indexed": egfr_indexed,
                "egfr_2009_absolute": egfr_deindexed,
                "ecrcl_egfr_indexed_distance": distance_indexed,
                "ecrcl_egfr_deindexed_distance": distance_deindexed,
                "ecrcl_recommended_dose": kidney.get("ecrcl_recommended_dose", pd.NA),
                "egfr_recommended_dose": kidney.get("egfr_recommended_dose", pd.NA),
                "discordant": kidney.get("discordant", pd.NA),
                "has_pgp_inhibitor": kidney.get("has_pgp_inhibitor", pd.NA),
                "has_cyp3a4_inhibitor": kidney.get("has_cyp3a4_inhibitor", pd.NA),
                "doac_prescribed_dose": kidney.get("doac_prescribed_dose", pd.NA),
                "dose_mismatch_ecrcl": kidney.get("dose_mismatch_ecrcl", pd.NA),
                "dose_mismatch_egfr": kidney.get("dose_mismatch_egfr", pd.NA),
                "doac_drug": patient.get("doac_drug", kidney.get("doac_drug", pd.NA)),
                "age": patient.get("age", kidney.get("age", pd.NA)),
                "sex": patient.get("sex", kidney.get("sex", pd.NA)),
                "weight_kg": kidney.get("weight_kg", pd.NA),
                "excluded": patient.get("excluded", pd.NA),
            }
        )

    column_order = [
        "patient_record",
        "patient_id",
        "source_file",
        "bleeding_event",
        "high_risk_trigger",
        "ecrcl_cg_ml_min",
        "egfr_2009_indexed",
        "egfr_2009_absolute",
        "ecrcl_egfr_indexed_distance",
        "ecrcl_egfr_deindexed_distance",
        "ecrcl_recommended_dose",
        "egfr_recommended_dose",
        "discordant",
        "has_pgp_inhibitor",
        "has_cyp3a4_inhibitor",
        "doac_prescribed_dose",
        "dose_mismatch_ecrcl",
        "dose_mismatch_egfr",
        "doac_drug",
        "age",
        "sex",
        "weight_kg",
        "excluded",
    ]
    return pd.DataFrame(rows, columns=column_order)


def _load_admission_kidney(patients_df: pd.DataFrame) -> pd.DataFrame:
    if not config.KIDNEY_FUNCTION_CSV.exists():
        return pd.DataFrame()

    kidney = pd.read_csv(config.KIDNEY_FUNCTION_CSV)
    kidney["patient_record"] = kidney["patient_record"].map(_normalize_patient_record)
    kidney["date_return"] = pd.to_datetime(kidney["date_return"], errors="coerce")
    kidney = kidney.sort_values("date_return")
    first = kidney.groupby("patient_record", as_index=False).first()

    excluded_ids = {
        _normalize_patient_record(v)
        for v in patients_df.loc[patients_df["excluded"] == "yes", "patient_record"]
    }
    return first[~first["patient_record"].isin(excluded_ids)].copy()


def _discordance_type_group(ecrcl: object, egfr_deindexed: object) -> str | None:
    if pd.isna(ecrcl) or pd.isna(egfr_deindexed):
        return None
    distance = float(ecrcl) - float(egfr_deindexed)
    if distance < 0:
        return "under"
    if distance > 0:
        return "over"
    return None


def _is_evaluable_dose_row(row: pd.Series, metric: str) -> bool:
    prescribed = row.get("doac_prescribed_dose")
    if prescribed is None or pd.isna(prescribed) or str(prescribed).strip() == "":
        return False
    if str(prescribed).strip() in config.DOSE_FLAGS_EXCLUDED_FROM_EXPORT:
        return False
    if str(row.get("doac_dose_flag", "")).strip() in config.DOSE_FLAGS_EXCLUDED_FROM_EXPORT:
        return False

    recommended_col = (
        "ecrcl_recommended_dose" if metric == "ecrcl" else "egfr_recommended_dose"
    )
    contra_col = (
        "ecrcl_contraindicated" if metric == "ecrcl" else "egfr_contraindicated"
    )
    recommended = row.get(recommended_col)
    if recommended is None or pd.isna(recommended) or str(recommended).strip() == "":
        return False
    if str(row.get(contra_col, "")).strip() == "yes":
        return False
    if str(recommended).strip() == "contraindicated":
        return False
    return True


def _build_dose_discordance_exports(
    patients_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        "source_file",
        "drug",
        "actual_dose",
        "recommended_dose",
        "eCrCL",
        "cyp3a4_inhibitor",
        "p_gp_inhibitor",
        "eGFR",
        "eGFR_deindexed",
        "discordance_type_group",
    ]
    kidney = _load_admission_kidney(patients_df)
    if kidney.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, _empty_dose_discordance_summary()

    rows: list[dict] = []
    for _, row in kidney.iterrows():
        ecrcl = pd.to_numeric(row.get("ecrcl_cg_ml_min"), errors="coerce")
        egfr_indexed = pd.to_numeric(row.get("egfr_2009_indexed"), errors="coerce")
        egfr_deindexed = pd.to_numeric(row.get("egfr_2009_absolute"), errors="coerce")
        group = _discordance_type_group(ecrcl, egfr_deindexed)
        if group is None:
            continue

        rows.append(
            {
                "source_file": row.get("source_file", pd.NA),
                "drug": row.get("doac_drug", pd.NA),
                "actual_dose": row.get("doac_prescribed_dose", pd.NA),
                "recommended_dose": row.get("ecrcl_recommended_dose", pd.NA),
                "eCrCL": ecrcl if pd.notna(ecrcl) else pd.NA,
                "cyp3a4_inhibitor": row.get("has_cyp3a4_inhibitor", pd.NA),
                "p_gp_inhibitor": row.get("has_pgp_inhibitor", pd.NA),
                "eGFR": egfr_indexed if pd.notna(egfr_indexed) else pd.NA,
                "eGFR_deindexed": egfr_deindexed if pd.notna(egfr_deindexed) else pd.NA,
                "discordance_type_group": group,
                "patient_record": row["patient_record"],
                "dose_mismatch_ecrcl": row.get("dose_mismatch_ecrcl", pd.NA),
                "dose_mismatch_egfr": row.get("dose_mismatch_egfr", pd.NA),
            }
        )

    if not rows:
        empty = pd.DataFrame(columns=columns)
        return empty, _empty_dose_discordance_summary()

    detail = pd.DataFrame(rows)
    export_df = detail.reindex(columns=columns)
    summary = _build_dose_discordance_summary(detail)
    return export_df, summary


def _empty_dose_discordance_summary() -> dict:
    return {
        "cohort": "non_excluded",
        "dose_discordance_definition": (
            "actual_dose != EHRA recommended_dose (ecrcl_recommended_dose track)"
        ),
        "ecrcl_egfr_group_definition": {
            "under": "eCrCL < eGFR_deindexed (distance < 0)",
            "over": "eCrCL > eGFR_deindexed (distance > 0)",
        },
        "under": _dose_discordance_group_summary(pd.DataFrame()),
        "over": _dose_discordance_group_summary(pd.DataFrame()),
    }


def _build_dose_discordance_summary(detail: pd.DataFrame) -> dict:
    return {
        "cohort": "non_excluded",
        "dose_discordance_definition": (
            "actual_dose != EHRA recommended_dose (ecrcl_recommended_dose track)"
        ),
        "egfr_dose_discordance_definition": (
            "actual_dose != EHRA recommended_dose (egfr_recommended_dose track)"
        ),
        "ecrcl_egfr_group_definition": {
            "under": "eCrCL < eGFR_deindexed (distance < 0)",
            "over": "eCrCL > eGFR_deindexed (distance > 0)",
        },
        "under": _dose_discordance_group_summary(
            detail[detail["discordance_type_group"] == "under"]
        ),
        "over": _dose_discordance_group_summary(
            detail[detail["discordance_type_group"] == "over"]
        ),
    }


def _is_evaluable_dose_detail(row: pd.Series, metric: str) -> bool:
    actual = row.get("actual_dose")
    recommended = row.get("recommended_dose")
    if actual is None or pd.isna(actual) or str(actual).strip() == "":
        return False
    if str(actual).strip() in config.DOSE_FLAGS_EXCLUDED_FROM_EXPORT:
        return False
    if recommended is None or pd.isna(recommended) or str(recommended).strip() == "":
        return False
    if str(recommended).strip() == "contraindicated":
        return False
    mismatch_col = "dose_mismatch_ecrcl" if metric == "ecrcl" else "dose_mismatch_egfr"
    mismatch = row.get(mismatch_col)
    return mismatch is not None and not pd.isna(mismatch) and str(mismatch).strip() != ""


def _dose_discordance_group_summary(group_df: pd.DataFrame) -> dict:
    if group_df.empty:
        return {
            "admissions": 0,
            "evaluable_ecrcl_dose": {"admissions": 0, "dose_discordant": 0, "pct": 0.0},
            "evaluable_egfr_dose": {"admissions": 0, "dose_discordant": 0, "pct": 0.0},
            "doac_pct": {},
        }

    admissions = int(group_df["patient_record"].nunique())
    ecrcl_eval = group_df[
        group_df.apply(lambda row: _is_evaluable_dose_detail(row, "ecrcl"), axis=1)
    ]
    egfr_eval = group_df[
        group_df.apply(lambda row: _is_evaluable_dose_detail(row, "egfr"), axis=1)
    ]

    ecrcl_disc = int((ecrcl_eval["dose_mismatch_ecrcl"] == "yes").sum())
    egfr_disc = int((egfr_eval["dose_mismatch_egfr"] == "yes").sum())

    return {
        "admissions": admissions,
        "evaluable_ecrcl_dose": {
            "admissions": int(len(ecrcl_eval)),
            "dose_discordant": ecrcl_disc,
            "pct": _pct(ecrcl_disc, len(ecrcl_eval)),
        },
        "evaluable_egfr_dose": {
            "admissions": int(len(egfr_eval)),
            "dose_discordant": egfr_disc,
            "pct": _pct(egfr_disc, len(egfr_eval)),
        },
        "doac_pct": _doac_distribution_pct(group_df.rename(columns={"drug": "doac_drug"})),
    }


def _write_dose_discordance_checkpoint(summary: dict) -> None:
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with config.CHECKPOINT_DOSE_DISCORDANCE_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_kidney_summary() -> dict[str, dict]:
    if not config.KIDNEY_FUNCTION_CSV.exists():
        return {}

    kidney = pd.read_csv(config.KIDNEY_FUNCTION_CSV)
    kidney["date_return"] = pd.to_datetime(kidney["date_return"], errors="coerce")
    kidney = kidney.sort_values("date_return")

    summary: dict[str, dict] = {}
    kidney["patient_record"] = kidney["patient_record"].map(_normalize_patient_record)
    for patient_record, group in kidney.groupby("patient_record"):
        row = group.iloc[0]
        summary[patient_record] = {
            "ecrcl_cg_ml_min": row.get("ecrcl_cg_ml_min", pd.NA),
            "egfr_2009_indexed": row.get("egfr_2009_indexed", pd.NA),
            "egfr_2009_absolute": row.get("egfr_2009_absolute", pd.NA),
            "ecrcl_recommended_dose": row.get("ecrcl_recommended_dose", pd.NA),
            "egfr_recommended_dose": row.get("egfr_recommended_dose", pd.NA),
            "discordant": row.get("discordant", pd.NA),
            "has_pgp_inhibitor": row.get("has_pgp_inhibitor", pd.NA),
            "has_cyp3a4_inhibitor": row.get("has_cyp3a4_inhibitor", pd.NA),
            "doac_prescribed_dose": row.get("doac_prescribed_dose", pd.NA),
            "dose_mismatch_ecrcl": row.get("dose_mismatch_ecrcl", pd.NA),
            "dose_mismatch_egfr": row.get("dose_mismatch_egfr", pd.NA),
            "doac_drug": row.get("doac_drug", pd.NA),
            "age": row.get("age", pd.NA),
            "sex": row.get("sex", pd.NA),
            "weight_kg": row.get("weight_kg", pd.NA),
        }
    return summary


def _format_date(value: object) -> str | object:
    if pd.isna(value):
        return pd.NA
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NA
    return parsed.date().isoformat()


def _build_cohort_summary(
    triggers_df: pd.DataFrame,
    bleeding_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> dict:
    cohort = analysis_df[analysis_df["excluded"] == "no"].copy()
    n_cohort = int(cohort["patient_record"].nunique())
    bleeding_mask = cohort["bleeding_event"] == "yes"
    trigger_mask = cohort["high_risk_trigger"] == "yes"
    n_bleeding = int(cohort[bleeding_mask]["patient_record"].nunique())
    n_trigger = int(cohort[trigger_mask]["patient_record"].nunique())
    n_both = int(
        cohort[bleeding_mask & trigger_mask]["patient_record"].nunique()
    )

    trigger_types = (
        triggers_df["trigger_type"].value_counts().to_dict()
        if not triggers_df.empty
        else {}
    )

    acute_bleeding_df = bleeding_df[bleeding_df["event_class"] == "bleeding_event"]
    history_bleeding_df = bleeding_df[bleeding_df["event_class"] == "history_bleeding"]
    procedure_df = bleeding_df[bleeding_df["event_class"] == "procedure"]

    return {
        "cohort": "non_excluded",
        "admissions": n_cohort,
        "outcomes": {
            "bleeding_event": {
                "admissions": n_bleeding,
                "pct": _pct(n_bleeding, n_cohort),
            },
            "high_risk_trigger": {
                "admissions": n_trigger,
                "pct": _pct(n_trigger, n_cohort),
            },
            "bleeding_and_trigger": {
                "admissions": n_both,
                "pct": _pct(n_both, n_cohort),
            },
        },
        "checkpoint_rows": {
            "trigger_rows": int(len(triggers_df)),
            "bleeding_event_rows": int(len(acute_bleeding_df)),
            "history_bleeding_rows": int(len(history_bleeding_df)),
            "procedure_rows": int(len(procedure_df)),
            "total_bleeding_checkpoint_rows": int(len(bleeding_df)),
        },
        "trigger_types": {key: int(value) for key, value in trigger_types.items()},
        "ecrcl_egfr_discordance": _build_ecrcl_egfr_discordance_summary(
            cohort,
            triggers_df,
        ),
    }


def _build_ecrcl_egfr_pair_summary(
    cohort: pd.DataFrame,
    triggers_df: pd.DataFrame,
    *,
    egfr_col: str,
    distance_definition: str,
    under_definition: str,
    over_definition: str,
) -> dict:
    kidney = cohort.copy()
    kidney["ecrcl_cg_ml_min"] = pd.to_numeric(kidney["ecrcl_cg_ml_min"], errors="coerce")
    kidney[egfr_col] = pd.to_numeric(kidney[egfr_col], errors="coerce")
    kidney["ecrcl_egfr_distance"] = kidney["ecrcl_cg_ml_min"] - kidney[egfr_col]

    both = kidney[kidney["ecrcl_cg_ml_min"].notna() & kidney[egfr_col].notna()]
    equal = both[both["ecrcl_egfr_distance"] == 0]

    return {
        "distance_definition": distance_definition,
        "admissions_with_both_metrics": int(both["patient_record"].nunique()),
        "admissions_equal_ecrcl_egfr": int(equal["patient_record"].nunique()),
        "under": _ecrcl_egfr_group_summary(
            both[both["ecrcl_egfr_distance"] < 0],
            triggers_df,
            group_type="under",
            definition=under_definition,
        ),
        "over": _ecrcl_egfr_group_summary(
            both[both["ecrcl_egfr_distance"] > 0],
            triggers_df,
            group_type="over",
            definition=over_definition,
        ),
    }


def _build_ecrcl_egfr_discordance_summary(
    cohort: pd.DataFrame,
    triggers_df: pd.DataFrame,
) -> dict:
    return {
        "ecrcl_vs_egfr_indexed": _build_ecrcl_egfr_pair_summary(
            cohort,
            triggers_df,
            egfr_col="egfr_2009_indexed",
            distance_definition=(
                "ecrcl_cg_ml_min - egfr_2009_indexed (mL/min vs mL/min/1.73 m²)"
            ),
            under_definition="eCrCL < eGFR (distance < 0)",
            over_definition="eCrCL > eGFR (distance > 0)",
        ),
        "ecrcl_vs_egfr_deindexed": _build_ecrcl_egfr_pair_summary(
            cohort,
            triggers_df,
            egfr_col="egfr_2009_absolute",
            distance_definition="ecrcl_cg_ml_min - egfr_2009_absolute (mL/min)",
            under_definition="eCrCL < eGFR_deindexed (distance < 0)",
            over_definition="eCrCL > eGFR_deindexed (distance > 0)",
        ),
    }


def _ecrcl_egfr_group_summary(
    group_df: pd.DataFrame,
    triggers_df: pd.DataFrame,
    group_type: str,
    definition: str,
) -> dict:
    admissions = int(group_df["patient_record"].nunique())
    bleeding_admissions = int(
        group_df[group_df["bleeding_event"] == "yes"]["patient_record"].nunique()
    )

    patient_records = set(group_df["patient_record"].astype(str))
    group_triggers = triggers_df[
        triggers_df["patient_record"].astype(str).isin(patient_records)
    ]

    trigger_type_admissions: dict[str, int] = {}
    if not group_triggers.empty:
        for trigger_type, subset in group_triggers.groupby("trigger_type"):
            trigger_type_admissions[str(trigger_type)] = int(
                subset["patient_record"].nunique()
            )

    doac_pct = _doac_distribution_pct(group_df)

    return {
        "type": group_type,
        "definition": definition,
        "admissions": admissions,
        "bleeding_admissions": bleeding_admissions,
        "bleeding_pct": _pct(bleeding_admissions, admissions),
        "trigger_types": trigger_type_admissions,
        "doac_pct": doac_pct,
    }


def _doac_distribution_pct(group_df: pd.DataFrame) -> dict[str, float]:
    drugs = group_df["doac_drug"].dropna()
    drugs = drugs[drugs.astype(str).str.strip() != ""]
    total = int(len(group_df))
    if total == 0:
        return {}

    counts = drugs.value_counts()
    return {
        str(drug): _pct(int(count), total)
        for drug, count in counts.items()
    }


def _pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * part / total, 1)


def _write_cohort_checkpoint(summary: dict) -> None:
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with config.CHECKPOINT_BLEEDING_COHORT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _print_summary(summary: dict, dose_summary: dict | None = None) -> None:
    print("Bleeding detection summary (non-excluded cohort):")
    print(f"  Admissions: {summary['admissions']}")
    outcomes = summary["outcomes"]
    print(f"  Bleeding events: {outcomes['bleeding_event']['admissions']} "
          f"({outcomes['bleeding_event']['pct']:.1f}%)")
    print(f"  High-risk triggers: {outcomes['high_risk_trigger']['admissions']}")
    print(f"  Both bleeding + trigger: {outcomes['bleeding_and_trigger']['admissions']}")
    rows = summary["checkpoint_rows"]
    print(f"  Trigger rows exported: {rows['trigger_rows']}")
    print(f"  Acute bleeding rows: {rows['bleeding_event_rows']}")
    print(f"  History bleeding rows: {rows['history_bleeding_rows']}")
    print(f"  Procedure rows: {rows['procedure_rows']}")
    print(f"  Total bleeding checkpoint rows: {rows['total_bleeding_checkpoint_rows']}")

    if summary["trigger_types"]:
        print("  Trigger types:")
        for trigger_type, count in summary["trigger_types"].items():
            print(f"    {trigger_type}: {count}")

    discordance = summary["ecrcl_egfr_discordance"]
    for pair_key, pair_label in (
        ("ecrcl_vs_egfr_indexed", "eCrCL vs eGFR (indexed)"),
        ("ecrcl_vs_egfr_deindexed", "eCrCL vs eGFR (de-indexed)"),
    ):
        pair = discordance[pair_key]
        under = pair["under"]
        over = pair["over"]
        print(
            f"  {pair_label} under: {under['admissions']} admissions, "
            f"bleeding {under['bleeding_pct']:.1f}%"
        )
        print(
            f"  {pair_label} over: {over['admissions']} admissions, "
            f"bleeding {over['bleeding_pct']:.1f}%"
        )

    if dose_summary:
        dose_under = dose_summary["under"]
        dose_over = dose_summary["over"]
        print(
            f"  Dose discordance (eCrCL track) under group: "
            f"{dose_under['evaluable_ecrcl_dose']['pct']:.1f}%"
        )
        print(
            f"  Dose discordance (eCrCL track) over group: "
            f"{dose_over['evaluable_ecrcl_dose']['pct']:.1f}%"
        )
