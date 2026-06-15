"""Extract weight and height measurements from admission text files."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.extract_metadata import _parse_admission_date, _parse_ids
from src.parse_text import normalize, parse_vietnamese_date_fragment


_WEIGHT_MIN_KG = 25.0
_WEIGHT_MAX_KG = 250.0
_HEIGHT_MIN_CM = 100.0
_HEIGHT_MAX_CM = 230.0

RE_INLINE_PAIR = re.compile(
    r"cân nặng:\s*(\d+(?:\.\d+)?)\s*kg\s*chiều cao:\s*(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
RE_MULTILINE_PAIR = re.compile(
    r"cân nặng:\s*\n\s*(\d+(?:\.\d+)?)\s*\n\s*kg\s*\n\s*chiều cao:\s*\n?\s*(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
RE_INLINE_WEIGHT = re.compile(r"cân nặng:\s*(\d+(?:\.\d+)?)\s*kg", re.IGNORECASE)
RE_INLINE_HEIGHT = re.compile(r"chiều cao:\s*(\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
RE_NEARBY_DATE = re.compile(r"ngày\s+(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE)


def extract_vitals(text_dir: Path | None = None) -> pd.DataFrame:
    """Parse all text files and write vitals_weight_height.csv + missing-WH checkpoint."""
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_EXPORT.mkdir(parents=True, exist_ok=True)
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for path in sorted(text_dir.glob("*.txt")):
        records.extend(_parse_file(path))

    columns = [
        "patient_record",
        "date_measured",
        "weight_kg",
        "height_cm",
        "source_context",
        "source_file",
    ]
    if records:
        df = pd.DataFrame(records)[columns]
        df = df.drop_duplicates(
            subset=[
                "patient_record",
                "date_measured",
                "weight_kg",
                "height_cm",
                "source_context",
            ],
            keep="first",
        )
    else:
        df = pd.DataFrame(columns=columns)

    df = df.loc[df['weight_kg'].between(_WEIGHT_MIN_KG, _WEIGHT_MAX_KG, inclusive='both'), :]
    df = df.loc[df['height_cm'].between(_HEIGHT_MIN_CM, _HEIGHT_MAX_CM, inclusive='both'), :]
    
    # check for inconsistent weight/height
    for group, data in df.groupby('patient_record'):
        max_weight = data['weight_kg'].max()
        min_weight = data['weight_kg'].min()
        max_height = data['height_cm'].max()
        min_height = data['height_cm'].min()
        if max_weight - min_weight > 50 or max_height - min_height > 50:
            print(f"Patient {group} has inconsistent weight/height: {min_weight} - {max_weight} kg, {min_height} - {max_height} cm")
            accepted_weight = min_weight
            accepted_height = min_height
            df.loc[data.index, 'weight_kg'] = accepted_weight
            df.loc[data.index, 'height_cm'] = accepted_height
    df = df.loc[df['patient_record'].isin(df['patient_record'].unique()), :]
    df.to_csv(config.VITALS_WH_CSV, index=False)
    _write_missing_wh_checkpoint(df)
    return df


def _parse_file(path: Path) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    norm_text = normalize(raw_text)
    lines = raw_text.splitlines()
    _, patient_record = _parse_ids(lines, norm_text)
    if not patient_record:
        return []

    admission_date = _parse_admission_date(norm_text)
    admission_str = admission_date.isoformat() if admission_date else None

    records: list[dict] = []
    covered_spans: list[tuple[int, int]] = []

    for m in RE_MULTILINE_PAIR.finditer(raw_text):
        weight, height = _normalize_pair(float(m.group(1)), float(m.group(2)))
        context = _context_snippet(raw_text, m.start(), m.end())
        records.append(
            _vital_row(
                patient_record,
                path.name,
                weight,
                height,
                _date_near_span(raw_text, m.start(), m.end()) or admission_str,
                context,
            )
        )
        covered_spans.append((m.start(), m.end()))

    for m in RE_INLINE_PAIR.finditer(norm_text):
        start = norm_text.find(m.group(0))
        if start >= 0 and _overlaps_covered(start, start + len(m.group(0)), covered_spans):
            continue
        weight, height = _normalize_pair(float(m.group(1)), float(m.group(2)))
        records.append(
            _vital_row(
                patient_record,
                path.name,
                weight,
                height,
                _date_near_span(raw_text, start, start + len(m.group(0))) or admission_str,
                m.group(0)[:120],
            )
        )

    standalone_weights = _find_standalone_weights(raw_text, norm_text, covered_spans)
    standalone_heights = _find_standalone_heights(raw_text, norm_text, covered_spans)
    records.extend(
        _pair_standalone(
            patient_record,
            path.name,
            raw_text,
            admission_str,
            standalone_weights,
            standalone_heights,
        )
    )
    return records


def _find_standalone_weights(
    raw_text: str,
    norm_text: str,
    covered_spans: list[tuple[int, int]],
) -> list[tuple[float, int, str]]:
    found: list[tuple[float, int, str]] = []
    for m in RE_INLINE_WEIGHT.finditer(norm_text):
        start = norm_text.find(m.group(0), m.start())
        if _overlaps_covered(start, start + len(m.group(0)), covered_spans):
            continue
        found.append((float(m.group(1)), start, m.group(0)[:80]))
    return found


def _find_standalone_heights(
    raw_text: str,
    norm_text: str,
    covered_spans: list[tuple[int, int]],
) -> list[tuple[float, int, str]]:
    found: list[tuple[float, int, str]] = []
    for m in RE_INLINE_HEIGHT.finditer(norm_text):
        start = norm_text.find(m.group(0), m.start())
        if _overlaps_covered(start, start + len(m.group(0)), covered_spans):
            continue
        found.append((float(m.group(1)), start, m.group(0)[:80]))
    return found


def _pair_standalone(
    patient_record: str,
    source_file: str,
    raw_text: str,
    admission_str: str | None,
    weights: list[tuple[float, int, str]],
    heights: list[tuple[float, int, str]],
) -> list[dict]:
    records: list[dict] = []
    used_heights: set[int] = set()

    for weight, w_pos, w_context in weights:
        best_idx: int | None = None
        best_dist = 10_000
        for idx, (height, h_pos, _) in enumerate(heights):
            if idx in used_heights:
                continue
            dist = abs(w_pos - h_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and best_dist <= 5000:
            height, h_pos, h_context = heights[best_idx]
            used_heights.add(best_idx)
            norm_w, norm_h = _normalize_pair(weight, height)
            span_start = min(w_pos, h_pos)
            span_end = max(w_pos + len(w_context), h_pos + len(h_context))
            records.append(
                _vital_row(
                    patient_record,
                    source_file,
                    norm_w,
                    norm_h,
                    _date_near_span(raw_text, span_start, span_end) or admission_str,
                    f"{w_context} | {h_context}",
                )
            )
        elif _is_plausible_weight(weight):
            records.append(
                _vital_row(
                    patient_record,
                    source_file,
                    weight,
                    None,
                    _date_near_span(raw_text, w_pos, w_pos + len(w_context)) or admission_str,
                    w_context,
                )
            )

    for idx, (height, h_pos, h_context) in enumerate(heights):
        if idx in used_heights:
            continue
        if _is_plausible_height(height):
            records.append(
                _vital_row(
                    patient_record,
                    source_file,
                    None,
                    height,
                    _date_near_span(raw_text, h_pos, h_pos + len(h_context)) or admission_str,
                    h_context,
                )
            )
    return records


def _vital_row(
    patient_record: str,
    source_file: str,
    weight_kg: float | None,
    height_cm: float | None,
    date_measured: str | None,
    source_context: str,
) -> dict:
    return {
        "patient_record": patient_record,
        "date_measured": date_measured,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "source_context": source_context.strip(),
        "source_file": source_file,
    }


def _normalize_pair(weight: float, height: float) -> tuple[float, float]:
    if _is_plausible_weight(weight) and _is_plausible_height(height):
        return weight, height
    if _is_plausible_weight(height) and _is_plausible_height(weight):
        return height, weight
    return weight, height


def _is_plausible_weight(value: float) -> bool:
    return _WEIGHT_MIN_KG <= value <= _WEIGHT_MAX_KG


def _is_plausible_height(value: float) -> bool:
    return _HEIGHT_MIN_CM <= value <= _HEIGHT_MAX_CM


def _overlaps_covered(start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    return any(not (end <= c_start or start >= c_end) for c_start, c_end in covered)


def _context_snippet(raw_text: str, start: int, end: int, radius: int = 60) -> str:
    snippet_start = max(0, start - radius)
    snippet_end = min(len(raw_text), end + radius)
    return " ".join(raw_text[snippet_start:snippet_end].split())


def _date_near_span(raw_text: str, start: int, end: int) -> str | None:
    window = raw_text[max(0, start - 300) : min(len(raw_text), end + 300)]
    matches = list(RE_NEARBY_DATE.finditer(window))
    if not matches:
        return None
    parsed_dates: list[date] = []
    for m in matches:
        parsed = parse_vietnamese_date_fragment(m.group(1))
        if parsed:
            parsed_dates.append(parsed)
    if not parsed_dates:
        return None
    return parsed_dates[-1].isoformat()


def _write_missing_wh_checkpoint(vitals_df: pd.DataFrame) -> None:
    if config.PATIENTS_CSV.exists():
        patients = pd.read_csv(config.PATIENTS_CSV, dtype=str)
        all_records = patients["patient_record"].dropna().unique()
    else:
        all_records = vitals_df["patient_record"].dropna().unique()

    valid_records = _records_with_valid_wh_tuple(vitals_df)
    missing = [rec for rec in all_records if rec not in valid_records]
    checkpoint = pd.DataFrame({"patient_record": missing})
    checkpoint.to_csv(config.CHECKPOINT_MISSING_WH_CSV, index=False)


def _records_with_valid_wh_tuple(vitals_df: pd.DataFrame) -> set[str]:
    valid: set[str] = set()
    if vitals_df.empty:
        return valid

    for patient_record, group in vitals_df.groupby("patient_record"):
        weights = [
            float(v)
            for v in group["weight_kg"].dropna()
            if _is_plausible_weight(float(v))
        ]
        heights = [
            float(v)
            for v in group["height_cm"].dropna()
            if _is_plausible_height(float(v))
        ]

        for weight in weights:
            for height in heights:
                norm_w, norm_h = _normalize_pair(weight, height)
                if _is_plausible_weight(norm_w) and _is_plausible_height(norm_h):
                    valid.add(str(patient_record))
                    break
            if str(patient_record) in valid:
                break

        for _, row in group.iterrows():
            weight = row.get("weight_kg")
            height = row.get("height_cm")
            if pd.isna(weight) or pd.isna(height):
                continue
            norm_w, norm_h = _normalize_pair(float(weight), float(height))
            if _is_plausible_weight(norm_w) and _is_plausible_height(norm_h):
                valid.add(str(patient_record))
                break

    return valid


if __name__ == "__main__":
    result = extract_vitals()
    print(f"Wrote {len(result)} vitals rows to {config.VITALS_WH_CSV}")
    if config.CHECKPOINT_MISSING_WH_CSV.exists():
        missing = pd.read_csv(config.CHECKPOINT_MISSING_WH_CSV)
        print(f"Missing weight/height checkpoint: {len(missing)} records")
