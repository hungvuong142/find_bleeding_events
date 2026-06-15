"""Extract laboratory results from admission text files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import config
from src.extract_metadata import _parse_ids
from src.parse_text import normalize, parse_lab_date_return, split_lab_reports

_KNOWN_UNITS = frozenset(
    {
        "g/l",
        "l/l",
        "giây",
        "%",
        "µmol/l",
        "mmol/l",
        "mg/l",
        "u/l",
        "ng/l",
        "t/l",
        "fl",
        "pg",
        "s",
        "gi",
    }
)


def extract_labs(text_dir: Path | None = None) -> pd.DataFrame:
    """Parse all text files and write labs_long.csv."""
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_EXPORT.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for path in sorted(text_dir.glob("*.txt")):
        records.extend(_parse_file(path))

    columns = [
        "patient_record",
        "lab_type",
        "date_return",
        "value",
        "unit",
        "reference_range",
        "raw_test_name",
        "source_file",
    ]
    if records:
        df = pd.DataFrame(records)[columns]
        df = df.drop_duplicates(
            subset=["patient_record", "lab_type", "date_return", "value"],
            keep="first",
        )
    else:
        df = pd.DataFrame(columns=columns)

    df.to_csv(config.LABS_LONG_CSV, index=False)
    return df


def _parse_file(path: Path) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    norm_text = normalize(raw_text)
    lines = raw_text.splitlines()
    _, patient_record = _parse_ids(lines, norm_text)
    if not patient_record:
        return []

    records: list[dict] = []
    for block in split_lab_reports(raw_text):
        date_return = parse_lab_date_return(normalize(block))
        date_str = date_return.isoformat() if date_return else None
        records.extend(
            _extract_from_block(
                block,
                patient_record=patient_record,
                source_file=path.name,
                date_return=date_str,
            )
        )
    return records


def _extract_from_block(
    raw_block: str,
    *,
    patient_record: str,
    source_file: str,
    date_return: str | None,
) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[str, str | None, float]] = set()
    norm_block = normalize(raw_block)

    for raw_line in raw_block.splitlines():
        line = normalize(raw_line)
        if not line:
            continue
        for lab_type, aliases in config.LAB_TYPES.items():
            for alias in aliases:
                parsed = _parse_compact_line(line, lab_type, alias)
                if parsed:
                    _append_lab(records, seen, patient_record, source_file, date_return, parsed)
                    break
                parsed = _parse_bmql_line(line, lab_type, alias)
                if parsed:
                    _append_lab(records, seen, patient_record, source_file, date_return, parsed)
                    break

    for parsed in _extract_egfr(norm_block):
        _append_lab(records, seen, patient_record, source_file, date_return, parsed)

    return records


def _append_lab(
    records: list[dict],
    seen: set[tuple[str, str | None, float]],
    patient_record: str,
    source_file: str,
    date_return: str | None,
    parsed: dict,
) -> None:
    key = (parsed["lab_type"], date_return, parsed["value"])
    if key in seen:
        return
    seen.add(key)
    records.append(
        {
            "patient_record": patient_record,
            "lab_type": parsed["lab_type"],
            "date_return": date_return,
            "value": parsed["value"],
            "unit": parsed.get("unit"),
            "reference_range": parsed.get("reference_range"),
            "raw_test_name": parsed.get("raw_test_name"),
            "source_file": source_file,
        }
    )


def _parse_compact_line(line: str, lab_type: str, alias: str) -> dict | None:
    alias_esc = re.escape(alias)
    m = re.search(
        rf"(\d+(?:\.\d+)?)\s+([\d.]+\s*-\s*[\d.]+)\s*{alias_esc}(?:\s+(\S+))?",
        line,
        re.IGNORECASE,
    )
    if m:
        unit = m.group(3)
        if unit and unit.lower() not in _KNOWN_UNITS:
            unit = _default_unit(lab_type)
        else:
            unit = unit or _default_unit(lab_type)
        return {
            "lab_type": lab_type,
            "value": float(m.group(1)),
            "reference_range": m.group(2).strip(),
            "raw_test_name": alias,
            "unit": unit,
        }

    m = re.search(rf"(\d+(?:\.\d+)?)\s{{2,}}{alias_esc}(?:\s|\b)", line, re.IGNORECASE)
    if m:
        return {
            "lab_type": lab_type,
            "value": float(m.group(1)),
            "reference_range": None,
            "raw_test_name": alias,
            "unit": _default_unit(lab_type),
        }
    return None


def _parse_bmql_line(line: str, lab_type: str, alias: str) -> dict | None:
    if alias not in line:
        return None

    if not re.match(
        r"(µmol/l|mmol/l|g/l|l/l|mg/l|u/l|ng/l|%|gi)\b",
        line,
        re.IGNORECASE,
    ):
        return None

    alias_esc = re.escape(alias)
    m = re.search(
        rf"(\S+)\s+([\d.]+\s*-\s*[\d.]+){alias_esc}.*?(\d+(?:\.\d+)?)\s+\d+\s+\S",
        line,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(
            rf"(\S+)\s+([\d.]+\s*-\s*[\d.]+){alias_esc}.*?(\d+(?:\.\d+)?)\s+\d+\b",
            line,
            re.IGNORECASE,
        )
    if m:
        return {
            "lab_type": lab_type,
            "value": float(m.group(3)),
            "reference_range": m.group(2).strip(),
            "raw_test_name": alias,
            "unit": m.group(1),
        }
    return None


def _extract_egfr(norm_block: str) -> list[dict]:
    records: list[dict] = []
    patterns = (
        r"mức lọc cầu thận ước tính \(egfr\)\s*\[ckd-epi 2009\]\s*(\d+(?:\.\d+)?)",
        r"\[ckd-epi 2009\]\s*(\d+(?:\.\d+)?)",
        r"\begfr\s*:\s*(\d+(?:\.\d+)?)\b",
        r"\begfr\s+(\d+(?:\.\d+)?)\b",
    )
    seen_values: set[float] = set()
    for pattern in patterns:
        for m in re.finditer(pattern, norm_block, re.IGNORECASE):
            value = float(m.group(1))
            if value in seen_values:
                continue
            seen_values.add(value)
            records.append(
                {
                    "lab_type": "egfr_2009_indexed",
                    "value": value,
                    "reference_range": None,
                    "raw_test_name": "egfr [ckd-epi 2009]",
                    "unit": "ml/min/1.73m2",
                }
            )
    return records


def _default_unit(lab_type: str) -> str | None:
    return {
        "creatinine": "µmol/l",
        "hgb": "g/l",
        "hct": "l/l",
        "plt": "g/l",
        "pt_inr": None,
        "pt_sec": "s",
        "aptt": "s",
    }.get(lab_type)


if __name__ == "__main__":
    result = extract_labs()
    print(f"Wrote {len(result)} lab rows to {config.LABS_LONG_CSV}")
    if not result.empty:
        print(result.groupby("lab_type").size().to_string())
