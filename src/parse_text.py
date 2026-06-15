"""Low-level text utilities: normalization and section splitting."""

from __future__ import annotations

import re
from datetime import date

from src import config


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip PDF artifacts, fix glued tokens."""
    text = text.lower()
    text = re.sub(config.RE_PDF_ARTIFACT, "", text)
    text = re.sub(r"(\d)\s*kg\s*chiều cao", r"\1 kg chiều cao", text) # glued tokens
    text = re.sub(r"\s+", " ", text) # spaces
    return text.strip()


def split_clinical_notes(text: str) -> list[str]:
    """Split text into tờ điều trị sections."""
    parts = re.split(config.RE_CLINICAL_NOTE_SECTION, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def split_lab_reports(text: str) -> list[str]:
    """Split text into lab-report blocks by known report headers."""
    pattern = "|".join(f"(?:{h})" for h in config.RE_LAB_REPORT_HEADERS)
    parts = re.split(rf"(?i)(?={pattern})", text)
    blocks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        head = part[:400].lower()
        if not any(header in head for header in config.RE_LAB_REPORT_HEADERS):
            continue
        blocks.append(part)
    return blocks


def parse_lab_date_return(block: str) -> date | None:
    """Parse date_return from a lab-report block."""
    block = block.lower()
    for header in config.RE_LAB_DATE_RETURN:
        m = re.search(
            rf"{header}:\s*(?:\d{{1,2}}:\d{{2}}\s+)?(\d{{1,2}}/\d{{1,2}}/\d{{4}})",
            block,
        )
        if m:
            parsed = parse_vietnamese_date_fragment(m.group(1))
            if parsed:
                return parsed

    m = re.search(
        r"(\d{1,2}:\d{2})\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        block,
    )
    if m:
        day, month, year = (int(m.group(i)) for i in range(2, 5))
        return _safe_date(year, month, day)

    m = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        block,
    )
    if m:
        day, month, year = (int(m.group(i)) for i in range(1, 4))
        return _safe_date(year, month, day)

    return None


def parse_vietnamese_date_fragment(fragment: str) -> date | None:
    """Parse a Vietnamese date fragment into a date object."""
    fragment = fragment.strip().lower()

    m = re.search(
        r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
        fragment,
    )
    if m:
        day, month, year = (int(m.group(i)) for i in range(1, 4))
        return _safe_date(year, month, day)

    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", fragment)
    if m:
        day, month, year = (int(m.group(i)) for i in range(1, 4))
        return _safe_date(year, month, day)

    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2})\b", fragment)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year += 2000 if year < 100 else 0
        return _safe_date(year, month, day)

    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
