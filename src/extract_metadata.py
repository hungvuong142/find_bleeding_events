"""Extract patient metadata from admission text files."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.parse_text import normalize, parse_vietnamese_date_fragment


def extract_patient_metadata(text_dir: Path | None = None) -> pd.DataFrame:
    """Parse all text files and write patients.csv + exclusion checkpoint."""
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_EXPORT.mkdir(parents=True, exist_ok=True)
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for path in sorted(text_dir.glob("*.txt")):
        records.append(_parse_file(path))

    df = pd.DataFrame(records)
    column_order = [
        "source_file",
        "patient_id",
        "patient_record",
        "age",
        "sex",
        "date_admission",
        "date_discharge",
        "doac_drug",
        "excluded",
        "exclusion_reason",
        "pregnancy_details",
    ]
    df = df[column_order]
    df.to_csv(config.PATIENTS_CSV, index=False)

    excluded = df[df["excluded"] == "yes"].copy()
    excluded.to_csv(config.CHECKPOINT_EXCLUDED_CSV, index=False)
    return df


def _parse_file(path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    norm_text = normalize(raw_text)
    lines = raw_text.splitlines()

    patient_id, patient_record = _parse_ids(lines, norm_text)
    date_admission = _parse_admission_date(norm_text)
    date_discharge = _parse_discharge_date(norm_text)
    sex = _parse_sex(norm_text)
    age = _parse_age(norm_text, date_admission, lines)
    doac_drug = _detect_doac_drug(norm_text)
    excluded, exclusion_reason, pregnancy_details = _apply_exclusions(age, norm_text)

    return {
        "source_file": path.name,
        "patient_id": patient_id,
        "patient_record": patient_record,
        "age": age,
        "sex": sex,
        "date_admission": date_admission.isoformat() if date_admission else None,
        "date_discharge": date_discharge.isoformat() if date_discharge else None,
        "doac_drug": doac_drug,
        "excluded": "yes" if excluded else "no",
        "exclusion_reason": exclusion_reason,
        "pregnancy_details": pregnancy_details,
    }


def _parse_ids(lines: list[str], norm_text: str) -> tuple[str | None, str | None]:
    patient_id: str | None = None
    patient_record: str | None = None

    if len(lines) >= 4:
        line1 = lines[0].strip().lower()
        line2 = lines[1].strip().lower()
        if line1.startswith("mã bn") and line2.startswith("mã đt"):
            m_id = re.match(r"^(\d{10})", lines[2].strip())
            if m_id:
                patient_id = m_id.group(1)
            m_rec = re.match(r"^(\d{12})", lines[3].strip())
            if m_rec:
                patient_record = m_rec.group(1)

    if not patient_id:
        m = re.search(config.RE_PATIENT_ID, norm_text)
        if m:
            patient_id = m.group(1)

    if not patient_record:
        m = re.search(config.RE_PATIENT_RECORD, norm_text)
        if m:
            patient_record = m.group(1)

    return patient_id, patient_record


def _parse_admission_date(norm_text: str) -> date | None:
    m = re.search(
        r"12\.\s*vào viện:.*?(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
        norm_text,
    )
    if m:
        return parse_vietnamese_date_fragment(m.group(1))

    m = re.search(
        r"vào viện:.*?(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
        norm_text,
    )
    if m:
        return parse_vietnamese_date_fragment(m.group(1))
    return None


def _parse_discharge_date(norm_text: str) -> date | None:
    m = re.search(
        r"18\.\s*ra viện:\s*(?:\d{1,2}\s+giờ\s+\d{1,2}(?:\s+ph(?:út)?)?\s+)?"
        r"(?:ngày\s+)?(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
        norm_text,
    )
    if m:
        return parse_vietnamese_date_fragment(m.group(1))
    return None


def _parse_sex(norm_text: str) -> str | None:
    for pattern in (
        r"giới tính:\s*(nam|nữ)\b",
        r"giới:\s*(nam|nữ)\b",
        r"bệnh nhân\s+(nam|nữ)\s+\d{1,3}\s*tuổi",
    ):
        m = re.search(pattern, norm_text)
        if m:
            return m.group(1)

    if re.search(r"2\.\s*nữ\s*x\b|2\.\s*nữx\b|\bnữx\b", norm_text):
        return "nữ"
    if re.search(r"1\.\s*nam\s*x\b|1\.\s*namx\b", norm_text) and not re.search(
        r"nữx", norm_text
    ):
        return "nam"
    return None


def _parse_age(
    norm_text: str,
    date_admission: date | None,
    lines: list[str] | None = None,
) -> int | None:
    admin_section = norm_text[:8000]

    for pattern in (
        r"bệnh nhân\s+(?:nam|nữ)\s+(\d{1,3})\s*tuổi",
        r"tuổi:\s*(\d{1,3})\s*tuổi",
        r"tuổi:\s*(\d{1,3})\b",
        r"năm sinh:\s*\d{4}\s*\((\d{1,3})\s*tuổi\)",
    ):
        m = re.search(pattern, admin_section)
        if m:
            age = int(m.group(1))
            if _is_plausible_age(age):
                return age

    if lines and date_admission:
        age = _parse_age_from_header_birth_digits(lines, date_admission)
        if age is not None:
            return age

    for scope in (admin_section, norm_text):
        m = re.search(r"năm sinh:\s*(\d{4})", scope)
        if m and date_admission:
            age = date_admission.year - int(m.group(1))
            if _is_plausible_age(age):
                return age

        m = re.search(
            r"sinh ngày:\s*(?:\d\s+){0,12}(\d{1,2})\s*(?:\d\s+){0,12}(\d{1,2})\s*(?:\d\s+){0,12}(\d{4})",
            scope,
        )
        if m and date_admission:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            birth = _safe_date(year, month, day)
            if birth:
                age = _age_on_date(birth, date_admission)
                if _is_plausible_age(age):
                    return age

        m = re.search(r"sinh ngày:\s*(\d{1,2}/\d{1,2}/\d{4})", scope)
        if m and date_admission:
            birth = parse_vietnamese_date_fragment(m.group(1))
            if birth:
                age = _age_on_date(birth, date_admission)
                if _is_plausible_age(age):
                    return age

    for m in re.finditer(r"\b(\d{2,3})\s*tuổi\b", admin_section):
        age = int(m.group(1))
        if _is_plausible_age(age):
            return age

    return None


def _parse_age_from_header_birth_digits(
    lines: list[str],
    date_admission: date,
) -> int | None:
    for line in lines[:40]:
        digits = re.sub(r"\s+", "", line.strip())
        if not re.fullmatch(r"\d{8,10}", digits):
            continue
        day = int(digits[0:2])
        month = int(digits[2:4])
        year = int(digits[4:8])
        birth = _safe_date(year, month, day)
        if not birth:
            continue
        age = _age_on_date(birth, date_admission)
        if _is_plausible_age(age):
            return age
    return None


def _detect_doac_drug(norm_text: str) -> str | None:
    counts: dict[str, int] = {}
    first_pos: dict[str, int] = {}
    for alias, generic in config.DOAC_ALIASES.items():
        for m in re.finditer(re.escape(alias), norm_text):
            counts[generic] = counts.get(generic, 0) + 1
            first_pos.setdefault(generic, m.start())

    if not counts:
        return None

    return max(counts, key=lambda drug: (counts[drug], -first_pos[drug]))


def _apply_exclusions(
    age: int | None,
    norm_text: str,
) -> tuple[bool, str | None, str | None]:
    reasons: list[str] = []
    pregnancy_details: str | None = None # need to create a None value for this
    if age is not None and age < config.MIN_COHORT_AGE:
        reasons.append(f"age_lt_{config.MIN_COHORT_AGE}")
    pregnancy, pregnancy_details = _is_clinical_pregnancy(norm_text)
    if pregnancy:
        reasons.append("pregnancy")
    if not reasons:
        return False, None, None
    return True, "; ".join(reasons), pregnancy_details


def _is_clinical_pregnancy(norm_text: str) -> tuple[bool, str | None]:
    scrubbed = norm_text
    sex = _parse_sex(norm_text)
    if sex != "nữ":
        return False, None
    for pattern in config.PREGNANCY_BOILERPLATE_IGNORE:
        scrubbed = re.sub(pattern, " ", scrubbed)
    pregnancy_details = []
    for pattern in config.PREGNANCY_EXCLUDE_PATTERNS:
        if m:=re.search(pattern, scrubbed):
            start, end = m.span()
            pregnancy_details.append(scrubbed[start-100: end+100])
    if pregnancy_details:
        return True, "; ".join(pregnancy_details)
    return False, None


def _is_plausible_age(age: int) -> bool:
    return 0 < age <= 125


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _age_on_date(birth: date, on_date: date) -> int:
    age = on_date.year - birth.year
    if (on_date.month, on_date.day) < (birth.month, birth.day):
        age -= 1
    return age


if __name__ == "__main__":
    result = extract_patient_metadata()
    print(f"Wrote {len(result)} records to {config.PATIENTS_CSV}")
    excluded = (result["excluded"] == "yes").sum()
    if excluded:
        print(f"Excluded {excluded} records to {config.CHECKPOINT_EXCLUDED_CSV}")
