"""Extract concurrent medications and interaction flags from admission text files."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.extract_metadata import _parse_ids
from src.parse_text import normalize, parse_vietnamese_date_fragment

RE_ORDER_FORM_HEADER = re.compile(r"phiếu thực\s+hiện y lệnh", re.IGNORECASE)
RE_HIEN_Y_LENH_HEADER = re.compile(r"hiện y lệnh", re.IGNORECASE)
RE_BLOCK_ORDER_DATE = re.compile(
    r"ngày sử dụng:?\s*(?:\d{1,2}:\d{2}\s+)?(\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)
RE_DISCHARGE_HEADER = re.compile(r"(?:^|\n)đơn thuốc\s*(?:\n|$)", re.IGNORECASE)
RE_ORDER_DATE = re.compile(
    r"ngày sử dụng:?\s*(?:\d{1,2}:\d{2}\s+)?(\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)
RE_DOSE_STRENGTH = re.compile(r"(\d+(?:,\d+)?)\s*mg\b", re.IGNORECASE)
RE_GENERIC_IN_PARENS = re.compile(r"\(([^)]+)\)")
RE_TABLE_ROW = re.compile(
    r"(?:^|\n)\s*(\d+)\s+(?:\(\d+\)\s+)?(?=[a-zà-ỹ])",
    re.IGNORECASE,
)
RE_DISCHARGE_ITEM = re.compile(
    r"(?:^|\n)\s*(\d+)\.\s+(.+?)(?=(?:\n\s*\d+\.\s+)|(?:\n\s*lời dặn)|(?:\n\s*ngày\s+\d)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_QTY_UNITS = r"viên|ống|chai|gói|bơm|đôi|lọ"
RE_PAREN_RETRIEVAL = re.compile(
    rf"(\d+(?:,\d+)?)\s*\(\s*thu\s+hồi\s+(\d+(?:,\d+)?)\s*\)\s*({_QTY_UNITS})\b",
    re.IGNORECASE,
)
RE_PARTIAL_RETRIEVAL = re.compile(
    rf"(?:x\s+)?(\d+(?:,\d+)?)\s*({_QTY_UNITS})\s*"
    rf"\(?\s*thu\s+hồi\s+(\d+(?:,\d+)?)\s*(?:{_QTY_UNITS})?\s*\)?",
    re.IGNORECASE,
)
RE_STANDARD_QTY = re.compile(
    rf"x\s+(\d+(?:,\d+)?)\s*({_QTY_UNITS})\b",
    re.IGNORECASE,
)
RE_FULL_RETRIEVAL = re.compile(
    rf"đã\s+thu\s+hồi\s*\(?\s*thu\s+hồi\s+(\d+(?:,\d+)?)\s*\)?\s*({_QTY_UNITS})?",
    re.IGNORECASE,
)
RE_TABLE_QTY = re.compile(
    rf"(?<!\d)(\d+(?:,\d+)?)\s*({_QTY_UNITS})\b",
    re.IGNORECASE,
)
RE_STANDALONE_QTY_LINE = re.compile(
    rf"^(\d+(?:,\d+)?)\s*({_QTY_UNITS})\s*$",
    re.IGNORECASE,
)
RE_BILLING_NUMBERS = re.compile(r"\d{1,3}(?:\.\d{3}){2,}")

_ORDER_FORM_STOP_MARKERS = (
    "xét nghiệm:",
    "siêu âm:",
    "chẩn đoán hình ảnh:",
    "bác sĩ điều trị gia đình",
    "chăm sóc cấp",
    "theo dõi toàn trạng",
    "chế độ ăn",
    "thủ thuật:",
)

_DISCHARGE_CONSUMABLE_PATTERNS = (
    r"\bgăng tay\b",
    r"\bkhẩu trang\b",
    r"\bbông gạc\b",
    r"\bbăng cuộn\b",
    r"\bbăng keo\b",
    r"\bbăng dính\b",
    r"\bbăng thun\b",
    r"\bbăng vết thương\b",
    r"\bbơm tiêm\b",
    r"\bkim tiêm\b",
    r"\bkim luồn\b",
    r"\bdây truyền\b",
    r"\bchạc\b",
    r"\bchỉ không tan\b",
    r"\bchỉ tiêu\b",
    r"\btất áp lực\b",
    r"\bvớ y khoa\b",
    r"\bđeo vớ\b",
    r"\bgạc hn\b",
    r"\bvật tư\b",
)

_DISCHARGE_IV_FLUID_PATTERNS = (
    r"\bnatri clorid\b",
    r"\bnatri chlorid\b",
    r"\bsodium chloride\b",
    r"\bglucose\b",
    r"\bdextrose\b",
    r"\blactated ringer\b",
    r"\bringer['']s\b",
    r"\bdịch truyền\b",
    r"\btruyền tĩnh mạch\b",
    r"\btruyền tm\b",
    r"\bgiữ ven\b",
    r"\bgiữ vein\b",
)

_DISCHARGE_SUPPLY_UNITS = frozenset({"đôi", "cái", "bộ", "cuộn", "sợi", "túi"})
_DISCHARGE_MEDICATION_UNITS = frozenset({"viên", "gói"})
_SAMPLES_PER_DOAC = 50


@dataclass(frozen=True)
class InhibitorEntry:
    generic: str
    aliases: tuple[str, ...]


def extract_medications(text_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all text files and write medications_long.csv + discharge_medications.csv."""
    # setup folders
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_EXPORT.mkdir(parents=True, exist_ok=True)
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    cyp3a4_entries = _load_inhibitor_entries(config.CYP3A4_INHIBITORS_CSV)
    pgp_entries = _load_inhibitor_entries(config.PGP_INHIBITORS_CSV)

    inpatient_rows: list[dict] = []
    discharge_rows: list[dict] = []
    filtered_discharge_rows: list[dict] = []

    for path in sorted(text_dir.glob("*.txt")):
        inpatient_rows.extend(
            _parse_inpatient_medications(path, cyp3a4_entries, pgp_entries)
        )
        accepted, filtered = _parse_discharge_medications(
            path, cyp3a4_entries, pgp_entries
        )
        discharge_rows.extend(accepted)
        filtered_discharge_rows.extend(filtered)

    inpatient_df = _finalize_inpatient_df(inpatient_rows)
    discharge_df = _finalize_discharge_df(discharge_rows)

    inpatient_df.to_csv(config.MEDICATIONS_LONG_CSV, index=False)
    discharge_df.to_csv(config.DISCHARGE_MEDICATIONS_CSV, index=False)

    _write_medication_checkpoints(
        inpatient_df,
        discharge_df,
        filtered_discharge_rows,
        cyp3a4_entries,
        pgp_entries,
        text_dir,
    )
    return inpatient_df, discharge_df


def _load_inhibitor_entries(csv_path: Path) -> list[InhibitorEntry]:
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    entries: list[InhibitorEntry] = []
    for _, row in df.iterrows():
        generic = normalize(str(row.get("generic", "")))
        if not generic:
            continue
        alias_field = normalize(str(row.get("alias", "")))
        aliases = tuple(
            token
            for token in re.split(r"[\s,;/]+", alias_field)
            if token and token != generic
        )
        entries.append(InhibitorEntry(generic=generic, aliases=aliases))
    return entries


def _parse_inpatient_medications(
    path: Path,
    cyp3a4_entries: list[InhibitorEntry],
    pgp_entries: list[InhibitorEntry],
) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    norm_text = normalize(raw_text)
    lines = raw_text.splitlines()
    _, patient_record = _parse_ids(lines, norm_text)
    if not patient_record:
        return []

    daily_entries: dict[tuple[str, str, str], dict] = {}
    for block, source_section in _iter_inpatient_order_blocks(raw_text):
        order_date = _parse_block_order_date(block)
        if not order_date:
            continue
        date_key = order_date.isoformat()
        for drug in _parse_order_form_drugs(block):
            drug_key = _drug_match_key(drug["drug_raw"], drug["drug_generic"])
            bucket_key = (patient_record, date_key, drug_key)
            if bucket_key not in daily_entries:
                daily_entries[bucket_key] = {
                    "patient_record": patient_record,
                    "drug_raw": drug["drug_raw"],
                    "drug_generic": drug["drug_generic"],
                    "dose": drug["dose"],
                    "route": drug["route"],
                    "date_ordered": date_key,
                    "date_active": date_key,
                    "qty_ordered": drug["qty_ordered"],
                    "qty_active": drug["qty_active"],
                    "unit": drug["unit"],
                    "is_stopped": drug["is_stopped"],
                    "source_section": source_section,
                    "source_file": path.name,
                }
                continue

            existing = daily_entries[bucket_key]
            existing["qty_ordered"] = _sum_optional(
                existing["qty_ordered"], drug["qty_ordered"]
            )
            existing["qty_active"] = _sum_optional(
                existing["qty_active"], drug["qty_active"]
            )
            existing["is_stopped"] = (
                "yes" if _to_float(existing["qty_active"]) == 0 else "no"
            )
            if len(drug["drug_raw"]) > len(str(existing["drug_raw"])):
                existing["drug_raw"] = drug["drug_raw"]

    rows: list[dict] = []
    for entry in daily_entries.values():
        entry["is_cyp3a4_inhibitor"] = _match_inhibitor(
            entry["drug_raw"], entry["drug_generic"], cyp3a4_entries
        )
        entry["is_pgp_inhibitor"] = _match_inhibitor(
            entry["drug_raw"], entry["drug_generic"], pgp_entries
        )
        rows.append(entry)
    return rows


def _parse_discharge_medications(
    path: Path,
    cyp3a4_entries: list[InhibitorEntry],
    pgp_entries: list[InhibitorEntry],
) -> tuple[list[dict], list[dict]]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    norm_text = normalize(raw_text)
    lines = raw_text.splitlines()
    _, patient_record = _parse_ids(lines, norm_text)
    if not patient_record:
        return [], []

    accepted_rows: list[dict] = []
    filtered_rows: list[dict] = []
    for block in _split_discharge_blocks(raw_text):
        prescription_date = _parse_discharge_date(block)
        date_str = prescription_date.isoformat() if prescription_date else None
        for item in _parse_discharge_items(block):
            include, reason = _classify_discharge_item(item)
            base_row = {
                "patient_record": patient_record,
                "drug_raw": item["drug_raw"],
                "drug_generic": _resolve_generic_name(item["drug_raw"]),
                "dose": _extract_dose_strength(item["drug_raw"]),
                "route": _infer_route(
                    item["drug_raw"],
                    item.get("unit"),
                    drug_generic=_resolve_generic_name(item["drug_raw"]),
                ),
                "date_ordered": date_str,
                "date_active": date_str,
                "qty_ordered": item["qty_ordered"],
                "qty_active": item["qty_active"],
                "unit": item.get("unit"),
                "is_stopped": "no",
                "source_section": "đơn thuốc",
                "source_file": path.name,
            }
            if not include:
                filtered_rows.append(
                    {
                        **base_row,
                        "filter_reason": reason,
                    }
                )
                continue

            drug_raw = item["drug_raw"]
            drug_generic = base_row["drug_generic"]
            accepted_rows.append(
                {
                    **base_row,
                    "is_cyp3a4_inhibitor": _match_inhibitor(
                        drug_raw, drug_generic, cyp3a4_entries
                    ),
                    "is_pgp_inhibitor": _match_inhibitor(
                        drug_raw, drug_generic, pgp_entries
                    ),
                }
            )
    return accepted_rows, filtered_rows


def _split_header_blocks(raw_text: str, header: re.Pattern[str]) -> list[str]:
    matches = list(header.finditer(raw_text))
    if not matches:
        return []

    blocks: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        blocks.append(raw_text[start:end])
    return blocks


def _split_order_form_blocks(raw_text: str) -> list[str]:
    return _split_header_blocks(raw_text, RE_ORDER_FORM_HEADER)


def _split_hien_y_lenh_blocks(raw_text: str) -> list[str]:
    return _split_header_blocks(raw_text, RE_HIEN_Y_LENH_HEADER)


def _iter_inpatient_order_blocks(raw_text: str) -> list[tuple[str, str]]:
    """Yield (block, source_section) from structured inpatient order tables."""
    blocks: list[tuple[str, str]] = []
    for block in _split_order_form_blocks(raw_text):
        blocks.append((block, "phiếu thực hiện y lệnh"))
    for block in _split_hien_y_lenh_blocks(raw_text):
        blocks.append((block, "hiện y lệnh"))
    return blocks


def _split_discharge_blocks(raw_text: str) -> list[str]:
    matches = list(RE_DISCHARGE_HEADER.finditer(raw_text))
    if not matches:
        return []

    blocks: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        block = raw_text[start:end]
        if "ii.thông tin đơn thuốc" in block.lower():
            blocks.append(block)
    return blocks


def _parse_order_form_date(block: str) -> date | None:
    match = RE_ORDER_DATE.search(block)
    if not match:
        return None
    return parse_vietnamese_date_fragment(match.group(1))


def _parse_block_order_date(block: str) -> date | None:
    order_date = _parse_order_form_date(block)
    if order_date:
        return order_date
    match = RE_BLOCK_ORDER_DATE.search(block)
    if not match:
        return None
    return parse_vietnamese_date_fragment(match.group(1))


def _parse_discharge_date(block: str) -> date | None:
    match = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        block,
        re.IGNORECASE,
    )
    if match:
        day, month, year = (int(match.group(i)) for i in range(1, 4))
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def _parse_order_form_drugs(block: str) -> list[dict]:
    y_lenh = _extract_y_lenh_section(block)
    if not y_lenh:
        return []

    rows: list[dict] = []
    segments = _split_table_rows(y_lenh)
    for segment in segments:
        drug = _parse_drug_segment(segment)
        if drug:
            rows.append(drug)
    return rows


def _extract_y_lenh_section(block: str) -> str:
    lower = block.lower()
    start = lower.find("y lệnh")
    if start < 0:
        return ""

    section = block[start:]
    stop_at = len(section)
    for marker in _ORDER_FORM_STOP_MARKERS:
        marker_pos = section.lower().find(marker)
        if marker_pos > 0:
            stop_at = min(stop_at, marker_pos)
    section = section[:stop_at]

    header_end = section.lower().find("nguồn cấp")
    if header_end >= 0:
        section = section[header_end + len("nguồn cấp") :]
    return section.strip()


def _split_table_rows(section: str) -> list[str]:
    matches = list(RE_TABLE_ROW.finditer(section))
    if not matches:
        compact = " ".join(section.split())
        if compact:
            return [compact]
        return []

    segments: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section)
        segments.append(section[start:end].strip())
    return segments


def _parse_drug_segment(segment: str) -> dict | None:
    compact = " ".join(segment.split())
    if not compact or len(compact) < 4:
        return None
    if compact.lower().startswith("stt "):
        return None

    qty_active, qty_ordered, unit, is_stopped = _compute_active_quantity(compact)
    if qty_active is None and qty_ordered is None:
        return None

    drug_raw = _strip_quantity_from_drug_text(compact)
    if not drug_raw or len(drug_raw) < 3:
        return None

    drug_generic = _resolve_generic_name(drug_raw)
    return {
        "drug_raw": drug_raw[:500],
        "drug_generic": drug_generic,
        "dose": _extract_dose_strength(drug_raw),
        "route": _infer_route(compact, unit, drug_generic=drug_generic, drug_raw=drug_raw),
        "qty_ordered": qty_ordered,
        "qty_active": qty_active if qty_active is not None else 0.0,
        "unit": unit,
        "is_stopped": "yes" if is_stopped else "no",
    }


def _parse_discharge_items(block: str) -> list[dict]:
    info_start = block.lower().find("ii.thông tin đơn thuốc")
    if info_start < 0:
        return []
    section = block[info_start:]
    end = section.lower().find("lời dặn")
    if end > 0:
        section = section[:end]

    items: list[dict] = []
    for match in RE_DISCHARGE_ITEM.finditer(section):
        body = match.group(2).strip()
        if not body or body.startswith("("):
            continue
        if RE_BILLING_NUMBERS.search(body):
            continue

        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            continue

        drug_lines: list[str] = []
        qty_line: str | None = None
        for line in lines:
            standalone_qty = RE_STANDALONE_QTY_LINE.match(line)
            if standalone_qty:
                qty_line = line
                break
            if not re.search(
                rf"^\d+(?:,\d+)?\s*(?:{'|'.join(_DISCHARGE_MEDICATION_UNITS)})\b",
                line,
                re.I,
            ):
                drug_lines.append(line)

        if not drug_lines:
            continue

        drug_raw = _clean_drug_name(" ".join(drug_lines))
        if not drug_raw:
            continue

        qty_active, qty_ordered, unit, _ = _compute_active_quantity(
            qty_line or body.replace("\n", " ")
        )
        if qty_active is None and qty_ordered is None:
            tail_match = re.search(
                rf"(\d+(?:,\d+)?)\s*({'|'.join(_DISCHARGE_MEDICATION_UNITS)})\s*$",
                normalize(drug_raw),
            )
            if tail_match:
                qty_active = qty_ordered = _parse_viet_number(tail_match.group(1))
                unit = tail_match.group(2)
                drug_raw = re.sub(
                    rf"\s*\d+(?:,\d+)?\s*{re.escape(unit)}\s*$",
                    "",
                    drug_raw,
                    flags=re.IGNORECASE,
                ).strip(" .,")
                drug_raw = _clean_drug_name(drug_raw)
            else:
                continue

        items.append(
            {
                "drug_raw": drug_raw[:500],
                "body": body,
                "qty_line": qty_line,
                "qty_ordered": qty_ordered,
                "qty_active": qty_active if qty_active is not None else qty_ordered,
                "unit": unit,
            }
        )
    return items


def _compute_active_quantity(text: str) -> tuple[float | None, float | None, str | None, bool]:
    norm = normalize(text.replace("\n", " "))

    paren_partial = RE_PAREN_RETRIEVAL.search(norm)
    if paren_partial:
        total = _parse_viet_number(paren_partial.group(1))
        retrieved = _parse_viet_number(paren_partial.group(2))
        unit = paren_partial.group(3)
        active = max(0.0, total - retrieved)
        return active, total, unit, active == 0.0

    partial = RE_PARTIAL_RETRIEVAL.search(norm)
    if partial:
        total = _parse_viet_number(partial.group(1))
        unit = partial.group(2)
        retrieved = _parse_viet_number(partial.group(3))
        active = max(0.0, total - retrieved)
        return active, total, unit, active == 0.0

    full_stop = RE_FULL_RETRIEVAL.search(norm)
    if full_stop:
        ordered = _parse_viet_number(full_stop.group(1))
        unit = full_stop.group(2) or "viên"
        return 0.0, ordered, unit, True

    if re.search(r"đã\s+thu\s+hồi", norm):
        return 0.0, None, None, True

    standard = RE_STANDARD_QTY.search(norm)
    if standard:
        qty = _parse_viet_number(standard.group(1))
        return qty, qty, standard.group(2), False

    table_qty = RE_TABLE_QTY.search(norm)
    if table_qty:
        qty = _parse_viet_number(table_qty.group(1))
        return qty, qty, table_qty.group(2), False

    return None, None, None, False


def _strip_quantity_from_drug_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    cleaned = RE_PAREN_RETRIEVAL.sub("", cleaned)
    cleaned = RE_PARTIAL_RETRIEVAL.sub("", cleaned)
    cleaned = re.sub(r"đã\s+thu\s+hồi.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = RE_STANDARD_QTY.sub("", cleaned)
    cleaned = RE_TABLE_QTY.sub("", cleaned)
    return _clean_drug_name(cleaned)


def _clean_drug_name(text: str) -> str:
    cleaned = re.sub(r"\blĩnh ở\b.*", "", text, flags=re.IGNORECASE)
    cleaned = re.split(
        r"\b(?:ngày uống|ngày tiêm|tiêm dưới|tiêm truyền|pha |truyền btđ)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,")
    return cleaned


def _classify_discharge_item(item: dict) -> tuple[bool, str | None]:
    drug_raw = item["drug_raw"]
    body = item.get("body", "")
    qty_line = item.get("qty_line")
    unit = (item.get("unit") or "").lower()
    norm = normalize(f"{drug_raw} {body}")

    if "chưa có hướng dẫn" in norm:
        return False, "no_usage_instruction"

    for pattern in _DISCHARGE_CONSUMABLE_PATTERNS:
        if re.search(pattern, norm):
            return False, "consumable_supply"

    for pattern in _DISCHARGE_IV_FLUID_PATTERNS:
        if re.search(pattern, norm):
            return False, "iv_fluid"

    if unit in _DISCHARGE_SUPPLY_UNITS:
        return False, "supply_unit"

    if unit == "chai":
        return False, "iv_fluid"

    if unit and unit not in _DISCHARGE_MEDICATION_UNITS:
        return False, "non_medication_unit"

    if not _looks_like_medication(drug_raw):
        return False, "not_medication"

    has_oral_instruction = "ngày uống" in norm or " uống" in norm
    has_standalone_qty = qty_line is not None
    has_embedded_qty = bool(
        re.search(rf"\d+(?:,\d+)?\s*({'|'.join(_DISCHARGE_MEDICATION_UNITS)})\b", norm)
    )
    if not has_oral_instruction and not has_standalone_qty and not has_embedded_qty:
        return False, "missing_prescription_format"

    if drug_raw.strip().startswith(".") and re.search(
        r"\b(?:dịch vụ|bhyt|yêu cầu)\d+\b", norm
    ):
        if not (has_oral_instruction or _is_doac(drug_raw, _resolve_generic_name(drug_raw))):
            return False, "billing_format"

    return True, None


def _is_doac(drug_raw: str | None, drug_generic: str | None) -> bool:
    if drug_generic in config.DOAC_DRUGS:
        return True
    norm = normalize(drug_raw or "")
    return any(alias in norm for alias in config.DOAC_ALIASES)


def _parse_viet_number(value: str) -> float:
    return float(value.replace(",", "."))


def _resolve_generic_name(drug_raw: str) -> str | None:
    norm = normalize(drug_raw)
    for alias, generic in config.DOAC_ALIASES.items():
        if alias in norm:
            return generic

    paren_matches = RE_GENERIC_IN_PARENS.findall(drug_raw)
    for candidate in reversed(paren_matches):
        candidate_norm = normalize(candidate)
        if any(
            skip in candidate_norm
            for skip in ("dưới dạng", "tài trợ", "tầng đùi", "độ 2", "tương đương")
        ):
            continue
        generic = candidate_norm.split(",")[0].strip()
        generic = re.sub(r"\s+\d+(?:,\d+)?\s*mg\b.*", "", generic).strip()
        if generic and len(generic) >= 3:
            return generic
    return None


def _extract_dose_strength(drug_raw: str) -> str | None:
    match = RE_DOSE_STRENGTH.search(drug_raw)
    if not match:
        return None
    return f"{match.group(1).replace(',', '.')} mg"


def _looks_like_medication(drug_raw: str) -> bool:
    norm = normalize(drug_raw)
    if not norm or norm.startswith("thuốc, dịch truyền"):
        return False
    if RE_BILLING_NUMBERS.search(norm):
        return False
    if any(re.search(pattern, norm) for pattern in _DISCHARGE_CONSUMABLE_PATTERNS):
        return False
    if any(re.search(pattern, norm) for pattern in _DISCHARGE_IV_FLUID_PATTERNS):
        return False
    return bool(
        re.search(r"\bmg\b|\([^)]+\)", norm)
        or any(alias in norm for alias in config.DOAC_ALIASES)
    )


def _infer_route(
    text: str,
    unit: str | None,
    drug_generic: str | None = None,
    drug_raw: str | None = None,
) -> str | None:
    if _is_doac(drug_raw, drug_generic):
        return "oral"

    norm = normalize(text)
    if "uống" in norm:
        return "oral"
    if unit and unit.lower() in {"viên", "gói"}:
        return "oral"
    if "tiêm" in norm or "truyền" in norm:
        return "injection"
    if unit and unit.lower() in {"ống", "bơm"}:
        return "injection"
    return None


def _match_inhibitor(
    drug_raw: str,
    drug_generic: str | None,
    entries: list[InhibitorEntry],
) -> str:
    norm = normalize(f"{_clean_drug_name(drug_raw)} {drug_generic or ''}")
    for entry in entries:
        if entry.generic in norm:
            return "yes"
        if any(alias in norm for alias in entry.aliases):
            return "yes"
    return "no"


def _drug_match_key(drug_raw: str, drug_generic: str | None) -> str:
    if drug_generic:
        return normalize(drug_generic)
    return normalize(drug_raw)[:120]


def _sum_optional(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return _to_float(left) + _to_float(right)


def _to_float(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _finalize_inpatient_df(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "patient_record",
        "drug_raw",
        "drug_generic",
        "dose",
        "route",
        "date_ordered",
        "date_active",
        "qty_ordered",
        "qty_active",
        "unit",
        "is_stopped",
        "source_section",
        "is_cyp3a4_inhibitor",
        "is_pgp_inhibitor",
        "source_file",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)[columns]
    doac_mask = df.apply(
        lambda row: _is_doac(row["drug_raw"], row["drug_generic"]),
        axis=1,
    )
    df.loc[doac_mask, "route"] = "oral"
    return df.sort_values(
        ["patient_record", "date_active", "drug_generic", "drug_raw"],
        na_position="last",
    ).reset_index(drop=True)


def _finalize_discharge_df(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "patient_record",
        "drug_raw",
        "drug_generic",
        "dose",
        "route",
        "date_ordered",
        "date_active",
        "qty_ordered",
        "qty_active",
        "unit",
        "is_stopped",
        "source_section",
        "is_cyp3a4_inhibitor",
        "is_pgp_inhibitor",
        "source_file",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)[columns]
    doac_mask = df.apply(
        lambda row: _is_doac(row["drug_raw"], row["drug_generic"]),
        axis=1,
    )
    df.loc[doac_mask, "route"] = "oral"
    return df.drop_duplicates(
        subset=["patient_record", "date_active", "drug_raw", "qty_active", "source_file"],
        keep="first",
    ).sort_values(
        ["patient_record", "date_active", "drug_generic", "drug_raw"],
        na_position="last",
    ).reset_index(drop=True)


def _write_medication_checkpoints(
    inpatient_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    filtered_discharge_rows: list[dict],
    cyp3a4_entries: list[InhibitorEntry],
    pgp_entries: list[InhibitorEntry],
    text_dir: Path,
) -> None:
    _write_doac_stop_samples(inpatient_df, text_dir)
    _write_medication_frequency_json(
        inpatient_df,
        discharge_df,
        cyp3a4_entries,
        pgp_entries,
    )
    _write_discharge_filtered_json(filtered_discharge_rows)


def _is_stop_logic_event(row: pd.Series) -> bool:
    if row.get("is_stopped") == "yes":
        return True
    qty_active = _to_float(row.get("qty_active"))
    qty_ordered = row.get("qty_ordered")
    if pd.isna(qty_ordered):
        return False
    return qty_active < float(qty_ordered)


def _write_doac_stop_samples(inpatient_df: pd.DataFrame, text_dir: Path) -> None:
    columns = [
        "patient_record",
        "date_ordered",
        "drug_raw",
        "drug_generic",
        "source_context",
        "source_file",
    ]
    if inpatient_df.empty:
        pd.DataFrame(columns=columns).to_csv(
            config.CHECKPOINT_DOAC_STOP_SAMPLES_CSV,
            index=False,
        )
        return

    doac_df = inpatient_df[
        inpatient_df["drug_generic"].isin(sorted(config.DOAC_DRUGS))
    ].copy()
    stop_df = doac_df[doac_df.apply(_is_stop_logic_event, axis=1)]

    samples: list[pd.DataFrame] = []
    for drug in sorted(config.DOAC_DRUGS):
        subset = stop_df[stop_df["drug_generic"] == drug]
        if subset.empty:
            continue
        sample_n = min(_SAMPLES_PER_DOAC, len(subset))
        samples.append(subset.sample(n=sample_n, random_state=42))

    if samples:
        sample_df = pd.concat(samples, ignore_index=True)
    else:
        sample_df = pd.DataFrame(columns=doac_df.columns)

    if not sample_df.empty:
        contexts: list[str] = []
        for _, row in sample_df.iterrows():
            contexts.append(
                _find_medication_context(
                    text_dir / str(row["source_file"]),
                    str(row["date_ordered"]),
                    str(row["drug_raw"]),
                )
            )
        sample_df = sample_df.assign(source_context=contexts)

    output = sample_df.reindex(columns=columns)
    output.to_csv(config.CHECKPOINT_DOAC_STOP_SAMPLES_CSV, index=False)


def _find_medication_context(
    path: Path,
    date_ordered: str,
    drug_raw: str,
) -> str:
    if not path.exists():
        return ""

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    drug_tokens = [
        token
        for token in re.split(r"\s+", normalize(drug_raw))
        if len(token) >= 4 and token not in {"dưới", "dạng", "tương", "đương"}
    ]
    if not drug_tokens:
        drug_tokens = [normalize(drug_raw)[:20]]

    for block in _split_order_form_blocks(raw_text):
        order_date = _parse_order_form_date(block)
        if not order_date or order_date.isoformat() != date_ordered:
            continue

        block_norm = normalize(block.replace("\n", " "))
        if not any(token in block_norm for token in drug_tokens[:2]):
            continue

        for line in block.splitlines():
            line_norm = normalize(line)
            if any(token in line_norm for token in drug_tokens[:2]):
                return " ".join(line.split())[:400]

        for segment in _split_table_rows(_extract_y_lenh_section(block)):
            segment_norm = normalize(" ".join(segment.split()))
            if any(token in segment_norm for token in drug_tokens[:2]):
                return " ".join(segment.split())[:400]

    return ""


def _resolve_inhibitor_name(
    row: pd.Series,
    entries: list[InhibitorEntry],
) -> str | None:
    norm = normalize(f"{row.get('drug_raw', '')} {row.get('drug_generic', '')}")
    for entry in entries:
        if entry.generic in norm:
            return entry.generic
        if any(alias in norm for alias in entry.aliases):
            return entry.generic
    return None


def _write_medication_frequency_json(
    inpatient_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cyp3a4_entries: list[InhibitorEntry],
    pgp_entries: list[InhibitorEntry],
) -> None:
    combined = pd.concat([inpatient_df, discharge_df], ignore_index=True)

    doac_counts = (
        combined[combined["drug_generic"].isin(sorted(config.DOAC_DRUGS))]
        .groupby("drug_generic")
        .size()
        .sort_index()
        .to_dict()
    )

    cyp3a4_counts: Counter[str] = Counter()
    pgp_counts: Counter[str] = Counter()
    for _, row in combined.iterrows():
        if row.get("is_cyp3a4_inhibitor") == "yes":
            name = _resolve_inhibitor_name(row, cyp3a4_entries) or normalize(
                str(row.get("drug_generic") or row.get("drug_raw", ""))[:80]
            )
            cyp3a4_counts[name] += 1
        if row.get("is_pgp_inhibitor") == "yes":
            name = _resolve_inhibitor_name(row, pgp_entries) or normalize(
                str(row.get("drug_generic") or row.get("drug_raw", ""))[:80]
            )
            pgp_counts[name] += 1

    payload = {
        "doac_by_generic": doac_counts,
        "cyp3a4_inhibitors": dict(sorted(cyp3a4_counts.items(), key=lambda x: (-x[1], x[0]))),
        "pgp_inhibitors": dict(sorted(pgp_counts.items(), key=lambda x: (-x[1], x[0]))),
        "totals": {
            "inpatient_rows": int(len(inpatient_df)),
            "discharge_rows": int(len(discharge_df)),
            "combined_rows": int(len(combined)),
            "doac_rows": int(sum(doac_counts.values())),
            "cyp3a4_rows": int((combined["is_cyp3a4_inhibitor"] == "yes").sum()),
            "pgp_rows": int((combined["is_pgp_inhibitor"] == "yes").sum()),
        },
    }
    config.CHECKPOINT_MEDICATION_FREQUENCY_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_discharge_filtered_json(filtered_discharge_rows: list[dict]) -> None:
    reason_counts: Counter[str] = Counter()
    drug_counts: Counter[str] = Counter()
    examples_by_reason: dict[str, list[dict]] = {}

    for row in filtered_discharge_rows:
        reason = str(row.get("filter_reason") or "unknown")
        reason_counts[reason] += 1
        drug_label = normalize(str(row.get("drug_raw", "")))[:120] or "unknown"
        drug_counts[drug_label] += 1
        if len(examples_by_reason.setdefault(reason, [])) < 5:
            examples_by_reason[reason].append(
                {
                    "patient_record": row.get("patient_record"),
                    "date_active": row.get("date_active"),
                    "drug_raw": row.get("drug_raw"),
                    "unit": row.get("unit"),
                    "source_file": row.get("source_file"),
                }
            )

    payload = {
        "total_filtered": len(filtered_discharge_rows),
        "by_reason": {
            reason: {
                "count": count,
                "examples": examples_by_reason.get(reason, []),
            }
            for reason, count in reason_counts.most_common()
        },
        "top_filtered_drugs": dict(drug_counts.most_common(30)),
    }
    config.CHECKPOINT_DISCHARGE_FILTERED_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    inpatient, discharge = extract_medications()
    print(f"Wrote {len(inpatient)} inpatient rows to {config.MEDICATIONS_LONG_CSV}")
    print(f"Wrote {len(discharge)} discharge rows to {config.DISCHARGE_MEDICATIONS_CSV}")
