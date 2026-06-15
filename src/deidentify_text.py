"""Apply PII replacement patterns to admission text source files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src import config


@dataclass(frozen=True)
class ReplacePattern:
    id: str
    regex: re.Pattern[str]
    replacement: str
    priority: int = 0


PATTERNS: list[ReplacePattern] = sorted(
    [
        ReplacePattern(
            "ho_va_ten",
            re.compile(
                r"họ (?:và)?\s*tên\s*:?\s*(?:người bệnh)?\s*"
                r"(.{3,80}?)(?=(?:\d|tuổi|năm sinh|,|$))",
                re.I | re.S | re.M,
            ),
            replacement="<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "admin_name_line",
            re.compile(
                r"(?:^|\n)((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)\s*\n(?=kinh\s*\n|1\.nam)",
                re.I | re.M,
            ),
            replacement="\n<<NamePlaceHolder>>\n",
            priority=1,
        ),
        ReplacePattern(
            "cccd",
            re.compile(r"căn cước công dân\s*:?\s*\d{8,13}", re.I),
            replacement="<<IDPlaceHolder>>",
            priority=2,
        ),
        ReplacePattern(
            "cccd_bare",
            re.compile(
                r"(?<!0000)\b\d{12}\b(?=\s*(?:9\.\s*đối tượng|\d\.\s*đối tượng))",
                re.I,
            ),
            replacement="<<IDPlaceHolder>>",
            priority=2,
        ),
        ReplacePattern(
            "sdt",
            re.compile(
                r"(?:số điện thoại|sđt|điện thoại|số đt)\s*:?\s*\d{9,13}",
                re.I,
            ),
            replacement="<<PhonePlaceHolder>>",
            priority=3,
        ),
        ReplacePattern(
            "phone_spaced",
            re.compile(
                r"điện thoại số\s*\n(?:\d\s+){5,12}\d",
                re.I | re.M,
            ),
            replacement="<<PhonePlaceHolder>>",
            priority=3,
        ),
        ReplacePattern(
            "staff_after_bac_si",
            re.compile(
                r"(bác sĩ điều trị\s*\n)"
                r"((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)"
                r"(?=\s*\n|\{signlibrary)",
                re.I | re.M,
            ),
            replacement=r"\1<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "staff_chieu",
            re.compile(
                r"(?:^|\n)((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)(?=\s+chiều:)",
                re.I | re.M,
            ),
            replacement="<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "staff_nguoi_in",
            re.compile(
                r"người in phiếu:\s*((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)",
                re.I,
            ),
            replacement="người in phiếu: <<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "informant_name",
            re.compile(
                r"(người cung cấp thông tin\s*\n)"
                r"((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)"
                r"(?=\s*\n)",
                re.I | re.M,
            ),
            replacement=r"\1<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "physician_labeled",
            re.compile(
                r"((?:bác sĩ thủ thuật|bs chỉ định|bác sĩ khám bệnh|bác sĩ làm bệnh án)"
                r"\s*:?\s*)"
                r"((?:[a-zà-ỹ]+\s+){1,4}[a-zà-ỹ]+)",
                re.I,
            ),
            replacement=r"\1<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "digital_signed",
            re.compile(
                r"((?:digital signed by|printed by):\s*)"
                r"((?:[a-zà-ỹ]+\s+){1,4}[a-zà-ỹ]+)",
                re.I,
            ),
            replacement=r"\1<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "standalone_name_before_ma_bn",
            re.compile(
                r"(?:^|\n)((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)(?=\s*\nmã bn:)",
                re.I | re.M,
            ),
            replacement="<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "discharge_signatures",
            re.compile(
                r"(bệnh nhân người giao khoa dược bác sĩ khám bệnh\s*\n)"
                r"((?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+(?:\s{2,}(?:[a-zà-ỹ]+\s+){1,3}[a-zà-ỹ]+)?)",
                re.I | re.M,
            ),
            replacement=r"\1<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "name_before_ma_dt",
            re.compile(
                r"((?:[a-zà-ỹ]+\s+){1,4}[a-zà-ỹ]+)\s*(\(\d{4}\)/mã đt:)",
                re.I,
            ),
            replacement=r"<<NamePlaceHolder>> \2",
            priority=1,
        ),
        ReplacePattern(
            "name_before_department",
            re.compile(
                r"((?:[a-zà-ỹ]+\s+){1,4}[a-zà-ỹ]+)(?=(?:trung tâm|khoa huyết|khoa tim|phòng ))",
                re.I,
            ),
            replacement="<<NamePlaceHolder>>",
            priority=1,
        ),
        ReplacePattern(
            "phone_mobile",
            re.compile(r"\b0[35789]\d{8}\b"),
            replacement="<<PhonePlaceHolder>>",
            priority=4,
        ),
    ],
    key=lambda pattern: pattern.priority,
)


def deidentify_text(text: str, patterns: list[ReplacePattern] | None = None) -> str:
    """Return text with all configured PII patterns replaced by placeholders."""
    patterns = patterns or PATTERNS
    for pattern in patterns:
        text = pattern.regex.sub(pattern.replacement, text)
    return text


def deidentify_file(path: Path, patterns: list[ReplacePattern] | None = None) -> bool:
    """De-identify one text file in place. Returns True if content changed."""
    original = path.read_text(encoding="utf-8", errors="replace")
    updated = deidentify_text(original, patterns=patterns)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def deidentify_directory(
    text_dir: Path | None = None,
    patterns: list[ReplacePattern] | None = None,
) -> tuple[int, int]:
    """De-identify all .txt files in text_dir. Returns (changed_count, total_count)."""
    text_dir = text_dir or config.TEXT_DIR
    paths = sorted(text_dir.glob("*.txt"))
    changed = sum(1 for path in paths if deidentify_file(path, patterns=patterns))
    return changed, len(paths)


if __name__ == "__main__":
    changed_count, total_count = deidentify_directory()
    print(f"De-identified {changed_count}/{total_count} files in {config.TEXT_DIR}")
