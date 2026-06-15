"""Generate stratified QA samples and pipeline-wide summary counts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src import config

QA_SAMPLE_SIZE = 10
QA_RANDOM_SEED = 42


def _normalize_patient_record(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return str(value)
    return digits.zfill(12)


def _uses_standard_header(lines: list[str]) -> bool:
    if len(lines) < 4:
        return False
    line1 = lines[0].strip().lower()
    line2 = lines[1].strip().lower()
    if not (line1.startswith("mã bn") and line2.startswith("mã đt")):
        return False
    return bool(re.match(r"^(\d{10})", lines[2].strip())) and bool(
        re.match(r"^(\d{12})", lines[3].strip())
    )


def find_non_standard_header_files(text_dir: Path | None = None) -> list[str]:
    """Return source filenames whose patient IDs require regex fallback."""
    text_dir = text_dir or config.TEXT_DIR
    fallback_files: list[str] = []
    for path in sorted(text_dir.glob("*.txt")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not _uses_standard_header(lines):
            fallback_files.append(path.name)
    return fallback_files


def _sample_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sample(n=min(n, len(df)), random_state=QA_RANDOM_SEED)


def _build_egfr_mismatch_samples(kidney_df: pd.DataFrame) -> pd.DataFrame:
    mismatch = kidney_df[kidney_df["egfr_lab_recalc_mismatch"] == "yes"].copy()
    if mismatch.empty:
        return pd.DataFrame()

    mismatch["egfr_lab_recalc_delta"] = pd.to_numeric(
        mismatch["egfr_lab_recalc_delta"],
        errors="coerce",
    )
    mismatch = mismatch.sort_values(
        "egfr_lab_recalc_delta",
        key=lambda series: series.abs(),
        ascending=False,
    )
    sampled = _sample_rows(mismatch, QA_SAMPLE_SIZE)
    return pd.DataFrame(
        {
            "qa_category": "egfr_mismatch",
            "patient_record": sampled["patient_record"].map(_normalize_patient_record),
            "source_file": sampled["source_file"],
            "detail": sampled.apply(
                lambda row: (
                    f"lab={row.get('egfr_2009_indexed_lab', '')}, "
                    f"recalc={row.get('egfr_2009_indexed_recalc', '')}, "
                    f"delta={row.get('egfr_lab_recalc_delta', '')}, "
                    f"creatinine={row.get('creatinine_umol_l', '')} µmol/L"
                ),
                axis=1,
            ),
            "context_snippet": "",
        }
    )


def _build_bleeding_samples(bleeding_df: pd.DataFrame) -> pd.DataFrame:
    acute = bleeding_df[bleeding_df["event_class"] == "bleeding_event"].copy()
    if acute.empty:
        return pd.DataFrame()

    acute = acute.sort_values(["patient_record", "date_encounter"])
    acute = acute.drop_duplicates(subset=["patient_record"], keep="first")
    sampled = _sample_rows(acute, QA_SAMPLE_SIZE)
    return pd.DataFrame(
        {
            "qa_category": "bleeding_positive",
            "patient_record": sampled["patient_record"].map(_normalize_patient_record),
            "source_file": sampled["source_file"],
            "detail": sampled["trigger_type"].fillna(""),
            "context_snippet": sampled["source_context"].fillna("").str.slice(0, 300),
        }
    )


def _build_exclusion_samples(
    patients_df: pd.DataFrame,
    missing_wh_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    excluded = patients_df[patients_df["excluded"] == "yes"].copy()
    for _, row in _sample_rows(excluded, QA_SAMPLE_SIZE).iterrows():
        rows.append(
            {
                "qa_category": "excluded",
                "patient_record": _normalize_patient_record(row["patient_record"]),
                "source_file": row["source_file"],
                "detail": str(row.get("exclusion_reason", "")),
                "context_snippet": str(row.get("pregnancy_details", ""))[:300],
            }
        )

    if len(rows) >= QA_SAMPLE_SIZE:
        return pd.DataFrame(rows[:QA_SAMPLE_SIZE])

    missing_records = set(
        missing_wh_df["patient_record"].astype(str).map(_normalize_patient_record)
    )
    missing_patients = patients_df[
        patients_df["patient_record"]
        .astype(str)
        .map(_normalize_patient_record)
        .isin(missing_records)
    ].copy()
    if not missing_patients.empty:
        for _, row in _sample_rows(
            missing_patients, QA_SAMPLE_SIZE - len(rows)
        ).iterrows():
            rows.append(
                {
                    "qa_category": "missing_vitals",
                    "patient_record": _normalize_patient_record(row["patient_record"]),
                    "source_file": row["source_file"],
                    "detail": "missing weight/height tuple",
                    "context_snippet": "",
                }
            )

    return pd.DataFrame(rows)


def _build_non_standard_header_samples(
    patients_df: pd.DataFrame,
    header_files: list[str],
    remaining: int,
) -> pd.DataFrame:
    if remaining <= 0 or not header_files:
        return pd.DataFrame()

    subset = patients_df[patients_df["source_file"].isin(header_files)].copy()
    if subset.empty:
        subset = pd.DataFrame({"source_file": header_files})
        subset["patient_record"] = ""
        subset["exclusion_reason"] = "non_standard_header"

    sampled = _sample_rows(subset, remaining)
    return pd.DataFrame(
        {
            "qa_category": "non_standard_header",
            "patient_record": sampled.get("patient_record", pd.Series(dtype=str))
            .fillna("")
            .map(lambda value: _normalize_patient_record(value) if value else ""),
            "source_file": sampled["source_file"],
            "detail": "IDs parsed via regex fallback; manual header review",
            "context_snippet": "",
        }
    )


def generate_qa_sample(text_dir: Path | None = None) -> pd.DataFrame:
    """Write stratified manual-review sample (~30 records) to qa_sample.csv."""
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    kidney_df = pd.read_csv(config.KIDNEY_FUNCTION_CSV, dtype=str)
    bleeding_df = pd.read_csv(config.CHECKPOINT_BLEEDING_CSV, dtype=str)
    patients_df = pd.read_csv(config.PATIENTS_CSV, dtype=str)
    missing_wh_df = pd.read_csv(config.CHECKPOINT_MISSING_WH_CSV, dtype=str)

    non_standard_headers = find_non_standard_header_files(text_dir)

    parts = [
        _build_egfr_mismatch_samples(kidney_df),
        _build_bleeding_samples(bleeding_df),
        _build_exclusion_samples(patients_df, missing_wh_df),
    ]
    qa_df = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    remaining = (QA_SAMPLE_SIZE * 3) - len(qa_df)
    header_part = _build_non_standard_header_samples(
        patients_df,
        non_standard_headers,
        remaining,
    )
    if not header_part.empty:
        qa_df = pd.concat([qa_df, header_part], ignore_index=True)
    if qa_df.empty:
        qa_df = pd.DataFrame(
            columns=[
                "qa_category",
                "patient_record",
                "source_file",
                "detail",
                "context_snippet",
            ]
        )

    qa_df.to_csv(config.QA_SAMPLE_CSV, index=False)
    return qa_df


def _count_text_files(text_dir: Path) -> int:
    return len(list(text_dir.glob("*.txt")))


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_pipeline_summary(text_dir: Path | None = None) -> dict:
    """Aggregate extraction, kidney, and bleeding counts into pipeline_summary.json."""
    text_dir = text_dir or config.TEXT_DIR
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    patients_df = pd.read_csv(config.PATIENTS_CSV, dtype=str)
    labs_df = pd.read_csv(config.LABS_LONG_CSV)
    vitals_df = pd.read_csv(config.VITALS_WH_CSV)
    meds_df = pd.read_csv(config.MEDICATIONS_LONG_CSV, dtype=str)
    kidney_df = pd.read_csv(config.KIDNEY_FUNCTION_CSV, dtype=str)
    triggers_df = pd.read_csv(config.CHECKPOINT_TRIGGERS_CSV, dtype=str)
    bleeding_df = pd.read_csv(config.CHECKPOINT_BLEEDING_CSV, dtype=str)
    analysis_df = pd.read_csv(config.BLEEDING_DISCORDANCE_CSV, dtype=str)
    missing_wh_df = pd.read_csv(config.CHECKPOINT_MISSING_WH_CSV, dtype=str)
    excluded_df = pd.read_csv(config.CHECKPOINT_EXCLUDED_CSV, dtype=str)
    qa_df = pd.read_csv(config.QA_SAMPLE_CSV, dtype=str) if config.QA_SAMPLE_CSV.exists() else pd.DataFrame()

    non_standard_headers = find_non_standard_header_files(text_dir)
    non_excluded = patients_df[patients_df["excluded"] == "no"]
    acute_bleeding = bleeding_df[bleeding_df["event_class"] == "bleeding_event"]

    kidney_cohort = kidney_df[kidney_df["excluded"] == "no"]
    nearest_kidney = (
        kidney_cohort.sort_values("date_return")
        .groupby("patient_record", as_index=False)
        .first()
    )

    summary = {
        "input": {
            "text_files": _count_text_files(text_dir),
            "non_standard_header_files": len(non_standard_headers),
            "non_standard_header_file_list": non_standard_headers,
        },
        "patients": {
            "total_admissions": int(len(patients_df)),
            "unique_patient_id": int(patients_df["patient_id"].nunique()),
            "excluded": int(len(excluded_df)),
            "non_excluded": int(len(non_excluded)),
            "doac_drug_counts": patients_df["doac_drug"]
            .fillna("unknown")
            .value_counts()
            .to_dict(),
        },
        "extraction": {
            "lab_rows": int(len(labs_df)),
            "lab_types": labs_df["lab_type"].value_counts().to_dict(),
            "vitals_rows": int(len(vitals_df)),
            "medication_rows": int(len(meds_df)),
            "missing_weight_height": int(len(missing_wh_df)),
            "pgp_inhibitor_records": int(
                (meds_df["is_pgp_inhibitor"].str.lower() == "yes").sum()
            ),
            "cyp3a4_inhibitor_records": int(
                (meds_df["is_cyp3a4_inhibitor"].str.lower() == "yes").sum()
            ),
        },
        "kidney_function": {
            "creatinine_rows": int(len(kidney_cohort)),
            "egfr_lab_recalc_mismatch_rows": int(
                (kidney_cohort["egfr_lab_recalc_mismatch"] == "yes").sum()
            ),
            "discordant_dose_recommendations": int(
                (nearest_kidney["discordant"] == "yes").sum()
            ),
            "dose_mismatch_ecrcl": int(
                (nearest_kidney["dose_mismatch_ecrcl"] == "yes").sum()
            ),
            "dose_mismatch_egfr": int(
                (nearest_kidney["dose_mismatch_egfr"] == "yes").sum()
            ),
        },
        "bleeding": {
            "trigger_rows": int(len(triggers_df)),
            "trigger_types": triggers_df["trigger_type"].value_counts().to_dict(),
            "bleeding_checkpoint_rows": int(len(bleeding_df)),
            "acute_bleeding_events": int(len(acute_bleeding)),
            "acute_bleeding_admissions": int(acute_bleeding["patient_record"].nunique()),
            "analysis_bleeding_yes": int((analysis_df["bleeding_event"] == "yes").sum()),
            "analysis_high_risk_trigger_yes": int(
                (analysis_df["high_risk_trigger"] == "yes").sum()
            ),
        },
        "qa_sample": {
            "total_rows": int(len(qa_df)),
            "by_category": qa_df["qa_category"].value_counts().to_dict()
            if not qa_df.empty
            else {},
        },
        "downstream_checkpoints": {
            "kidney_function_filters": _load_json_if_exists(
                config.CHECKPOINT_KIDNEY_FUNCTION_JSON
            ),
            "bleeding_cohort_summary": _load_json_if_exists(
                config.CHECKPOINT_BLEEDING_COHORT_JSON
            ),
            "dose_discordance": _load_json_if_exists(
                config.CHECKPOINT_DOSE_DISCORDANCE_JSON
            ),
        },
    }

    with config.PIPELINE_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return summary


def run_qa(text_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Generate QA sample CSV and pipeline summary JSON."""
    qa_df = generate_qa_sample(text_dir)
    summary = generate_pipeline_summary(text_dir)
    _print_qa_summary(qa_df, summary)
    return qa_df, summary


def _print_qa_summary(qa_df: pd.DataFrame, summary: dict) -> None:
    print("Pipeline QA summary:")
    print(f"  QA sample rows: {len(qa_df)} -> {config.QA_SAMPLE_CSV}")
    if not qa_df.empty:
        for category, count in qa_df["qa_category"].value_counts().items():
            print(f"    {category}: {count}")
    print(f"  Pipeline summary -> {config.PIPELINE_SUMMARY_JSON}")
    print(
        "  Non-standard header files for manual review: "
        f"{summary['input']['non_standard_header_files']}"
    )


if __name__ == "__main__":
    run_qa()
