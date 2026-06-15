"""CLI entrypoint for the DOAC kidney/bleeding extraction pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import config


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for path in (
        config.DATA_EXPORT,
        config.DATA_CHECKPOINTS,
        config.DATA_CONFIG,
    ):
        path.mkdir(parents=True, exist_ok=True)


def run_extract(text_dir: Path | None = None) -> None:
    """Run all structured extraction steps."""
    from src.extract_labs import extract_labs
    from src.extract_medications import extract_medications
    from src.extract_metadata import extract_patient_metadata
    from src.extract_vitals import extract_vitals

    print("Extracting patient metadata...")
    patients_df = extract_patient_metadata(text_dir)
    print(f"  {len(patients_df)} admissions -> {config.PATIENTS_CSV}")

    print("Extracting labs...")
    labs_df = extract_labs(text_dir)
    print(f"  {len(labs_df)} lab rows -> {config.LABS_LONG_CSV}")

    print("Extracting vitals...")
    vitals_df = extract_vitals(text_dir)
    print(f"  {len(vitals_df)} vitals rows -> {config.VITALS_WH_CSV}")

    print("Extracting medications...")
    meds_df, discharge_df = extract_medications(text_dir)
    print(f"  {len(meds_df)} medication rows -> {config.MEDICATIONS_LONG_CSV}")
    print(f"  {len(discharge_df)} discharge rows -> {config.DISCHARGE_MEDICATIONS_CSV}")


def run_kidney() -> None:
    from src.kidney_function import compute_kidney_function

    print("Computing kidney function and DOAC dose rules...")
    compute_kidney_function()


def run_bleeding(text_dir: Path | None = None) -> None:
    from src.bleeding_detect import detect_bleeding_and_triggers

    print("Detecting bleeding events and paraclinical triggers...")
    detect_bleeding_and_triggers(text_dir)


def run_analysis(text_dir: Path | None = None) -> None:
    from src.pipeline_qa import run_qa

    print("Generating QA sample and pipeline summary...")
    run_qa(text_dir)


def run_all(text_dir: Path | None = None) -> None:
    started = time.perf_counter()
    text_dir = text_dir or config.TEXT_DIR
    file_count = len(list(text_dir.glob("*.txt")))
    print(f"Running full pipeline on {file_count} text files in {text_dir}")

    run_extract(text_dir)
    run_kidney()
    run_bleeding(text_dir)
    run_analysis(text_dir)

    elapsed = time.perf_counter() - started
    print(f"Pipeline complete in {elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DOAC eCrCL vs eGFR extraction and bleeding analysis pipeline.",
    )
    parser.add_argument(
        "--step",
        choices=("all", "extract", "kidney", "bleeding", "analysis"),
        default="all",
        help="Pipeline step to run (default: all).",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=None,
        help=f"Input text directory (default: {config.TEXT_DIR}).",
    )
    args = parser.parse_args(argv)

    ensure_directories()
    text_dir = args.text_dir

    if args.step == "extract":
        run_extract(text_dir)
        return 0

    if args.step == "kidney":
        run_kidney()
        return 0

    if args.step == "bleeding":
        run_bleeding(text_dir)
        return 0

    if args.step == "analysis":
        run_analysis(text_dir)
        return 0

    if args.step == "all":
        run_all(text_dir)
        return 0

    print(f"Step '{args.step}' is not yet implemented.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
