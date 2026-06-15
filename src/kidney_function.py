"""Compute eCrCL, eGFR, and EHRA dose recommendations."""

from __future__ import annotations

import json
import math
from typing import Literal

import pandas as pd

from src import config

MetricKind = Literal["ecrcl", "egfr"]
EGFR_LAB_RECALC_TOLERANCE = 1.0


def bsa_du_bois(height_cm: float, weight_kg: float) -> float:
    """Body surface area (m²) via Du Bois formula."""
    return math.sqrt(height_cm * weight_kg / 3600.0)


def creatinine_umol_to_mg_dl(creatinine_umol_l: float) -> float:
    return creatinine_umol_l / config.CREATININE_UMOL_TO_MG_DL


def cockcroft_gault(
    creatinine_umol_l: float,
    age: float,
    weight_kg: float,
    sex: str,
) -> float:
    """Cockcroft-Gault creatinine clearance (mL/min, unadjusted)."""
    scr_mg_dl = creatinine_umol_to_mg_dl(creatinine_umol_l)
    if scr_mg_dl <= 0:
        return float("nan")
    clearance = (140.0 - age) * weight_kg / (72.0 * scr_mg_dl)
    if sex == "nữ":
        clearance *= 0.85
    return clearance


def ckd_epi_2009(
    creatinine_umol_l: float,
    age: float,
    sex: str,
) -> float:
    """CKD-EPI 2009 indexed eGFR (mL/min/1.73 m²)."""
    scr_mg_dl = creatinine_umol_to_mg_dl(creatinine_umol_l)
    if scr_mg_dl <= 0:
        return float("nan")

    if sex == "nữ":
        kappa = 0.7
        alpha = -0.329
        sex_factor = 1.018
    else:
        kappa = 0.9
        alpha = -0.411
        sex_factor = 1.0

    ratio = scr_mg_dl / kappa
    egfr = (
        141.0
        * min(ratio, 1.0) ** alpha
        * max(ratio, 1.0) ** -1.209
        * 0.993 ** age
        * sex_factor
    )
    return egfr


def deindex_egfr(egfr_indexed: float, bsa_m2: float) -> float:
    """Convert indexed eGFR to absolute mL/min."""
    return egfr_indexed * bsa_m2 / config.BSA_INDEX_M2


def _kidney_clearance_in_reduction_band(
    doac: str,
    clearance: float,
) -> bool:
    """Return True when clearance metric triggers dose reduction."""
    if math.isnan(clearance):
        return False

    if doac in ("edoxaban", "rivaroxaban"):
        return (
            config.ECRCL_EDOXABAN_RIVAROXABAN_LOW
            <= clearance
            <= config.ECRCL_EDOXABAN_RIVAROXABAN_HIGH
        )
    if doac == "dabigatran":
        return (
            config.ECRCL_DABIGATRAN_LOW
            <= clearance
            <= config.ECRCL_DABIGATRAN_HIGH
        )
    return False


def _is_contraindicated(
    doac: str,
    clearance: float,
) -> bool:
    if math.isnan(clearance):
        return False

    rules = config.DOAC_DOSE_RULES[doac]
    threshold = rules["contraindicated_below_ecrcl"]
    return clearance < threshold


def _evaluate_reduction_triggers(
    doac: str,
    *,
    clearance: float,
    weight_kg: float | None,
    age: float | None,
    creatinine_umol_l: float,
    has_pgp_inhibitor: bool,
) -> dict[str, bool]:
    """Evaluate EHRA 2021 reduction criteria for one kidney metric track."""
    clearance_band = _kidney_clearance_in_reduction_band(doac, clearance)
    return {
        "weight_le_60": (
            weight_kg is not None
            and not math.isnan(weight_kg)
            and weight_kg <= config.WEIGHT_REDUCTION_KG
        ),
        "age_ge_80": (
            age is not None
            and not math.isnan(age)
            and age >= config.AGE_REDUCTION_YEARS
        ),
        "creatinine_ge_133": creatinine_umol_l >= config.CREATININE_APIXABAN_UMOL,
        "ecrcl_15_49": clearance_band,
        "ecrcl_30_49": clearance_band,
        "pgp_inhibitor": has_pgp_inhibitor,
    }


def _should_reduce_dose(doac: str, triggers: dict[str, bool]) -> bool:
    rules = config.DOAC_DOSE_RULES[doac]
    active = [triggers[name] for name in rules["reduction_triggers"]]

    logic = rules["trigger_logic"]
    if logic == "any_1_of_3":
        return any(active)
    if logic == "any_2_of_3":
        return sum(active) >= 2
    return all(active)


def recommended_dose(
    doac: str,
    *,
    clearance: float,
    metric: MetricKind,
    weight_kg: float | None,
    age: float | None,
    creatinine_umol_l: float,
    has_pgp_inhibitor: bool,
) -> str:
    """Return EHRA 2021 recommended dose label for one kidney metric track."""
    if doac not in config.DOAC_DOSE_RULES:
        return ""

    if _is_contraindicated(doac, clearance):
        return "contraindicated"

    rules = config.DOAC_DOSE_RULES[doac]
    triggers = _evaluate_reduction_triggers(
        doac,
        clearance=clearance,
        weight_kg=weight_kg,
        age=age,
        creatinine_umol_l=creatinine_umol_l,
        has_pgp_inhibitor=has_pgp_inhibitor,
    )
    if _should_reduce_dose(doac, triggers):
        return rules["reduced_dose"]
    return rules["standard_dose"]


def _extract_dose_mg(dose: str | None) -> float | None:
    if not dose or pd.isna(dose):
        return None
    dose_norm = str(dose).strip().lower().replace(",", ".")
    mg_match = pd.Series([dose_norm]).str.extract(
        r"(\d+(?:\.\d+)?)\s*mg", expand=False
    )[0]
    if pd.isna(mg_match):
        return None
    return float(mg_match)


def _canonical_prescribed_dose(doac: str, dose: str | None) -> str | None:
    """Map parsed medication dose strings to EHRA canonical labels."""
    mg = _extract_dose_mg(dose)
    if mg is None:
        return None

    mapping: dict[str, dict[float, str]] = {
        "edoxaban": {60.0: "60 mg/day", 30.0: "30 mg/day"},
        "apixaban": {5.0: "5 mg BID", 2.5: "2.5 mg BID"},
        "rivaroxaban": {20.0: "20 mg/day", 15.0: "15 mg/day"},
        "dabigatran": {150.0: "150 mg BID", 110.0: "110 mg BID"},
    }
    return mapping.get(doac, {}).get(mg)


def _classify_doac_dose(
    doac: str,
    dose: str | None,
) -> tuple[str | None, str | None]:
    """Return (prescribed_dose_label, dose_flag) for one parsed order dose."""
    canonical = _canonical_prescribed_dose(doac, dose)
    if canonical:
        return canonical, None

    mg = _extract_dose_mg(dose)
    if mg is None:
        return None, None

    if doac == "rivaroxaban" and mg == 10.0:
        return config.DOSE_FLAG_PROPHYLAXIS, config.DOSE_FLAG_PROPHYLAXIS
    if doac == "rivaroxaban" and mg == 2.5:
        return config.DOSE_FLAG_PAD, config.DOSE_FLAG_PAD
    if doac == "edoxaban" and mg == 15.0:
        return "15 mg/day", config.DOSE_FLAG_NOT_APPROVED

    return None, None


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(
        config.PATIENTS_CSV,
        parse_dates=["date_admission", "date_discharge"],
    )
    labs = pd.read_csv(config.LABS_LONG_CSV, parse_dates=["date_return"])
    vitals = pd.read_csv(config.VITALS_WH_CSV, parse_dates=["date_measured"])
    meds = pd.read_csv(
        config.MEDICATIONS_LONG_CSV,
        parse_dates=["date_ordered", "date_active"],
    )
    creatinine = labs[labs["lab_type"] == "creatinine"].copy()
    egfr_labs = labs[labs["lab_type"] == "egfr_2009_indexed"].copy()
    return patients, creatinine, egfr_labs, vitals, meds


def _admission_med_window(
    patient: pd.Series,
    meds: pd.DataFrame,
    *,
    drug_generic: str | None = None,
) -> pd.DataFrame:
    record = patient["patient_record"]
    window = meds[(meds["patient_record"] == record) & (meds["is_stopped"] == "no")]
    if drug_generic:
        window = window[window["drug_generic"] == drug_generic]

    admission = patient["date_admission"]
    discharge = patient["date_discharge"]
    if not pd.isna(admission) and not pd.isna(discharge):
        window = window[
            (window["date_active"] >= admission)
            & (window["date_active"] <= discharge)
        ]
    return window


def _generic_matches_markers(generic: str | None, markers: tuple[str, ...]) -> bool:
    if not generic or pd.isna(generic):
        return False
    generic_norm = str(generic).lower()
    return any(marker in generic_norm for marker in markers)


def _triple_therapy_flags(
    patients: pd.DataFrame,
    meds: pd.DataFrame,
) -> pd.DataFrame:
    """Flag admissions with concurrent DOAC + aspirin + clopidogrel."""
    rows: list[dict] = []
    for _, patient in patients.iterrows():
        record = patient["patient_record"]
        window = _admission_med_window(patient, meds)
        has_doac = bool(window["drug_generic"].isin(config.DOAC_DRUGS).any())
        has_aspirin = bool(
            window["drug_generic"]
            .apply(lambda g: _generic_matches_markers(g, config.ASPIRIN_GENERIC_MARKERS))
            .any()
        )
        has_clopidogrel = bool(
            window["drug_generic"]
            .apply(
                lambda g: _generic_matches_markers(g, config.CLOPIDOGREL_GENERIC_MARKERS)
            )
            .any()
        )
        rows.append(
            {
                "patient_record": record,
                "triple_therapy": (
                    "yes" if has_doac and has_aspirin and has_clopidogrel else "no"
                ),
            }
        )
    return pd.DataFrame(rows)


def _inhibitor_flags(
    patients: pd.DataFrame,
    meds: pd.DataFrame,
) -> pd.DataFrame:
    """Per patient_record: concurrent P-gp / CYP3A4 inhibitor during admission."""
    rows: list[dict] = []
    for _, patient in patients.iterrows():
        record = patient["patient_record"]
        admission = patient["date_admission"]
        discharge = patient["date_discharge"]
        if pd.isna(admission) or pd.isna(discharge):
            rows.append(
                {
                    "patient_record": record,
                    "has_pgp_inhibitor": False,
                    "has_cyp3a4_inhibitor": False,
                }
            )
            continue

        window = _admission_med_window(patient, meds)
        rows.append(
            {
                "patient_record": record,
                "has_pgp_inhibitor": bool((window["is_pgp_inhibitor"] == "yes").any()),
                "has_cyp3a4_inhibitor": bool(
                    (window["is_cyp3a4_inhibitor"] == "yes").any()
                ),
            }
        )
    return pd.DataFrame(rows)


def _prescribed_doac_doses(
    patients: pd.DataFrame,
    meds: pd.DataFrame,
) -> pd.DataFrame:
    """Most frequent prescribed DOAC dose (or special dose flag) per admission."""
    rows: list[dict] = []
    for _, patient in patients.iterrows():
        record = patient["patient_record"]
        doac = patient["doac_drug"]
        if pd.isna(doac) or doac not in config.DOAC_DOSE_RULES:
            rows.append(
                {
                    "patient_record": record,
                    "doac_prescribed_dose": None,
                    "doac_dose_flag": None,
                }
            )
            continue

        window = _admission_med_window(patient, meds, drug_generic=doac)
        classified = [
            _classify_doac_dose(doac, row_dose) for row_dose in window["dose"]
        ]
        labels = [label for label, _ in classified if label]

        if not labels:
            rows.append(
                {
                    "patient_record": record,
                    "doac_prescribed_dose": None,
                    "doac_dose_flag": None,
                }
            )
            continue

        mode_label = pd.Series(labels).mode().iloc[0]
        mode_flag = _dose_flag_for_label(doac, mode_label)

        rows.append(
            {
                "patient_record": record,
                "doac_prescribed_dose": mode_label,
                "doac_dose_flag": mode_flag,
            }
        )
    return pd.DataFrame(rows)


def _dose_flag_for_label(doac: str, label: str | None) -> str | None:
    if not label:
        return None
    if label == config.DOSE_FLAG_PROPHYLAXIS:
        return config.DOSE_FLAG_PROPHYLAXIS
    if label == config.DOSE_FLAG_PAD:
        return config.DOSE_FLAG_PAD
    if doac == "edoxaban" and label == "15 mg/day":
        return config.DOSE_FLAG_NOT_APPROVED
    return None


def _join_nearest_vitals(
    creatinine: pd.DataFrame,
    vitals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach nearest weight/height on or before creatinine date; else admission vitals."""
    pieces: list[pd.DataFrame] = []
    for record, creat_group in creatinine.groupby("patient_record", sort=False):
        vit_group = vitals[vitals["patient_record"] == record].sort_values(
            "date_measured"
        )
        creat_group = creat_group.sort_values("date_return").copy()

        if vit_group.empty:
            creat_group["weight_kg"] = float("nan")
            creat_group["height_cm"] = float("nan")
            creat_group["vitals_date_used"] = pd.NaT
            creat_group["vitals_join_method"] = "missing"
            pieces.append(creat_group)
            continue

        fallback = vit_group.iloc[0]
        for idx, row in creat_group.iterrows():
            on_or_before = vit_group[vit_group["date_measured"] <= row["date_return"]]
            if on_or_before.empty:
                chosen = fallback
                method = "admission_fallback"
            else:
                chosen = on_or_before.iloc[-1]
                method = "on_or_before"

            creat_group.at[idx, "weight_kg"] = chosen["weight_kg"]
            creat_group.at[idx, "height_cm"] = chosen["height_cm"]
            creat_group.at[idx, "vitals_date_used"] = chosen["date_measured"]
            creat_group.at[idx, "vitals_join_method"] = method

        pieces.append(creat_group)

    return pd.concat(pieces, ignore_index=True)


def _attach_lab_egfr(creatinine: pd.DataFrame, egfr_labs: pd.DataFrame) -> pd.DataFrame:
    egfr_lookup = egfr_labs.rename(
        columns={"value": "egfr_2009_indexed_lab", "date_return": "egfr_lab_date"}
    )[
        ["patient_record", "egfr_lab_date", "egfr_2009_indexed_lab", "source_file"]
    ].rename(columns={"source_file": "egfr_source_file"})

    merged = creatinine.merge(
        egfr_lookup,
        left_on=["patient_record", "date_return"],
        right_on=["patient_record", "egfr_lab_date"],
        how="left",
    )
    merged = merged.drop(columns=["egfr_lab_date"])
    return merged


def _build_kidney_rows(
    patients: pd.DataFrame,
    creatinine: pd.DataFrame,
    egfr_labs: pd.DataFrame,
    vitals: pd.DataFrame,
    meds: pd.DataFrame,
) -> pd.DataFrame:
    inhibitors = _inhibitor_flags(patients, meds)
    prescribed = _prescribed_doac_doses(patients, meds)
    triple = _triple_therapy_flags(patients, meds)

    base = _join_nearest_vitals(creatinine, vitals)
    base = _attach_lab_egfr(base, egfr_labs)
    base = base.merge(
        patients[
            [
                "patient_record",
                "patient_id",
                "age",
                "sex",
                "doac_drug",
                "excluded",
                "date_admission",
            ]
        ],
        on="patient_record",
        how="left",
    )
    base = base.merge(inhibitors, on="patient_record", how="left")
    base = base.merge(prescribed, on="patient_record", how="left")
    base = base.merge(triple, on="patient_record", how="left")

    output_rows: list[dict] = []
    for _, row in base.iterrows():
        weight = row["weight_kg"]
        height = row["height_cm"]
        age = row["age"]
        sex = row["sex"]
        creat_umol = float(row["value"])
        doac = row["doac_drug"]

        bsa = (
            bsa_du_bois(float(height), float(weight))
            if pd.notna(weight) and pd.notna(height)
            else float("nan")
        )
        ecrcl = (
            cockcroft_gault(creat_umol, float(age), float(weight), sex)
            if pd.notna(weight) and pd.notna(age) and pd.notna(sex)
            else float("nan")
        )
        egfr_recalc = (
            ckd_epi_2009(creat_umol, float(age), sex)
            if pd.notna(age) and pd.notna(sex)
            else float("nan")
        )

        egfr_lab = row["egfr_2009_indexed_lab"]
        if pd.notna(egfr_lab):
            egfr_indexed = float(egfr_lab)
            egfr_source = "lab"
        else:
            egfr_indexed = egfr_recalc
            egfr_source = "recalculated"

        egfr_absolute = (
            deindex_egfr(egfr_indexed, bsa) if pd.notna(bsa) and pd.notna(egfr_indexed) else float("nan")
        )

        mismatch = "no"
        delta = float("nan")
        if pd.notna(egfr_lab) and pd.notna(egfr_recalc):
            delta = abs(float(egfr_lab) - egfr_recalc)
            if delta > EGFR_LAB_RECALC_TOLERANCE:
                mismatch = "yes"

        has_pgp = bool(row["has_pgp_inhibitor"])
        has_cyp = bool(row["has_cyp3a4_inhibitor"])

        ecrcl_rec = ""
        egfr_rec = ""
        ecrcl_contra = "no"
        egfr_contra = "no"
        if doac in config.DOAC_DOSE_RULES:
            ecrcl_rec = recommended_dose(
                doac,
                clearance=ecrcl,
                metric="ecrcl",
                weight_kg=float(weight) if pd.notna(weight) else None,
                age=float(age) if pd.notna(age) else None,
                creatinine_umol_l=creat_umol,
                has_pgp_inhibitor=has_pgp,
            )
            egfr_rec = recommended_dose(
                doac,
                clearance=egfr_absolute,
                metric="egfr",
                weight_kg=float(weight) if pd.notna(weight) else None,
                age=float(age) if pd.notna(age) else None,
                creatinine_umol_l=creat_umol,
                has_pgp_inhibitor=has_pgp,
            )
            ecrcl_contra = "yes" if ecrcl_rec == "contraindicated" else "no"
            egfr_contra = "yes" if egfr_rec == "contraindicated" else "no"

        prescribed_dose = row["doac_prescribed_dose"]
        dose_flag = row["doac_dose_flag"]
        triple_therapy = row.get("triple_therapy", "no")
        ehra_canonical = prescribed_dose in {
            rules["standard_dose"]
            for rules in config.DOAC_DOSE_RULES.values()
        } | {
            rules["reduced_dose"]
            for rules in config.DOAC_DOSE_RULES.values()
        }
        mismatch_ecrcl = "no"
        mismatch_egfr = "no"
        if (
            prescribed_dose
            and ehra_canonical
            and ecrcl_rec
            and ecrcl_rec != "contraindicated"
        ):
            mismatch_ecrcl = "yes" if prescribed_dose != ecrcl_rec else "no"
        if (
            prescribed_dose
            and ehra_canonical
            and egfr_rec
            and egfr_rec != "contraindicated"
        ):
            mismatch_egfr = "yes" if prescribed_dose != egfr_rec else "no"

        discordant = "no"
        if (
            ecrcl_rec
            and egfr_rec
            and ecrcl_rec != "contraindicated"
            and egfr_rec != "contraindicated"
            and ecrcl_rec != egfr_rec
        ):
            discordant = "yes"

        output_rows.append(
            {
                "patient_record": row["patient_record"],
                "patient_id": row["patient_id"],
                "source_file": row["source_file"],
                "date_return": row["date_return"],
                "creatinine_umol_l": creat_umol,
                "age": age,
                "sex": sex,
                "weight_kg": weight,
                "height_cm": height,
                "vitals_date_used": row["vitals_date_used"],
                "vitals_join_method": row["vitals_join_method"],
                "bsa_m2": round(bsa, 4) if pd.notna(bsa) else pd.NA,
                "ecrcl_cg_ml_min": round(ecrcl, 2) if pd.notna(ecrcl) else pd.NA,
                "egfr_2009_indexed_recalc": (
                    round(egfr_recalc, 2) if pd.notna(egfr_recalc) else pd.NA
                ),
                "egfr_2009_indexed_lab": (
                    round(float(egfr_lab), 2) if pd.notna(egfr_lab) else pd.NA
                ),
                "egfr_2009_indexed": (
                    round(egfr_indexed, 2) if pd.notna(egfr_indexed) else pd.NA
                ),
                "egfr_source": egfr_source,
                "egfr_lab_recalc_mismatch": mismatch,
                "egfr_lab_recalc_delta": round(delta, 2) if pd.notna(delta) else pd.NA,
                "egfr_2009_absolute": (
                    round(egfr_absolute, 2) if pd.notna(egfr_absolute) else pd.NA
                ),
                "doac_drug": doac,
                "has_pgp_inhibitor": "yes" if has_pgp else "no",
                "has_cyp3a4_inhibitor": "yes" if has_cyp else "no",
                "doac_prescribed_dose": prescribed_dose,
                "doac_dose_flag": dose_flag if pd.notna(dose_flag) else "",
                "triple_therapy": triple_therapy,
                "ecrcl_recommended_dose": ecrcl_rec,
                "egfr_recommended_dose": egfr_rec,
                "ecrcl_contraindicated": ecrcl_contra,
                "egfr_contraindicated": egfr_contra,
                "dose_mismatch_ecrcl": mismatch_ecrcl,
                "dose_mismatch_egfr": mismatch_egfr,
                "discordant": discordant,
                "excluded": row["excluded"],
            }
        )

    return pd.DataFrame(output_rows)


def _is_blank_dose(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _exclude_from_export_mask(df: pd.DataFrame) -> pd.Series:
    blank = df["doac_prescribed_dose"].apply(_is_blank_dose)
    excluded_flag = df["doac_dose_flag"].isin(config.DOSE_FLAGS_EXCLUDED_FROM_EXPORT)
    excluded_label = df["doac_prescribed_dose"].isin(config.DOSE_FLAGS_EXCLUDED_FROM_EXPORT)
    return blank | excluded_flag | excluded_label


def _cohort_slice_summary(df: pd.DataFrame, mask: pd.Series) -> dict:
    subset = df[mask]
    if subset.empty:
        return {
            "admissions": 0,
            "creatinine_rows": 0,
            "patient_records": [],
            "source_files": [],
            "by_doac_drug": {},
        }

    admissions = subset.groupby("patient_record", as_index=False).first()
    return {
        "admissions": int(admissions["patient_record"].nunique()),
        "creatinine_rows": int(len(subset)),
        "patient_records": sorted(admissions["patient_record"].astype(str).tolist()),
        "source_files": sorted(admissions["source_file"].astype(str).unique().tolist()),
        "by_doac_drug": (
            admissions["doac_drug"].value_counts().astype(int).to_dict()
        ),
    }


def _build_filter_checkpoint(
    full_df: pd.DataFrame,
    export_df: pd.DataFrame,
) -> dict:
    prophylaxis_mask = (
        (full_df["doac_dose_flag"] == config.DOSE_FLAG_PROPHYLAXIS)
        | (full_df["doac_prescribed_dose"] == config.DOSE_FLAG_PROPHYLAXIS)
    )
    pad_mask = (
        (full_df["doac_dose_flag"] == config.DOSE_FLAG_PAD)
        | (full_df["doac_prescribed_dose"] == config.DOSE_FLAG_PAD)
    )
    not_approved_mask = full_df["doac_dose_flag"] == config.DOSE_FLAG_NOT_APPROVED
    triple_mask = full_df["triple_therapy"] == "yes"
    blank_mask = full_df["doac_prescribed_dose"].apply(_is_blank_dose)
    excluded_mask = _exclude_from_export_mask(full_df)

    admissions_before = int(full_df["patient_record"].nunique())
    admissions_after = int(export_df["patient_record"].nunique())

    return {
        "filtered_prophylaxis_dose": {
            "action": "excluded_from_export",
            "dose_flag": config.DOSE_FLAG_PROPHYLAXIS,
            **_cohort_slice_summary(full_df, prophylaxis_mask),
        },
        "filtered_PAD_dose": {
            "action": "excluded_from_export",
            "dose_flag": config.DOSE_FLAG_PAD,
            **_cohort_slice_summary(full_df, pad_mask),
        },
        "filtered_blank_prescribed_dose": {
            "action": "excluded_from_export",
            **_cohort_slice_summary(full_df, blank_mask),
        },
        "flagged_not_approved_dosage": {
            "action": "included_in_export",
            "dose_flag": config.DOSE_FLAG_NOT_APPROVED,
            **_cohort_slice_summary(full_df, not_approved_mask),
        },
        "flagged_triple_therapy": {
            "action": "included_in_export",
            "therapy_flag": "triple_therapy",
            **_cohort_slice_summary(full_df, triple_mask),
        },
        "export_summary": {
            "creatinine_rows_before_filter": int(len(full_df)),
            "creatinine_rows_after_filter": int(len(export_df)),
            "creatinine_rows_removed": int(excluded_mask.sum()),
            "admissions_before_filter": admissions_before,
            "admissions_after_filter": admissions_after,
            "admissions_removed": admissions_before - admissions_after,
        },
    }


def _write_filter_checkpoint(checkpoint: dict) -> None:
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with config.CHECKPOINT_KIDNEY_FUNCTION_JSON.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _filter_kidney_export(df: pd.DataFrame) -> pd.DataFrame:
    """Remove prophylaxis/PAD doses and blank prescribed doses from export CSV."""
    return df[~_exclude_from_export_mask(df)].copy()


def _print_summary(df: pd.DataFrame) -> None:
    """Print cohort-level kidney/dose summary statistics."""
    cohort = df[df["excluded"] == "no"].copy()
    if cohort.empty:
        print("Kidney function: no non-excluded records to summarize.")
        return

    nearest = (
        cohort.sort_values("date_return")
        .groupby("patient_record", as_index=False)
        .first()
    )
    discordant_records = nearest[nearest["discordant"] == "yes"]
    reclassified = int(len(discordant_records))
    total_records = int(nearest["patient_record"].nunique())
    pct_reclassified = (
        100.0 * reclassified / total_records if total_records else 0.0
    )

    print("Kidney function summary (nearest admission creatinine per record):")
    print(f"  Records analyzed: {total_records}")
    print(
        f"  Discordant eCrCL vs eGFR recommendation: "
        f"{reclassified} ({pct_reclassified:.1f}%)"
    )

    for drug in sorted(config.DOAC_DRUGS):
        drug_df = nearest[nearest["doac_drug"] == drug]
        if drug_df.empty:
            continue
        ecrcl_mm = drug_df[drug_df["dose_mismatch_ecrcl"] == "yes"]
        egfr_mm = drug_df[drug_df["dose_mismatch_egfr"] == "yes"]
        print(
            f"  {drug}: prescribed≠recommended "
            f"(eCrCL {len(ecrcl_mm)}/{len(drug_df)}, "
            f"eGFR {len(egfr_mm)}/{len(drug_df)})"
        )

    pgp_changed = nearest[
        (nearest["doac_drug"] == "edoxaban")
        & (nearest["has_pgp_inhibitor"] == "yes")
        & (nearest["ecrcl_recommended_dose"] == "30 mg/day")
    ]
    print(
        "  Edoxaban records with P-gp inhibitor driving 30 mg/day "
        f"(eCrCL track): {len(pgp_changed)}"
    )

    mismatch_labs = cohort[cohort["egfr_lab_recalc_mismatch"] == "yes"]
    print(
        f"  Creatinine rows with lab vs recalculated eGFR mismatch: "
        f"{len(mismatch_labs)}/{len(cohort)}"
    )


def compute_kidney_function() -> pd.DataFrame:
    """Join labs, vitals, and meds; write kidney_function.csv."""
    config.DATA_EXPORT.mkdir(parents=True, exist_ok=True)
    config.DATA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    patients, creatinine, egfr_labs, vitals, meds = _load_inputs()
    full_df = _build_kidney_rows(
        patients, creatinine, egfr_labs, vitals, meds
    )
    export_df = _filter_kidney_export(full_df)
    export_df["doac_dose_flag"] = export_df["doac_dose_flag"].fillna("")
    checkpoint = _build_filter_checkpoint(full_df, export_df)
    _write_filter_checkpoint(checkpoint)

    export_df.to_csv(config.KIDNEY_FUNCTION_CSV, index=False)
    _print_summary(export_df)
    return export_df


if __name__ == "__main__":
    compute_kidney_function()
