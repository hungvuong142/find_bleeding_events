"""Project paths, regex patterns, DOAC thresholds, and keyword lists."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "text_vie_encoded"
DATA_DIR = PROJECT_ROOT / "data"
DATA_EXPORT = DATA_DIR / "export"
DATA_CHECKPOINTS = DATA_DIR / "checkpoints"
DATA_CONFIG = DATA_DIR / "config"

CYP3A4_INHIBITORS_CSV = DATA_CONFIG / "cyp3a4_inhibitors.csv"
PGP_INHIBITORS_CSV = DATA_CONFIG / "pgp_inhibitors.csv"

# Export outputs
PATIENTS_CSV = DATA_EXPORT / "patients.csv"
LABS_LONG_CSV = DATA_EXPORT / "labs_long.csv"
VITALS_WH_CSV = DATA_EXPORT / "vitals_weight_height.csv"
MEDICATIONS_LONG_CSV = DATA_EXPORT / "medications_long.csv"
DISCHARGE_MEDICATIONS_CSV = DATA_EXPORT / "discharge_medications.csv"
DOAC_DOSING_INPUTS_CSV = DATA_EXPORT / "doac_dosing_inputs.csv"
KIDNEY_FUNCTION_CSV = DATA_EXPORT / "kidney_function.csv"
BLEEDING_DISCORDANCE_CSV = DATA_EXPORT / "bleeding_discordance_analysis.csv"

# Checkpoints
CHECKPOINT_EXCLUDED_CSV = DATA_CHECKPOINTS / "checkpoint_excluded_age_or_pregnancy.csv"
CHECKPOINT_MISSING_WH_CSV = DATA_CHECKPOINTS / "checkpoint_missing_weight_height.csv"
CHECKPOINT_TRIGGERS_CSV = DATA_CHECKPOINTS / "checkpoint_high_risk_triggers.csv"
CHECKPOINT_BLEEDING_CSV = DATA_CHECKPOINTS / "checkpoint_bleeding_events.csv"
CHECKPOINT_DOAC_STOP_SAMPLES_CSV = (
    DATA_CHECKPOINTS / "checkpoint_doac_stop_samples.csv"
)
CHECKPOINT_MEDICATION_FREQUENCY_JSON = (
    DATA_CHECKPOINTS / "checkpoint_medication_frequency.json"
)
CHECKPOINT_DISCHARGE_FILTERED_JSON = (
    DATA_CHECKPOINTS / "checkpoint_discharge_filtered.json"
)
CHECKPOINT_KIDNEY_FUNCTION_JSON = (
    DATA_CHECKPOINTS / "checkpoint_kidney_function_filters.json"
)
CHECKPOINT_BLEEDING_COHORT_JSON = (
    DATA_CHECKPOINTS / "checkpoint_bleeding_cohort_summary.json"
)
CHECKPOINT_DOSE_DISCORDANCE_CSV = (
    DATA_CHECKPOINTS / "checkpoint_dose_discordance.csv"
)
CHECKPOINT_DOSE_DISCORDANCE_JSON = (
    DATA_CHECKPOINTS / "checkpoint_dose_discordance.json"
)
QA_SAMPLE_CSV = DATA_CHECKPOINTS / "qa_sample.csv"
PIPELINE_SUMMARY_JSON = DATA_CHECKPOINTS / "pipeline_summary.json"

# ---------------------------------------------------------------------------
# Patient ID regex (fallback when lines 3–4 are malformed)
# ---------------------------------------------------------------------------

RE_PATIENT_ID = r"mã bn:\s*(\d{10})"
RE_PATIENT_RECORD = r"mã đt:\s*(\d{12})"

# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

RE_CLINICAL_NOTE_SECTION = r"(?i)(?=tờ điều trị)"
RE_LAB_REPORT_HEADERS = (
    r"phiếu báo cáo kết quả xét nghiệm",
    r"phiếu kết quả xét nghiệm",
)

# ---------------------------------------------------------------------------
# Lab types and Vietnamese aliases
# ---------------------------------------------------------------------------

LAB_TYPES: dict[str, tuple[str, ...]] = {
    "creatinine": ("định lượng creatinin",),
    "hgb": ("hgb (hemoglobin)",),
    "hct": ("hct (hematocrit)",),
    "plt": ("plt (số lượng tiểu cầu)",),
    "pt_inr": ("pt - inr",),
    "pt_sec": ("pt (s)",),
    "aptt": ("aptt (s)",),
}

RE_LAB_DATE_RETURN = (
    r"thời gian duyệt kết quả",
    r"thời gian duyệt kq",
)

# ---------------------------------------------------------------------------
# DOAC drugs — brand/generic aliases → canonical generic name
# ---------------------------------------------------------------------------

DOAC_ALIASES: dict[str, str] = {
    "lixiana": "edoxaban",
    "edoxaban": "edoxaban",
    "rivacryst": "rivaroxaban",
    "xarelto": "rivaroxaban",
    "rivaroxaban": "rivaroxaban",
    "pradaxa": "dabigatran",
    "dabigatran": "dabigatran",
    "eliquis": "apixaban",
    "apixaban": "apixaban",
}

DOAC_DRUGS = frozenset(DOAC_ALIASES.values())

# Antiplatelet generics for triple-therapy detection (substring match)
ASPIRIN_GENERIC_MARKERS = ("acid acetylsalicylic", 'asa', 'aspirin')
CLOPIDOGREL_GENERIC_MARKERS = ("clopidogrel",)

# Non-standard DOAC dose flags (see kidney_function.py)
DOSE_FLAG_PROPHYLAXIS = "prophylaxis_dose" # rivaroxaban 2.5; 10 mg/day
DOSE_FLAG_PAD = "PAD_dose"
DOSE_FLAG_NOT_APPROVED = "not_approved_dosage" # Edoxaban 15 mg/day
DOSE_FLAGS_EXCLUDED_FROM_EXPORT = frozenset({DOSE_FLAG_PROPHYLAXIS, DOSE_FLAG_PAD})

# ---------------------------------------------------------------------------
# EHRA 2021 dose-reduction rules (standard dose, reduced dose)
# Pgp-inhibitors, CYP3A4 inhibitors: please check in 'data/config/cyp3a4_inhibitors.csv' and 'data/config/pgp_inhibitors.csv'
# ---------------------------------------------------------------------------

DOAC_DOSE_RULES: dict[str, dict] = {
    "edoxaban": {
        "standard_dose": "60 mg QD",
        "reduced_dose": "30 mg QD",
        "not_approved_dosage": "15 mg QD", # not approved dosage
        "reduction_triggers": ("weight_le_60", "ecrcl_15_49", "pgp_inhibitor"),
        "trigger_logic": "any_1_of_3",
        "contraindicated_below_ecrcl": 15,
    },
    "apixaban": {
        "standard_dose": "5 mg BID",
        "reduced_dose": "2.5 mg BID",
        "reduction_triggers": ("weight_le_60", "age_ge_80", "creatinine_ge_133"),
        "trigger_logic": "any_2_of_3",
        "contraindicated_below_ecrcl": 15,
    },
    "rivaroxaban": {
        "standard_dose": "20 mg QD",
        "reduced_dose": "15 mg QD",
        "prophylaxis_dose": "10 mg QD", # prohylaxis dose
        "pad_dose": "2.5 mg QD", # PAD dose
        "reduction_triggers": ("ecrcl_15_49",),
        "trigger_logic": "all",
        "contraindicated_below_ecrcl": 15,
    },
    "dabigatran": {
        "standard_dose": "150 mg BID",
        "reduced_dose": "110 mg BID",
        "reduction_triggers": ("ecrcl_30_49",),
        "trigger_logic": "all",
        "contraindicated_below_ecrcl": 30,
    },
}

# Threshold constants for dose rules
WEIGHT_REDUCTION_KG = 60
AGE_REDUCTION_YEARS = 80
CREATININE_APIXABAN_UMOL = 133
ECRCL_EDOXABAN_RIVAROXABAN_LOW = 15
ECRCL_EDOXABAN_RIVAROXABAN_HIGH = 49
ECRCL_DABIGATRAN_LOW = 30
ECRCL_DABIGATRAN_HIGH = 49
ECRCL_APIXABAN_LOW = 15
MIN_COHORT_AGE = 18 # exclude patients under 18 years old

# Creatinine unit conversion (µmol/L → mg/dL for Cockcroft-Gault)
CREATININE_UMOL_TO_MG_DL = 88.4 # convert µmol/L to mg/dL for Cockcroft-Gault
BSA_INDEX_M2 = 1.73 # normal body surface area in m2

# ---------------------------------------------------------------------------
# Exclusion — pregnancy (strict clinical patterns only)
# ---------------------------------------------------------------------------

PREGNANCY_EXCLUDE_PATTERNS = (
    r"\bthai\s+\d+\s*tuần\b",
    r"\b\d+\s*tuần\s+thai\b",
    r"\bivf\b"
)

PREGNANCY_BOILERPLATE_IGNORE = (
    r"chụp x quang:\s*chống chỉ định với phụ nữ có thai"
)

# ---------------------------------------------------------------------------
# Bleeding detection keywords (search inside tờ điều trị only)
# ---------------------------------------------------------------------------

BLEEDING_KEYWORDS = (
    "chảy máu",
    "xuất huyết",
    "phân đen",
    "dịch dẫn lưu hồng",
)

BLEEDING_INTERVENTION_KEYWORDS = (
    "kẹp",
    "clip",
    "mổ cầm máu",
)

BLEEDING_NEGATION_PATTERNS = (
    r"không xuất huyết",
    r"không chảy máu",
    r"không có dấu hiệu[^.]{0,40}xuất huyết",
    r"không có dấu hiệu[^.]{0,40}chảy máu",
)

# ---------------------------------------------------------------------------
# Paraclinical trigger thresholds
# ---------------------------------------------------------------------------

TRIGGER_INR_THRESHOLD = 5.0
TRIGGER_APTT_THRESHOLD_SEC = 100.0
TRIGGER_HCT_HGB_DROP_FRACTION = 0.25

DOAC_STOP_PATTERNS = (
    r"ngừng",
    r"dừng",
    r"tạm dừng",
    r"đình chỉ",
    r'thu hồi',
)

# ---------------------------------------------------------------------------
# PDF artifact cleanup
# ---------------------------------------------------------------------------

RE_PDF_ARTIFACT = r"\{signlibrary\.[^}]+\}"
