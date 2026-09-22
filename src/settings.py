"""Locked settings from the midpoint final audit (t02/t03)."""

REFERENCE_ALTITUDE = 60

COMMON_FIELDS = ["1", "2", "3"]

EXPECTED_CLASSES = [0, 1, 2, 3, 4]

CV_SPLITS = 5

CV_REPEATS = 10

RF_TREES = 300

RANDOM_SEED = 42

BOOTSTRAP_REPS = 3000

EPS = 1e-8

MIDPOINT = 0.50

GPF = "GlobalPF50"

CPF = "CorePF50"

PREV = "PGF50"

TAIL = "CPC90"

VRI = "VRI"

METHODS = {
    "VRI_Threshold": {
        "kind": "THRESHOLD", "features": [VRI],
        "representation": "VRI", "classifier": "Threshold",
        "role": "Legacy baseline",
    },
    "GlobalPF_Threshold": {
        "kind": "THRESHOLD", "features": [GPF],
        "representation": "Global penetration fraction", "classifier": "Threshold",
        "role": "Single physical threshold baseline",
    },
    "GlobalPF_LR": {
        "kind": "LR", "features": [GPF],
        "representation": "Global penetration fraction", "classifier": "LR",
        "role": "Single physical probabilistic baseline",
    },
    "CorePF_LR": {
        "kind": "LR", "features": [CPF],
        "representation": "Core penetration fraction", "classifier": "LR",
        "role": "Central penetration-magnitude component",
    },
    "Prevalence_LR": {
        "kind": "LR", "features": [PREV],
        "representation": "Penetration prevalence", "classifier": "LR",
        "role": "Spatial-prevalence component",
    },
    "LocalTail_LR": {
        "kind": "LR", "features": [TAIL],
        "representation": "Local deep-tail depth", "classifier": "LR",
        "role": "Localized tail component",
    },
    "MidpointCompact_LR": {
        "kind": "LR", "features": [CPF, PREV],
        "representation": "Core penetration fraction + penetration prevalence",
        "classifier": "LR",
        "role": "PRIMARY compact transferable representation",
    },
    "MidpointExtended_LR": {
        "kind": "LR", "features": [CPF, PREV, TAIL],
        "representation": "Core penetration fraction + penetration prevalence + local deep-tail depth",
        "classifier": "LR",
        "role": "Extended local-detail representation",
    },
    "MidpointCompact_RF": {
        "kind": "RF", "features": [CPF, PREV],
        "representation": "Core penetration fraction + penetration prevalence",
        "classifier": "RF",
        "role": "Classifier-flexibility control",
    },
}

ALTITUDE_METHODS = {
    "MidpointCompact_LR": METHODS["MidpointCompact_LR"],
    "MidpointExtended_LR": METHODS["MidpointExtended_LR"],
}

SENSOR_METHODS = {
    "GlobalPF_LR": METHODS["GlobalPF_LR"],
    "CorePF_LR": METHODS["CorePF_LR"],
    "MidpointCompact_LR": METHODS["MidpointCompact_LR"],
    "MidpointExtended_LR": METHODS["MidpointExtended_LR"],
}

PAIRWISE = [
    ("MidpointCompact_LR", "GlobalPF_Threshold",
     "compact representation vs global physical threshold"),
    ("MidpointExtended_LR", "MidpointCompact_LR",
     "does local deep-tail depth add to the midpoint compact representation?"),
    ("MidpointCompact_RF", "MidpointCompact_LR",
     "does RF improve the same midpoint compact representation?"),
]

ALTITUDES = [20, 40, 60, 80, 100, 120]

LOW_Z_PERCENTILE = 0.1

HIGH_Z_PERCENTILE = 99.9

XY_LOW_PERCENTILE = 1.0

XY_HIGH_PERCENTILE = 99.0

GRID_N = 5

GRID_MIN_POINTS = 20

GRID_MIN_VALID_CELLS = 6

PGF_LOCAL_THRESHOLD = 0.20

CIRCLE_CORE_C = 0.25

CORE_MIN_POINTS = 30

COLUMN_RADIAL_BANDS = 5

COLUMN_ANGULAR_SECTORS = 8

COLUMN_MIN_POINTS = 15

COLUMN_MIN_VALID = 8

THRESHOLDS = [MIDPOINT]
PROFILE_COLS = [f"p{i}" for i in range(1, 21)]

# Manuscript ancillary statistics used their own numerical tolerance.
CONTEXT_EPS = 1e-12
