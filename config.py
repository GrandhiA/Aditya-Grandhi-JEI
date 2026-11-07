from pathlib import Path
from dataclasses import dataclass

DATASET = "Master_with_Engineered_Biomarkers.csv"
TARGET = "Mortality"
DROP_COLUMNS = [ ]
PATIENT_ID_COLUMN = None
TEST_SIZE = 0.20
RANDOM_STATE = 42
ONE_HOT_ENCODE = True
SCALE_FEATURES = True
IMPUTE_MISSING = False
MODEL = "logistic"

MODEL_SETTINGS = {
    "logistic": {

        "max_iter": 5000,
        "class_weight": "balanced",
        "solver": "lbfgs"
    },

    "rf": {
        "n_estimators": 500,
        "max_depth": None,
        "random_state": RANDOM_STATE
    },

    "xgb": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": RANDOM_STATE
    }

}

USE_CROSS_VALIDATION = True
CV_FOLDS = 5
USE_BOOTSTRAP = True
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
USE_PERMUTATION_IMPORTANCE = True
PERMUTATION_REPEATS = 30
USE_SHAP = True
MAX_SHAP_SAMPLES = 500
MAKE_CALIBRATION_CURVE = True
SAVE_ROC_CURVE = True
SAVE_PR_CURVE = True
SAVE_CONFUSION_MATRIX = True
SAVE_CORRELATION_HEATMAP = True
SAVE_PERMUTATION_PLOT = True
SAVE_SHAP_PLOTS = True
SAVE_CALIBRATION_PLOT = True
FIGURE_DPI = 300
OUTPUT_FOLDER = Path("Results")

SAVE = {
    "roc": True,
    "pr": True,
    "calibration": True,
    "confusion": True,
    "heatmap": True,
    "permutation": True,
    "shap": True,
    "metrics": True,
    "bootstrap": True,
    "predictions": True
}

SAVE_CROSS_VALIDATION = True
SAVE_CONFIGURATION = True

VERBOSE = {
    "dataset": True,
    "classes": True,
    "model": True,
    "features": True
}

REMOVE_CONSTANT_COLUMNS = True
CHECK_CORRELATION = True
CORRELATION_THRESHOLD = 0.95


from dataclasses import dataclass


@dataclass
class Config:
    # Dataset
    DATASET: str = DATASET
    TARGET: str = TARGET
    DROP_COLUMNS: list = None
    PATIENT_ID_COLUMN = PATIENT_ID_COLUMN

    # Train/Test
    TEST_SIZE: float = TEST_SIZE
    RANDOM_STATE: int = RANDOM_STATE

    # Preprocessing
    ONE_HOT_ENCODE: bool = ONE_HOT_ENCODE
    SCALE_FEATURES: bool = SCALE_FEATURES
    IMPUTE_MISSING: bool = IMPUTE_MISSING

    # Feature Selection
    REMOVE_CONSTANT_COLUMNS: bool = REMOVE_CONSTANT_COLUMNS
    CHECK_CORRELATION: bool = CHECK_CORRELATION
    CORRELATION_THRESHOLD: float = CORRELATION_THRESHOLD

    # Modeling
    MODEL: str = MODEL
    MODEL_SETTINGS: dict = None

    # Cross Validation
    USE_CROSS_VALIDATION: bool = USE_CROSS_VALIDATION
    CV_FOLDS: int = CV_FOLDS

    # Bootstrap
    USE_BOOTSTRAP: bool = USE_BOOTSTRAP
    BOOTSTRAP_ITERATIONS: int = BOOTSTRAP_ITERATIONS
    CONFIDENCE_LEVEL: float = CONFIDENCE_LEVEL

    # Explainability
    USE_PERMUTATION_IMPORTANCE: bool = USE_PERMUTATION_IMPORTANCE
    PERMUTATION_REPEATS: int = PERMUTATION_REPEATS
    USE_SHAP: bool = USE_SHAP
    MAX_SHAP_SAMPLES: int = MAX_SHAP_SAMPLES

    # Output
    OUTPUT_FOLDER = OUTPUT_FOLDER
    SAVE = SAVE
    VERBOSE = VERBOSE


config = Config()
config.DROP_COLUMNS = DROP_COLUMNS
config.MODEL_SETTINGS = MODEL_SETTINGS