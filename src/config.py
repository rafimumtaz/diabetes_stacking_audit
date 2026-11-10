from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_dataset00.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN = "Target"
SELECTED_CLASSES = [
    "Prediabetic",
    "Type 1 Diabetes",
    "Type 2 Diabetes",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Daftar investigasi awal untuk skenario ablation.
# Ini BUKAN daftar fitur yang terbukti menyebabkan leakage.
# Ubah setelah menelaah dokumentasi definisi fitur dan pembentukan label.
POTENTIALLY_LABEL_PROXIMAL_FEATURES = [
    "Age",
    "BMI",
    "Insulin Levels",
    "Blood Glucose Levels",
    "Autoantibodies",
    "Genetic Markers",
    "Glucose Tolerance Test",
    "Pancreatic Health",
]

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "class_weight": None,
}

GB_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.08,
    "max_depth": 3,
    "random_state": RANDOM_STATE,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
