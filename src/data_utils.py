import pandas as pd
from .config import DATA_PATH, TARGET_COLUMN, SELECTED_CLASSES

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Letakkan diabetes_dataset00.csv di dalam folder data/."
        )

    # Dataset ini menggunakan header standar dan koma sebagai delimiter.
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom target '{TARGET_COLUMN}' tidak ditemukan. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    return df

def select_three_classes(df):
    selected = df[df[TARGET_COLUMN].isin(SELECTED_CLASSES)].copy()

    missing_classes = sorted(set(SELECTED_CLASSES) - set(selected[TARGET_COLUMN].unique()))
    if missing_classes:
        raise ValueError(f"Kelas berikut tidak ditemukan: {missing_classes}")

    selected[TARGET_COLUMN] = pd.Categorical(
        selected[TARGET_COLUMN],
        categories=SELECTED_CLASSES
    )
    selected[TARGET_COLUMN] = selected[TARGET_COLUMN].astype(str)

    return selected.reset_index(drop=True)

def split_feature_types(df, target_column):
    X = df.drop(columns=[target_column])
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return X, numeric_cols, categorical_cols
