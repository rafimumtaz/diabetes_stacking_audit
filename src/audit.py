import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-GUI agar aman di Windows/worker
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from .config import OUTPUT_DIR, TARGET_COLUMN, SELECTED_CLASSES
from .data_utils import split_feature_types

def _save_text(path, content):
    path.write_text(content, encoding="utf-8")

def run_dataset_audit(full_df, selected_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_df[TARGET_COLUMN].value_counts(dropna=False).rename_axis("class").reset_index(
        name="count"
    ).to_csv(OUTPUT_DIR / "class_distribution_all.csv", index=False)

    overview = []
    overview.append("DATASET OVERVIEW")
    overview.append("================")
    overview.append(f"Rows: {len(full_df)}")
    overview.append(f"Columns: {len(full_df.columns)}")
    overview.append(f"Duplicate rows: {int(full_df.duplicated().sum())}")
    overview.append(f"Missing cells: {int(full_df.isna().sum().sum())}")
    overview.append(f"Target column: {TARGET_COLUMN}")
    overview.append("\nDtypes:")
    overview.append(full_df.dtypes.astype(str).to_string())
    overview.append("\nMissing values by column:")
    overview.append(full_df.isna().sum().to_string())
    _save_text(OUTPUT_DIR / "dataset_overview.txt", "\n".join(overview))

    selected_df.to_csv(OUTPUT_DIR / "selected_three_class.csv", index=False)

    X, numeric_cols, categorical_cols = split_feature_types(selected_df, TARGET_COLUMN)
    y = selected_df[TARGET_COLUMN]

    if numeric_cols:
        selected_df.groupby(TARGET_COLUMN, observed=False)[numeric_cols].agg(
            ["count", "mean", "std", "min", "median", "max"]
        ).to_csv(OUTPUT_DIR / "numeric_summary_by_target.csv")

        range_rows = []
        for col in numeric_cols:
            for cls, group in selected_df.groupby(TARGET_COLUMN, observed=False):
                range_rows.append({
                    "feature": col,
                    "class": cls,
                    "min": group[col].min(),
                    "max": group[col].max(),
                    "n_unique": group[col].nunique(),
                })
        pd.DataFrame(range_rows).to_csv(
            OUTPUT_DIR / "numeric_feature_ranges_by_target.csv", index=False
        )

        corr = selected_df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix - Numeric Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_matrix_numeric.png", dpi=180)
        plt.close()

    if categorical_cols:
        cat_rows = []
        for col in categorical_cols:
            counts = (
                selected_df.groupby([TARGET_COLUMN, col], observed=False)
                .size().rename("count").reset_index()
            )
            counts.insert(0, "feature", col)
            cat_rows.append(counts)
        pd.concat(cat_rows, ignore_index=True).to_csv(
            OUTPUT_DIR / "categorical_feature_distribution.csv", index=False
        )

    # Mutual information: exploratory ranking only, not proof of leakage.
    X_mi = X.copy()
    for col in categorical_cols:
        X_mi[col] = X_mi[col].astype("string").fillna("__MISSING__")
        X_mi[col] = X_mi[col].astype("category").cat.codes
    X_mi = X_mi.replace([np.inf, -np.inf], np.nan)
    for col in X_mi.columns:
        if X_mi[col].isna().any():
            X_mi[col] = X_mi[col].fillna(X_mi[col].median())

    y_encoded = LabelEncoder().fit_transform(y.astype(str))
    discrete_mask = [col in categorical_cols for col in X_mi.columns]
    mi = mutual_info_classif(
        X_mi, y_encoded,
        discrete_features=discrete_mask,
        random_state=42
    )
    mi_df = pd.DataFrame({
        "feature": X_mi.columns,
        "mutual_information": mi,
        "feature_type": [
            "categorical" if c in categorical_cols else "numeric"
            for c in X_mi.columns
        ]
    }).sort_values("mutual_information", ascending=False)
    mi_df.to_csv(OUTPUT_DIR / "mutual_information.csv", index=False)

    # Numeric feature-target association: one-vs-rest eta-like variance ratio.
    assoc_rows = []
    for col in numeric_cols:
        overall_mean = selected_df[col].mean()
        total_ss = ((selected_df[col] - overall_mean) ** 2).sum()
        between_ss = 0.0
        for _, group in selected_df.groupby(TARGET_COLUMN, observed=False):
            if len(group):
                between_ss += len(group) * (group[col].mean() - overall_mean) ** 2
        eta_squared = between_ss / total_ss if total_ss else np.nan
        assoc_rows.append({"feature": col, "eta_squared": eta_squared})
    pd.DataFrame(assoc_rows).sort_values(
        "eta_squared", ascending=False
    ).to_csv(OUTPUT_DIR / "numeric_target_association.csv", index=False)

    # Target distribution chart
    plt.figure(figsize=(10, 5))
    order = selected_df[TARGET_COLUMN].value_counts().index
    sns.countplot(data=selected_df, x=TARGET_COLUMN, order=order)
    plt.title("Distribusi Tiga Kelas Target")
    plt.xlabel("Target")
    plt.ylabel("Jumlah sampel")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=180)
    plt.close()

    summary = f"""# Ringkasan Audit Dataset

## Ringkasan teknis
- Dataset awal: {len(full_df):,} baris dan {len(full_df.columns)} kolom.
- Subset tiga kelas: {len(selected_df):,} baris.
- Fitur numerik: {len(numeric_cols)}.
- Fitur kategorikal: {len(categorical_cols)}.
- Missing cells pada dataset awal: {int(full_df.isna().sum().sum())}.
- Duplikasi baris penuh: {int(full_df.duplicated().sum())}.

## Distribusi target subset
{selected_df[TARGET_COLUMN].value_counts().to_string()}

## Peringatan interpretasi
1. Mutual information dan eta-squared hanya mengukur asosiasi, bukan sebab-akibat.
2. Asosiasi fitur-target yang kuat dapat muncul karena hubungan nyata, aturan simulasi, atau kebocoran target; perlu dokumentasi sumber untuk membedakannya.
3. Tidak adanya missing values atau duplikasi tidak membuktikan validitas klinis.
4. Dataset tidak boleh disebut data rekam medis rumah sakit tanpa bukti dokumenter.
"""
    _save_text(OUTPUT_DIR / "audit_summary.md", summary)

    print("Audit dataset selesai. Lihat folder outputs/.")
