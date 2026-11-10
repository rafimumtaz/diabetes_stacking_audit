import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-GUI agar aman di Windows/worker
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer
)
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier

from .config import (
    OUTPUT_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    RF_PARAMS, GB_PARAMS, LGBM_PARAMS, POTENTIALLY_LABEL_PROXIMAL_FEATURES
)
from .data_utils import split_feature_types

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # Scaling tidak wajib untuk tree, tetapi tidak merusak alur preprocessing.
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return transformer

def make_models(X):
    preprocessor = build_preprocessor(X)

    rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(**RF_PARAMS))
    ])

    gb = Pipeline(steps=[
        ("preprocess", build_preprocessor(X)),
        ("model", GradientBoostingClassifier(**GB_PARAMS))
    ])

    lgbm = Pipeline(steps=[
        ("preprocess", build_preprocessor(X)),
        ("model", LGBMClassifier(**LGBM_PARAMS))
    ])

    # StackingClassifier melakukan training meta learner dari prediksi OOF.
    estimators = [
        ("rf", Pipeline([
            ("preprocess", build_preprocessor(X)),
            ("model", RandomForestClassifier(**RF_PARAMS))
        ])),
        ("gb", Pipeline([
            ("preprocess", build_preprocessor(X)),
            ("model", GradientBoostingClassifier(**GB_PARAMS))
        ])),
        ("lgbm", Pipeline([
            ("preprocess", build_preprocessor(X)),
            ("model", LGBMClassifier(**LGBM_PARAMS))
        ])),
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight=None
        ),
        stack_method="predict_proba",
        cv=StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        ),
        passthrough=False,
        n_jobs=1
    )

    return {
        "RandomForest": rf,
        "GradientBoosting": gb,
        "LightGBM": lgbm,
        "Stacking": stacking,
    }

def evaluate_one_model(name, model, X_train, X_test, y_train, y_test, scenario):
    print(f"  Training {name} | scenario={scenario}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "scenario": scenario,
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"classification_report_{scenario}_{name}.csv"
    )

    labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {name} ({scenario})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{scenario}_{name}.png", dpi=180)
    plt.close()

    # Permutation importance pada holdout test; dihitung untuk model yang sudah fit.
    # Ini adalah importance prediktif, bukan penjelasan kausal.
    try:
        perm = permutation_importance(
            model, X_test, y_test,
            scoring="f1_macro",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=1
        )
        imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        imp.to_csv(
            OUTPUT_DIR / f"feature_permutation_{scenario}_{name}.csv",
            index=False
        )
    except Exception as exc:
        print(f"  Peringatan: permutation importance gagal untuk {name}: {exc}")

    return metrics

def run_model_experiments(selected_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_full = selected_df.drop(columns=[TARGET_COLUMN])
    y = selected_df[TARGET_COLUMN].astype(str)

    reduced_drop = [
        col for col in POTENTIALLY_LABEL_PROXIMAL_FEATURES
        if col in X_full.columns
    ]

    scenarios = {
        "all_features": X_full.columns.tolist(),
        "numeric_only": X_full.select_dtypes(include=["number"]).columns.tolist(),
        "reduced_features": [c for c in X_full.columns if c not in reduced_drop],
    }

    all_metrics = []

    for scenario, columns in scenarios.items():
        if not columns:
            print(f"Skip {scenario}: tidak ada fitur.")
            continue

        X = X_full[columns].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        print(f"\n=== Skenario: {scenario} ===")
        print(f"Jumlah fitur: {len(columns)}")
        models = make_models(X)

        for model_name, model in models.items():
            result = evaluate_one_model(
                model_name, model,
                X_train, X_test, y_train, y_test,
                scenario
            )
            all_metrics.append(result)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    if not metrics_df.empty:
        pivot = metrics_df.pivot(
            index="model", columns="scenario", values="f1_macro"
        )
        plt.figure(figsize=(10, 6))
        pivot.plot(kind="bar", figsize=(10, 6))
        plt.ylabel("Macro-F1")
        plt.title("Perbandingan Macro-F1 antar Skenario Ablation")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ablation_comparison.png", dpi=180)
        plt.close()

    print("\nEksperimen model selesai.")
    print(metrics_df.to_string(index=False))
    return metrics_df
