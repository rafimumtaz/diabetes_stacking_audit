"""Train model khusus untuk demo Streamlit; hasil bukan alat diagnosis klinis."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'diabetes_dataset00.csv'
MODEL_DIR = ROOT / 'models'
OUTPUT_DIR = ROOT / 'outputs'
TARGET = 'Target'
CLASSES = ['Prediabetic', 'Type 1 Diabetes', 'Type 2 Diabetes']
SEED = 42


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(f'Dataset tidak ditemukan: {DATA}')

    raw = pd.read_csv(DATA)
    df = raw[raw[TARGET].isin(CLASSES)].copy()
    X, y = df.drop(columns=TARGET), df[TARGET].astype(str)
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = X.select_dtypes(exclude='number').columns.tolist()

    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical)
    ])

    def pipeline(estimator):
        return Pipeline([('preprocess', preprocessor), ('model', estimator)])

    estimators = [
        ('rf', pipeline(RandomForestClassifier(n_estimators=80, random_state=SEED, n_jobs=1))),
        ('gb', pipeline(GradientBoostingClassifier(n_estimators=60, learning_rate=0.08, max_depth=3, random_state=SEED))),
        ('lgbm', pipeline(LGBMClassifier(n_estimators=80, learning_rate=0.05, num_leaves=31, verbosity=-1, n_jobs=1, random_state=SEED)))
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method='predict_proba',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print('Training Stacking Ensemble pada subset 3 kelas...')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision_macro': float(precision_score(y_test, pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, pred, average='macro', zero_division=0)),
        'test_rows': int(len(y_test)),
        'train_rows': int(len(y_train)),
        'classes': CLASSES,
        'note': 'Holdout acak dari dataset Kaggle; bukan validasi klinis.'
    }
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(OUTPUT_DIR / 'streamlit_stacking_classification_report.csv')
    pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(OUTPUT_DIR / 'streamlit_stacking_confusion_matrix.csv')
    (MODEL_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    # Simpan model yang dilatih pada training partition; metrik berasal dari holdout terpisah.
    joblib.dump(model, MODEL_DIR / 'stacking_diabetes.joblib', compress=3)

    schema = {'target': TARGET, 'classes': CLASSES, 'features': []}
    for col in X.columns:
        if col in numeric:
            schema['features'].append({
                'name': col, 'type': 'numeric',
                'min': float(X[col].min()), 'max': float(X[col].max()),
                'median': float(X[col].median()), 'mean': float(X[col].mean())
            })
        else:
            vals = sorted(X[col].dropna().astype(str).unique().tolist())
            mode = str(X[col].mode(dropna=True).iloc[0])
            schema['features'].append({'name': col, 'type': 'categorical', 'options': vals, 'default': mode})
    (MODEL_DIR / 'schema.json').write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nTraining selesai.')
    print(json.dumps(metrics, indent=2))
    print(f'Model: {MODEL_DIR / "stacking_diabetes.joblib"}')

if __name__ == '__main__':
    main()
