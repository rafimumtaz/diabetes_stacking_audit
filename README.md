# Audit Dataset Diabetes dan Eksperimen Ablation

Project ini melakukan audit eksploratif terhadap dataset Kaggle yang diberikan, membentuk subset tiga kelas, memeriksa asosiasi fitur dengan target, dan membandingkan model baseline serta stacking.

## Penting: batas interpretasi

- Dataset harus disebut sesuai asal-usul dan dokumentasi sumbernya. Jangan menyebutnya sebagai rekam medis rumah sakit jika tidak terbukti.
- Pola fitur-target yang sangat kuat atau skor mendekati/tepat 100% adalah sinyal untuk investigasi, bukan bukti model klinis yang unggul.
- Korelasi, mutual information, feature importance, dan ablation tidak dapat sendirian membuktikan target leakage atau membuktikan data sintetis.
- Hasil split acak/cross-validation hanya mengukur konsistensi pada distribusi dataset ini; tidak membuktikan generalisasi ke pasien nyata.
- Kolom yang dikeluarkan pada skenario `reduced` merupakan daftar investigasi awal yang dapat diubah pada `config.py`, bukan pernyataan bahwa fitur tersebut pasti menyebabkan leakage.
- Model dan laporan ini adalah eksperimen riset, bukan alat diagnosis.

## Struktur folder

```text
diabetes_stacking_audit/
├── data/
│   └── diabetes_dataset00.csv
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── audit.py
│   ├── modeling.py
│   └── run_all.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Persiapan Python virtual environment

Windows PowerShell:

```powershell
cd path\ke\diabetes_stacking_audit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi, jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Letakkan dataset

Salin file `diabetes_dataset00.csv` ke folder `data/`.

Nama file harus persis:

```text
data/diabetes_dataset00.csv
```

## 3. Jalankan seluruh audit dan eksperimen

Dari root project:

```bash
python -m src.run_all
```

Atau:

```bash
python src/run_all.py
```

## 4. Output

Hasil disimpan dalam `outputs/`, antara lain:

- `dataset_overview.txt`
- `class_distribution_all.csv`
- `selected_three_class.csv`
- `numeric_summary_by_target.csv`
- `numeric_feature_ranges_by_target.csv`
- `categorical_feature_distribution.csv`
- `numeric_target_association.csv`
- `categorical_target_association.csv`
- `mutual_information.csv`
- `correlation_matrix_numeric.png`
- `target_distribution.png`
- `model_metrics.csv`
- `classification_report_*.csv`
- `confusion_matrix_*.png`
- `feature_permutation_*.csv`
- `ablation_comparison.png`
- `audit_summary.md`

## 5. Skenario ablation

Secara default, program menguji:

1. `all_features`: semua fitur yang tersedia.
2. `numeric_only`: hanya fitur numerik.
3. `reduced_features`: semua fitur kecuali fitur yang ditandai untuk pemeriksaan lebih lanjut pada `src/config.py`.

`reduced_features` adalah eksperimen sensitivitas. Jangan menyebut fitur yang dikeluarkan sebagai penyebab leakage sebelum ada bukti dari dokumentasi pembentukan label.

## 6. Model

Setiap skenario membandingkan:

- Random Forest
- Gradient Boosting
- LightGBM
- Stacking: tiga base learner di atas dengan Logistic Regression sebagai meta learner

Stacking memakai `StackingClassifier` dengan `StratifiedKFold` dan prediksi out-of-fold internal. Preprocessing ditempatkan dalam pipeline agar encoder dipelajari hanya dari data training/fold.

## 7. Pengaturan

Edit `src/config.py` untuk mengganti:

- nama file atau kolom target,
- tiga label yang dipilih,
- daftar fitur yang dikecualikan pada ablation,
- ukuran test set,
- random state,
- jumlah fold,
- jumlah estimator dan parameter model.

## 8. Interpretasi hasil

Periksa khususnya:

- Apakah satu atau beberapa fitur saja dapat memprediksi target hampir sempurna?
- Apakah skor turun tajam ketika fitur tertentu dikeluarkan?
- Apakah hasil stabil pada cross-validation?
- Apakah confusion matrix menunjukkan kelas tertentu terlalu mudah dibedakan?
- Apakah dokumentasi sumber menjelaskan cara data dan label dibuat?

Jangan menghapus fitur hanya karena fitur tersebut berkorelasi tinggi. Keputusan fitur harus didasarkan pada definisi fitur, cara label ditetapkan, tujuan eksperimen, dan dokumentasi sumber.
