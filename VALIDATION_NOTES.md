# Catatan Validasi Lanjutan

## Temuan diagnostik aturan sederhana

Subset tiga kelas dapat diperiksa dengan aturan eksploratif berikut:

- Jika `Age < 30`, aturan memberi label `Type 1 Diabetes`.
- Jika `Age >= 30` dan `Blood Glucose Levels < 125`, aturan memberi label `Prediabetic`.
- Selain itu, aturan memberi label `Type 2 Diabetes`.

Aturan ini perlu diperlakukan sebagai **diagnostik pola dataset**, bukan algoritma medis. Karena dirumuskan setelah inspeksi subset, skor pada seluruh subset bukan hasil evaluasi independen dan tidak boleh dipakai untuk mengklaim generalisasi.

## Model untuk UI

`models/stacking_diabetes.joblib` adalah Stacking Ensemble yang dilatih pada training partition stratified 80% (12.975 baris) dan diuji pada holdout 20% (3.244 baris). Base learner: Random Forest (80 trees), Gradient Boosting (60 estimators), LightGBM (80 estimators). Meta learner: Logistic Regression. Internal OOF stacking menggunakan Stratified 3-Fold.

Skor holdout tercatat 100% pada dataset ini. Ini justru harus ditampilkan bersama peringatan pola target yang sangat kuat. Model ini disimpan untuk demonstrasi antarmuka, bukan validasi klinis.

## Validasi lanjutan yang belum boleh diklaim selesai

1. Verifikasi dokumentasi sumber Kaggle dan definisi tiap variabel.
2. Jalankan `python -m src.validation` untuk diagnostic 5-fold CV pada tiga base model; proses dapat memerlukan waktu.
3. Evaluasi prediksi pada sumber data eksternal yang independen bila tersedia dan telah memperoleh izin/etik yang sesuai.
4. Tinjau kemungkinan label construction / target leakage berdasarkan dokumentasi asal data, bukan hanya korelasi atau feature importance.
