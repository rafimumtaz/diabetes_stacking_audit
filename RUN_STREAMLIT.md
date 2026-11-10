# Menjalankan UI Streamlit

## 1. Persiapan (Windows PowerShell)

Buka terminal di folder `diabetes_stacking_audit`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ui.txt
```

Jika aktivasi diblokir:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Jalankan UI

Artefak model demo sudah disertakan di folder `models/`. Jalankan:

```powershell
streamlit run app.py
```

Jika ingin melatih ulang model pada perangkat sendiri:

```powershell
python -m src.train_app_model
streamlit run app.py
```

Training ulang dapat memerlukan waktu. File model disimpan ke `models/`.

## 3. Screenshot untuk dokumentasi HKI

Ambil screenshot halaman berikut:

1. **Dashboard** — ringkasan jumlah data, distribusi kelas, ringkasan model.
2. **Simulasi Klasifikasi** — formulir input fitur numerik dan kategorikal.
3. **Hasil Klasifikasi** — isi formulir lalu tekan tombol Jalankan klasifikasi; screenshot hasil prediksi dan skor per kelas.
4. **Informasi Model** — diagram arsitektur Stacking Ensemble dan metrik holdout.
5. **Tentang Aplikasi** — deskripsi fungsi, teknologi, dan batasan aplikasi.

Simpan screenshot asli tanpa mengubah informasi yang tampil. Untuk laporan, beri keterangan bahwa aplikasi adalah prototipe penelitian dan bukan perangkat diagnosis.

## Catatan kompatibilitas

Artefak `joblib` disertakan bersama versi dependensi yang dipakai saat pembuatan. Gunakan `requirements-ui.txt` agar versi scikit-learn dan LightGBM selaras. Jangan memuat file joblib dari sumber yang tidak dipercaya.
