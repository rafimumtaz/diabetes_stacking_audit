from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'stacking_diabetes.joblib'
SCHEMA_PATH = ROOT / 'models' / 'schema.json'
METRICS_PATH = ROOT / 'models' / 'metrics.json'
DATA_PATH = ROOT / 'data' / 'diabetes_dataset00.csv'

st.set_page_config(page_title='Diabetes Research Classifier', page_icon='', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1380px;}
.hero {padding: 1.35rem 1.55rem; border-radius: 18px; background: linear-gradient(115deg,#102b46,#175b72); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; font-size:2rem; margin:0 0 .35rem 0;}
.hero p {color:#dcebf3; margin:0; font-size:1rem;}
.small-muted {color:#718096; font-size:.88rem;}
.result-panel {padding:1.2rem; border:1px solid #d8e2ea; border-radius:14px; background:rgba(30,100,130,.045);}
.warning-panel {padding:1rem 1.15rem; border-left:5px solid #d69e2e; border-radius:8px; background:rgba(214,158,46,.10);}
div[data-testid="stMetric"] {border:1px solid rgba(128,145,160,.25); padding: .85rem; border-radius:12px;}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding='utf-8'))

if not MODEL_PATH.exists() or not SCHEMA_PATH.exists() or not METRICS_PATH.exists():
    st.error('Model belum tersedia. Jalankan training terlebih dahulu: `python -m src.train_app_model`')
    st.stop()

model = load_model()
schema = load_schema()
metrics = load_metrics()

with st.sidebar:
    st.markdown('##  Diabetes Research')
    st.caption('Stacking Ensemble · Research Prototype')
    page = st.radio('Navigasi', ['Dashboard', 'Simulasi Klasifikasi', 'Informasi Model'], label_visibility='collapsed')
    st.divider()
    st.caption('Universitas Trunojoyo Madura')
    st.caption('Dokumentasi prototipe penelitian')

st.markdown('''<div class="hero"><h1>Diabetes Mellitus Classification</h1><p>Sistem klasifikasi eksperimental menggunakan Stacking Ensemble berbasis Random Forest, Gradient Boosting, dan LightGBM.</p></div>''', unsafe_allow_html=True)
st.write('')

if page == 'Dashboard':
    st.subheader('Dashboard Penelitian')
    st.caption('Ringkasan dataset dan model yang digunakan pada prototipe.')
    try:
        df = pd.read_csv(DATA_PATH)
        selected = df[df['Target'].isin(schema['classes'])]
        n_total, n_selected = len(df), len(selected)
        class_counts = selected['Target'].value_counts().reindex(schema['classes']).fillna(0).astype(int)
    except Exception:
        n_total, n_selected = 70000, 16219
        class_counts = pd.Series([5376,5446,5397], index=schema['classes'])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Data awal', f'{n_total:,}'.replace(',','.'))
    c2.metric('Data terpilih', f'{n_selected:,}'.replace(',','.'))
    c3.metric('Jumlah kelas', len(schema['classes']))
    c4.metric('Fitur input', len(schema['features']))
    st.divider()
    left,right = st.columns([1.05,1])
    with left:
        st.markdown('#### Distribusi kelas')
        chart_df = class_counts.rename_axis('Kelas').reset_index(name='Jumlah data').set_index('Kelas')
        st.bar_chart(chart_df, use_container_width=True)
        st.dataframe(class_counts.rename('Jumlah data').to_frame(), use_container_width=True)
    with right:
        st.markdown('#### Ringkasan model')
        st.markdown('**Stacking Ensemble**')
        st.write('Base learner: Random Forest, Gradient Boosting, dan LightGBM.')
        st.write('Meta learner: Logistic Regression.')
        st.write('Validasi internal: holdout stratified 80:20 untuk ringkasan training.')
        st.metric('Macro-F1 holdout', f"{metrics['f1_macro']*100:.2f}%")
        st.metric('Accuracy holdout', f"{metrics['accuracy']*100:.2f}%")
        st.caption('Angka di atas hanya menggambarkan performa pada pembagian dataset Kaggle yang digunakan, bukan performa klinis.')
    st.divider()
    st.markdown('#### Alur sistem')
    st.markdown('**Input karakteristik → preprocessing → Random Forest + Gradient Boosting + LightGBM → Logistic Regression (meta learner) → hasil klasifikasi eksperimental.**')

elif page == 'Simulasi Klasifikasi':
    st.subheader('Simulasi Klasifikasi')
    st.write('Masukkan nilai fitur sesuai pilihan yang tersedia. Nilai awal numerik menggunakan median subset penelitian.')
    values = {}
    numeric_features = [f for f in schema['features'] if f['type']=='numeric']
    categorical_features = [f for f in schema['features'] if f['type']=='categorical']
    with st.form('prediction_form'):
        st.markdown('#### 1. Karakteristik numerik')
        cols = st.columns(3)
        for i, feat in enumerate(numeric_features):
            with cols[i % 3]:
                values[feat['name']] = st.number_input(
                    feat['name'], min_value=float(feat['min']), max_value=float(feat['max']),
                    value=float(feat['median']), step=1.0, help=f"Rentang pada subset: {feat['min']:g}–{feat['max']:g}"
                )
        st.divider()
        st.markdown('#### 2. Karakteristik kategorikal')
        cols = st.columns(3)
        for i, feat in enumerate(categorical_features):
            with cols[i % 3]:
                options = feat['options']
                default_index = options.index(feat['default']) if feat['default'] in options else 0
                values[feat['name']] = st.selectbox(feat['name'], options, index=default_index)
        submitted = st.form_submit_button('🔍 Jalankan klasifikasi', type='primary', use_container_width=True)
    if submitted:
        row = pd.DataFrame([{f['name']: values[f['name']] for f in schema['features']}])
        try:
            prediction = model.predict(row)[0]
            probabilities = model.predict_proba(row)[0]
            class_order = list(model.classes_)
            score_df = pd.DataFrame({'Kelas':class_order, 'Skor model (%)':probabilities*100}).sort_values('Skor model (%)',ascending=False)
            st.divider()
            st.markdown('### Hasil klasifikasi eksperimental')
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.success(f'Kelas yang diprediksi model: **{prediction}**')
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption('Skor probabilitas merupakan keluaran model dan belum tentu terkalibrasi sebagai probabilitas risiko klinis.')
            a,b = st.columns([1,1.2])
            with a:
                st.markdown('#### Skor per kelas')
                st.dataframe(score_df.style.format({'Skor model (%)':'{:.2f}%'}), use_container_width=True, hide_index=True)
            with b:
                st.markdown('#### Visualisasi skor model')
                st.bar_chart(score_df.set_index('Kelas'), use_container_width=True)
            st.info('Hasil ini hanya simulasi model penelitian pada dataset Kaggle dan tidak boleh digunakan untuk menentukan diagnosis, pengobatan, atau keputusan medis.')
        except Exception as exc:
            st.error(f'Prediksi gagal: {exc}')

elif page == 'Informasi Model':
    st.subheader('Informasi Model')
    st.markdown('#### Arsitektur Stacking Ensemble')
    st.code('''Input data\n   │\n   ▼\nPreprocessing\n   │\n   ├── Random Forest ─────┐\n   ├── Gradient Boosting ─┼──► Logistic Regression\n   └── LightGBM ──────────┘       (Meta Learner)\n                                  │\n                                  ▼\n                         Prediksi tiga kelas''', language='text')
    st.markdown('#### Komponen')
    st.dataframe(pd.DataFrame([
        {'Komponen':'Base learner 1','Algoritma':'Random Forest'},
        {'Komponen':'Base learner 2','Algoritma':'Gradient Boosting'},
        {'Komponen':'Base learner 3','Algoritma':'LightGBM'},
        {'Komponen':'Meta learner','Algoritma':'Logistic Regression'},
        {'Komponen':'Validasi internal','Algoritma':'Stratified 3-Fold pada proses stacking'},
    ]), use_container_width=True, hide_index=True)
    st.markdown('#### Metrik holdout')
    m1,m2,m3,m4=st.columns(4)
    m1.metric('Accuracy',f"{metrics['accuracy']*100:.2f}%")
    m2.metric('Precision macro',f"{metrics['precision_macro']*100:.2f}%")
    m3.metric('Recall macro',f"{metrics['recall_macro']*100:.2f}%")
    m4.metric('F1 macro',f"{metrics['f1_macro']*100:.2f}%")
    st.caption(f"Data training: {metrics['train_rows']:,} baris · Data pengujian: {metrics['test_rows']:,} baris · Random state: 42")   


