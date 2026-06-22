"""
BAB IV - 4.8 Implementasi Aplikasi (Deployment)
Aplikasi Deteksi Intrusi Jaringan menggunakan XGBoost + Streamlit
=================================================================
Cara menjalankan:
    pip install streamlit xgboost pandas numpy scikit-learn
    streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# ─── Konfigurasi Halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Intrusi Jaringan Menggunakan Algoritma XGBoost",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Custom ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2rem; margin: 0; }
    .main-header p  { font-size: 1rem; margin-top: 0.5rem; opacity: 0.85; }

    .metric-card {
        background: #F0F4FF;
        border-left: 5px solid #1565C0;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .result-normal {
        background: #E8F5E9;
        border: 2px solid #43A047;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-attack {
        background: #FFEBEE;
        border: 2px solid #E53935;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .stAlert { border-radius: 8px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Klasifikasi Intrusi Jaringan Menggunakan Algoritma XGBoost</h1>
    <p>Deteksi Trafik Normal dan Serangan Port Scanning pada Dataset CICIDS2017</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────
MODEL_PATH = "output_bab4/xgboost_cicids2017.json"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

model = load_model(MODEL_PATH)

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/XGBoost_logo.png/320px-XGBoost_logo.png",
             width=160)
    st.markdown("### 📌 Informasi Model")
    st.markdown("""
    | Parameter | Nilai |
    |-----------|-------|
    | Algoritma | XGBoost |
    | n_estimators | 100 |
    | max_depth | 5 |
    | learning_rate | 0.1 |
    | Dataset | CICIDS2017 |
    """)
    st.markdown("---")
    st.markdown("### 🎯 Kelas Target")
    st.success("✅ **Normal** — Lalu lintas jaringan aman")
    st.error("🚨 **Attack** — Terdeteksi serangan jaringan")

# ─── Konten Utama ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📊 Prediksi CSV", "✍️ Input Manual"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — BERANDA
# ══════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>📁 Dataset</h4><p>CICIDS2017</p></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>⚙️ Algoritma</h4><p>XGBoost</p></div>',
                    unsafe_allow_html=True)
    with col3:
        status = "✅ Dimuat" if model else "❌ Belum ada"
        st.markdown(f'<div class="metric-card"><h4>🤖 Status Model</h4><p>{status}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Tentang Aplikasi")
    st.info("""
    Aplikasi ini merupakan implementasi model **XGBoost** untuk mendeteksi
    intrusi jaringan berdasarkan fitur trafik jaringan dari dataset **CICIDS2017**.

    **Cara Penggunaan:**
    1. Pilih tab **Prediksi CSV** untuk mengunggah file data jaringan
    2. Pilih tab **Input Manual** untuk memasukkan nilai fitur secara langsung
    3. Klik tombol **Prediksi** untuk melihat hasil klasifikasi
    """)

    st.markdown("### 🔍 Jenis Serangan yang Dapat Dideteksi")
    cols = st.columns(3)
    serangan = ["DDoS", "Brute Force", "Port Scan",
                "Botnet", "Web Attack", "Infiltration"]
    for i, s in enumerate(serangan):
        cols[i % 3].markdown(f"- 🔴 {s}")

# ══════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI CSV
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Upload File CSV")
    st.caption("Unggah file CSV yang berisi fitur trafik jaringan (tanpa kolom label).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="csv_upload")

    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
            df_input.columns = df_input.columns.str.strip()

            # Hapus kolom label jika ada
            for col in ["Label", "label", " Label"]:
                if col in df_input.columns:
                    df_input.drop(columns=[col], inplace=True)

            st.success(f"✅ File berhasil dimuat: **{df_input.shape[0]:,} baris**, **{df_input.shape[1]} fitur**")
            st.dataframe(df_input.head(5), use_container_width=True)

            if model is None:
                st.error("❌ Model belum dimuat. Jalankan `bab4_cicids2017_xgboost.py` terlebih dahulu.")
            else:
                if st.button("🔍 Prediksi Semua Data", type="primary", key="btn_csv"):
                    with st.spinner("Memproses prediksi..."):
                        # Align fitur
                        model_features = model.get_booster().feature_names
                        df_pred = df_input.copy()
                        for f in model_features:
                            if f not in df_pred.columns:
                                df_pred[f] = 0
                        df_pred = df_pred[model_features]
                        df_pred = df_pred.apply(pd.to_numeric, errors="coerce").fillna(0)
                        df_pred = df_pred.replace([np.inf, -np.inf], 0)

                        preds = model.predict(df_pred)
                        proba = model.predict_proba(df_pred)

                        df_result = df_input.copy()
                        df_result["Prediksi"] = ["Normal" if p == 0 else "Attack" for p in preds]
                        df_result["Confidence (%)"] = (np.max(proba, axis=1) * 100).round(2)

                    # Statistik
                    n_normal = (df_result["Prediksi"] == "Normal").sum()
                    n_attack = (df_result["Prediksi"] == "Attack").sum()

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Data", f"{len(df_result):,}")
                    col_b.metric("✅ Normal", f"{n_normal:,}", f"{n_normal/len(df_result)*100:.1f}%")
                    col_c.metric("🚨 Attack", f"{n_attack:,}", f"{n_attack/len(df_result)*100:.1f}%",
                                 delta_color="inverse")

                    st.markdown("#### Hasil Prediksi")
                    st.dataframe(
                        df_result[["Prediksi", "Confidence (%)"]].head(100),
                        use_container_width=True
                    )

                    # Download
                    csv_out = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Hasil Prediksi", csv_out,
                                       "hasil_prediksi.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi error: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 3 — INPUT MANUAL
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ✍️ Input Fitur Jaringan Secara Manual")
    st.caption("Masukkan nilai fitur jaringan untuk mendapatkan prediksi secara langsung.")

    # Fitur kunci CICIDS2017 yang umum digunakan
    default_features = {
        "Destination Port"       : (80,  0,  65535),
        "Flow Duration"          : (1000000, 0, 1e10),
        "Total Fwd Packets"      : (5,   0,  100000),
        "Total Backward Packets" : (3,   0,  100000),
        "Total Length of Fwd Packets": (500, 0, 1e8),
        "Total Length of Bwd Packets": (300, 0, 1e8),
        "Fwd Packet Length Max"  : (100, 0,  65535),
        "Fwd Packet Length Min"  : (20,  0,  65535),
        "Fwd Packet Length Mean" : (60,  0.0, 65535.0),
        "Bwd Packet Length Max"  : (80,  0,  65535),
        "Flow Bytes/s"           : (5000.0, 0.0, 1e10),
        "Flow Packets/s"         : (10.0, 0.0, 1e8),
        "Flow IAT Mean"          : (100000.0, 0.0, 1e10),
        "Flow IAT Std"           : (50000.0, 0.0, 1e10),
        "Fwd IAT Total"          : (500000, 0, 1e10),
        "Fwd IAT Mean"           : (125000.0, 0.0, 1e10),
    }

    col_left, col_right = st.columns(2)
    feature_values = {}
    items = list(default_features.items())

    for i, (feat, (default, min_v, max_v)) in enumerate(items):
        col = col_left if i % 2 == 0 else col_right
        feature_values[feat] = col.number_input(
            feat, value=float(default), min_value=float(min_v),
            max_value=float(max_v), key=f"fi_{i}"
        )

    st.markdown("---")
    if st.button("🔍 Prediksi", type="primary", key="btn_manual"):
        if model is None:
            st.error("❌ Model belum dimuat. Jalankan `bab4_cicids2017_xgboost.py` terlebih dahulu.")
        else:
            with st.spinner("Memproses..."):
                model_features = model.get_booster().feature_names
                input_df = pd.DataFrame([{f: 0 for f in model_features}])
                for feat, val in feature_values.items():
                    if feat in input_df.columns:
                        input_df[feat] = val

                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba) * 100
                label = "Normal" if pred == 0 else "Attack"

            st.markdown("---")
            st.markdown("### 🎯 Hasil Prediksi")

            if label == "Normal":
                st.markdown(f"""
                <div class="result-normal">
                    <h2>✅ NORMAL</h2>
                    <p>Lalu lintas jaringan terdeteksi <strong>aman</strong></p>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-attack">
                    <h2>🚨 ATTACK</h2>
                    <p>Terdeteksi <strong>serangan jaringan</strong>!</p>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Probabilitas per Kelas")
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Normal", f"{proba[0]*100:.2f}%")
            if len(proba) > 1:
                col_p2.metric("Attack", f"{proba[1]*100:.2f}%")

# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Klasifikasi Intrusi Jaringan Menggunakan Algoritma XGBoost | Skripsi 2025"
    "</div>",
    unsafe_allow_html=True
)
