# ============================================================
# APLIKASI KLASIFIKASI INTRUSI JARINGAN
# Algoritma XGBoost | Dataset CICIDS2017
# Jalankan: streamlit run app_streamlit.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# ── Konfigurasi halaman ─────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Intrusi Jaringan",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Header utama */
    .header-box {
        background: linear-gradient(135deg, #1a237e, #283593);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header-box h1 {
        color: white;
        font-size: 2rem;
        margin: 0 0 0.3rem 0;
    }
    .header-box p {
        color: #90CAF9;
        margin: 0;
        font-size: 1rem;
    }

    /* Card info */
    .info-card {
        background: #F8F9FF;
        border: 1px solid #E3E8FF;
        border-left: 4px solid #3F51B5;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* Hasil Normal */
    .hasil-normal {
        background: #E8F5E9;
        border: 2px solid #43A047;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .hasil-normal h2 { color: #1B5E20; margin: 0 0 0.5rem 0; }
    .hasil-normal p  { color: #2E7D32; margin: 0; }

    /* Hasil Attack */
    .hasil-attack {
        background: #FFEBEE;
        border: 2px solid #E53935;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .hasil-attack h2 { color: #B71C1C; margin: 0 0 0.5rem 0; }
    .hasil-attack p  { color: #C62828; margin: 0; }

    /* Metric card */
    [data-testid="metric-container"] {
        background: #F0F4FF;
        border: 1px solid #C5CAE9;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1a237e;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #90CAF9 !important;
        border-bottom: 1px solid #3949AB;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛡️ Klasifikasi Intrusi Jaringan</h1>
    <p>Menggunakan Algoritma XGBoost &nbsp;|&nbsp; Dataset CICIDS2017 &nbsp;|&nbsp; Skripsi 2025</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🤖 Informasi Model")
    st.markdown("""
    **Algoritma:** XGBoost

    **Parameter:**
    - n_estimators : 100
    - max_depth : 5
    - learning_rate : 0.1

    **Dataset:** CICIDS2017

    **Kelas:**
    - ✅ Normal Traffic
    - 🚨 Port Scanning
    """)

    st.markdown("---")
    st.markdown("### 📊 Hasil Training")
    st.markdown("""
    - Accuracy  : **100%**
    - Precision : **100%**
    - Recall    : **100%**
    - F1-Score  : **100%**
    """)

    st.markdown("---")
    st.markdown("### 📌 Cara Penggunaan")
    st.markdown("""
    1. Pilih menu di atas
    2. Upload CSV **atau** input manual
    3. Klik tombol Prediksi
    4. Lihat hasilnya
    """)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgboost_cicids2017.json")
    return model

try:
    model = load_model()
    fitur_model = model.get_booster().feature_names
except Exception as e:
    st.error(f"❌ **Model gagal dimuat!**")
    st.code(str(e))
    st.warning("Pastikan file **xgboost_cicids2017.json** ada di folder yang sama dengan file app ini.")
    st.stop()

# ── Status model ────────────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Status Model", "✅ Siap")
col_s2.metric("Jumlah Fitur", f"{len(fitur_model)}")
col_s3.metric("Algoritma", "XGBoost")

st.markdown("---")

# ============================================================
# MENU NAVIGASI
# ============================================================
menu = st.selectbox(
    "📂 Pilih Menu:",
    options=[
        "📁 Prediksi via Upload CSV",
        "✏️ Prediksi via Input Manual"
    ],
    index=0
)

st.markdown("")

# ============================================================
# MENU 1: UPLOAD CSV
# ============================================================
if menu == "📁 Prediksi via Upload CSV":

    st.markdown("### 📁 Upload File CSV")

    st.markdown("""
    <div class="info-card">
    <b>📋 Ketentuan file CSV:</b><br>
    • File berisi data trafik jaringan<br>
    • Tidak perlu ada kolom label (akan diabaikan jika ada)<br>
    • Nama kolom harus sesuai fitur dataset CICIDS2017
    </div>
    """, unsafe_allow_html=True)

    # ── Komponen upload ─────────────────────────────────────
    file_upload = st.file_uploader(
        label="Klik atau seret file CSV ke sini",
        type=["csv"],
        help="Upload file CSV berisi data trafik jaringan"
    )

    if file_upload is not None:

        # Baca file
        df = pd.read_csv(file_upload)
        df.columns = df.columns.str.strip()

        # Hapus SEMUA kolom yang mengandung kata label atau asli
        kolom_hapus = [k for k in df.columns if any(
            kata in k.lower() for kata in ["label", "asli", "attack type", "prediksi"]
)]
if kolom_hapus:
    df.drop(columns=kolom_hapus, inplace=True)
    st.info(f"Kolom dihapus otomatis: {kolom_hapus}")

        # Info file
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Total Baris", f"{len(df):,}")
        col_f2.metric("Total Kolom", f"{df.shape[1]}")
        col_f3.metric("Status File", "✅ Terbaca")

        st.markdown("**Preview data (5 baris pertama):**")
        st.dataframe(df.head(5), use_container_width=True)

        st.markdown("")

        # Tombol prediksi
        if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):

            with st.spinner("⏳ Sedang memproses prediksi..."):

                # Siapkan data sesuai fitur model
                df_pred = pd.DataFrame(0.0, index=df.index, columns=fitur_model)
                for fitur in fitur_model:
                    if fitur in df.columns:
                        df_pred[fitur] = pd.to_numeric(df[fitur], errors="coerce").fillna(0)
                df_pred = df_pred.replace([np.inf, -np.inf], 0)

                # Prediksi
                hasil = model.predict(df_pred)
                proba = model.predict_proba(df_pred)

                # Gabungkan hasil
                df_hasil = df.copy()
                df_hasil["Prediksi"]       = ["✅ Normal" if p == 1 else "🚨 Attack" for p in hasil]
                df_hasil["Confidence (%)"] = (np.max(proba, axis=1) * 100).round(2)

            # ── Ringkasan hasil ──────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")

            jml_normal = sum(1 for p in hasil if p == 1)
            jml_attack = sum(1 for p in hasil if p == 0)
            pct_normal = jml_normal / len(hasil) * 100
            pct_attack = jml_attack / len(hasil) * 100

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Total Data",  f"{len(hasil):,}")
            col_r2.metric("✅ Normal",   f"{jml_normal:,}", f"{pct_normal:.1f}%")
            col_r3.metric("🚨 Attack",   f"{jml_attack:,}", f"{pct_attack:.1f}%",
                          delta_color="inverse")

            # Progress bar proporsi
            st.markdown(f"**Proporsi Normal vs Attack:**")
            st.progress(pct_normal / 100)
            st.caption(f"Normal: {pct_normal:.1f}%  |  Attack: {pct_attack:.1f}%")

            # Tabel hasil
            st.markdown("**Detail hasil prediksi:**")
            st.dataframe(
                df_hasil[["Prediksi", "Confidence (%)"]].head(100),
                use_container_width=True
            )
            if len(df_hasil) > 100:
                st.caption(f"Menampilkan 100 dari {len(df_hasil):,} data.")

            # Download
            csv_out = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil Prediksi (CSV)",
                data=csv_out,
                file_name="hasil_prediksi.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        # Tampilan saat belum upload
        st.info("⬆️ Silakan upload file CSV di atas untuk memulai prediksi.")

# ============================================================
# MENU 2: INPUT MANUAL
# ============================================================
else:

    st.markdown("### ✏️ Input Nilai Fitur Jaringan")

    st.markdown("""
    <div class="info-card">
    <b>📋 Petunjuk:</b> Masukkan nilai fitur lalu lintas jaringan di bawah ini, 
    lalu klik tombol <b>Prediksi Sekarang</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Form input ───────────────────────────────────────────
    with st.form("form_manual"):

        st.markdown("#### Fitur Utama")
        col1, col2, col3 = st.columns(3)

        with col1:
            dst_port      = st.number_input("Destination Port",       min_value=0,   max_value=65535, value=80)
            flow_duration = st.number_input("Flow Duration",          min_value=0,   value=1000000)
            total_fwd     = st.number_input("Total Fwd Packets",      min_value=0,   value=5)
            total_bwd     = st.number_input("Total Backward Packets", min_value=0,   value=3)

        with col2:
            flow_bytes    = st.number_input("Flow Bytes/s",           min_value=0.0, value=5000.0,   format="%.2f")
            flow_packets  = st.number_input("Flow Packets/s",         min_value=0.0, value=10.0,     format="%.2f")
            fwd_iat_min   = st.number_input("Fwd IAT Min",            min_value=0.0, value=100000.0, format="%.2f")
            psh_flag      = st.number_input("PSH Flag Count",         min_value=0,   value=0)

        with col3:
            pkt_len_mean     = st.number_input("Packet Length Mean",     min_value=0.0, value=60.0,  format="%.2f")
            pkt_len_std      = st.number_input("Packet Length Std",      min_value=0.0, value=20.0,  format="%.2f")
            fwd_pkt_len_mean = st.number_input("Fwd Packet Length Mean", min_value=0.0, value=60.0,  format="%.2f")
            bwd_pkt_len_std  = st.number_input("Bwd Packet Length Std",  min_value=0.0, value=15.0,  format="%.2f")

        st.markdown("#### Fitur Tambahan")
        col4, col5 = st.columns(2)
        with col4:
            fwd_header_len = st.number_input("Fwd Header Length",     min_value=0,   value=20)
            init_win_fwd   = st.number_input("Init Win Bytes Forward", min_value=0,   value=65535)
        with col5:
            init_win_bwd   = st.number_input("Init Win Bytes Backward", min_value=0, value=65535)
            act_data_pkt   = st.number_input("Act Data Pkt Fwd",       min_value=0,  value=1)

        # Tombol submit di dalam form
        submitted = st.form_submit_button(
            "🔍 Prediksi Sekarang",
            type="primary",
            use_container_width=True
        )

    # ── Proses prediksi ──────────────────────────────────────
    if submitted:

        input_df = pd.DataFrame(0.0, index=[0], columns=fitur_model)

        nilai_input = {
            "Destination Port"         : float(dst_port),
            "Flow Duration"            : float(flow_duration),
            "Total Fwd Packets"        : float(total_fwd),
            "Total Backward Packets"   : float(total_bwd),
            "Flow Bytes/s"             : float(flow_bytes),
            "Flow Packets/s"           : float(flow_packets),
            "Fwd IAT Min"              : float(fwd_iat_min),
            "PSH Flag Count"           : float(psh_flag),
            "Packet Length Mean"       : float(pkt_len_mean),
            "Packet Length Std"        : float(pkt_len_std),
            "Fwd Packet Length Mean"   : float(fwd_pkt_len_mean),
            "Bwd Packet Length Std"    : float(bwd_pkt_len_std),
            "Fwd Header Length"        : float(fwd_header_len),
            "Init_Win_bytes_forward"   : float(init_win_fwd),
            "Init_Win_bytes_backward"  : float(init_win_bwd),
            "act_data_pkt_fwd"         : float(act_data_pkt),
        }

        for nama, nilai in nilai_input.items():
            if nama in input_df.columns:
                input_df[nama] = nilai

        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label      = "Normal" if pred == 1 else "Attack"
        confidence = max(proba) * 100
        p_normal   = proba[1] * 100
        p_attack   = proba[0] * 100

        # ── Tampilkan hasil ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        if label == "Normal":
            st.markdown(f"""
            <div class="hasil-normal">
                <h2>✅ NORMAL TRAFFIC</h2>
                <p>Lalu lintas jaringan terdeteksi <b>aman</b> dan tidak mengandung serangan</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hasil-attack">
                <h2>🚨 TERDETEKSI SERANGAN!</h2>
                <p>Lalu lintas jaringan teridentifikasi sebagai <b>Port Scanning Attack</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Confidence bar
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown(f"**Tingkat Keyakinan Model: {confidence:.2f}%**")
            st.progress(confidence / 100)
        with col_c2:
            st.metric("Prediksi", label)

        # Probabilitas detail
        st.markdown("**Probabilitas per kelas:**")
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("✅ Normal Traffic", f"{p_normal:.2f}%")
        col_p2.metric("🚨 Port Scanning",  f"{p_attack:.2f}%")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem'>"
    "Klasifikasi Intrusi Jaringan Menggunakan Algoritma XGBoost &nbsp;|&nbsp; "
    "Dataset CICIDS2017 &nbsp;|&nbsp; Skripsi 2025"
    "</p>",
    unsafe_allow_html=True
)
