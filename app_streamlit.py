# ============================================================
# APLIKASI KLASIFIKASI INTRUSI JARINGAN
# Algoritma XGBoost | Dataset CICIDS2017
# Jalankan: streamlit run app_streamlit.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# ── 52 fitur PERSIS sesuai model ────────────────────────────
FITUR_MODEL = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Average Packet Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min'
]

# ── Konfigurasi halaman ─────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Intrusi Jaringan",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1a237e, #283593);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header-box h1 { color: white; font-size: 2rem; margin: 0 0 0.3rem 0; }
    .header-box p  { color: #90CAF9; margin: 0; font-size: 1rem; }

    .info-card {
        background: #F8F9FF;
        border: 1px solid #E3E8FF;
        border-left: 4px solid #3F51B5;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .hasil-normal {
        background: #E8F5E9;
        border: 2px solid #43A047;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .hasil-normal h2 { color: #1B5E20; margin: 0 0 0.5rem 0; }
    .hasil-normal p  { color: #2E7D32; margin: 0; }

    .hasil-attack {
        background: #FFEBEE;
        border: 2px solid #E53935;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .hasil-attack h2 { color: #B71C1C; margin: 0 0 0.5rem 0; }
    .hasil-attack p  { color: #C62828; margin: 0; }

    [data-testid="metric-container"] {
        background: #F0F4FF;
        border: 1px solid #C5CAE9;
        border-radius: 10px;
        padding: 0.8rem 1rem;
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
    1. Pilih menu di bawah
    2. Upload CSV **atau** input manual
    3. Klik tombol Prediksi
    4. Lihat hasilnya
    """)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model("xgboost_cicids2017.json")
    return m

try:
    model = load_model()
except Exception as e:
    st.error("❌ **Model gagal dimuat!**")
    st.code(str(e))
    st.warning("Pastikan file **xgboost_cicids2017.json** ada di folder yang sama.")
    st.stop()

# ── Status model ────────────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Status Model", "✅ Siap")
col_s2.metric("Jumlah Fitur", f"{len(FITUR_MODEL)}")
col_s3.metric("Algoritma", "XGBoost")
st.markdown("---")

# ============================================================
# FUNGSI PREDIKSI — dipakai oleh kedua menu
# ============================================================
def siapkan_dan_prediksi(df_input):
    """
    Menerima dataframe apapun, menyusun ulang kolomnya
    sesuai FITUR_MODEL, lalu mengembalikan hasil prediksi.
    """
    # Buat dataframe kosong dengan 52 kolom persis
    df_pred = pd.DataFrame(0.0, index=df_input.index, columns=FITUR_MODEL)

    # Isi dari df_input — hanya kolom yang namanya cocok
    for fitur in FITUR_MODEL:
        if fitur in df_input.columns:
            df_pred[fitur] = pd.to_numeric(
                df_input[fitur], errors="coerce"
            ).fillna(0).values

    # Bersihkan inf
    df_pred.replace([np.inf, -np.inf], 0, inplace=True)

    hasil = model.predict(df_pred)
    proba = model.predict_proba(df_pred)
    return hasil, proba

# ============================================================
# MENU
# ============================================================
menu = st.selectbox(
    "📂 Pilih Menu:",
    options=["📁 Prediksi via Upload CSV", "✏️ Prediksi via Input Manual"],
    index=0
)
st.markdown("")

# ============================================================
# MENU 1 — UPLOAD CSV
# ============================================================
if menu == "📁 Prediksi via Upload CSV":

    st.markdown("### 📁 Upload File CSV")
    st.markdown("""
    <div class="info-card">
    <b>📋 Ketentuan file CSV:</b><br>
    • Gunakan file <b>dataset_tes_untuk_app.csv</b> yang didownload dari Colab<br>
    • Kolom label akan dihapus otomatis jika ada<br>
    • 52 kolom fitur harus sesuai dengan dataset CICIDS2017
    </div>
    """, unsafe_allow_html=True)

    file_upload = st.file_uploader(
        label="Klik atau seret file CSV ke sini",
        type=["csv"]
    )

    if file_upload is not None:

        # Baca CSV
        df = pd.read_csv(file_upload)

        # Strip spasi nama kolom
        df.columns = df.columns.str.strip()

        # Hapus kolom non-fitur (label, asli, prediksi, dll)
        kolom_buang = [
            k for k in df.columns
            if k not in FITUR_MODEL
        ]
        if kolom_buang:
            df.drop(columns=kolom_buang, inplace=True)
            st.info(f"Kolom dihapus (bukan fitur model): {kolom_buang}")

        # Cek kolom yang ada vs yang dibutuhkan
        kolom_ada    = [f for f in FITUR_MODEL if f in df.columns]
        kolom_kurang = [f for f in FITUR_MODEL if f not in df.columns]

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Total Baris",   f"{len(df):,}")
        col_f2.metric("Kolom Cocok",   f"{len(kolom_ada)} / {len(FITUR_MODEL)}")
        col_f3.metric("Status File",   "✅ Terbaca")

        if kolom_kurang:
            st.warning(f"⚠️ {len(kolom_kurang)} kolom tidak ditemukan, akan diisi 0: {kolom_kurang[:5]}{'...' if len(kolom_kurang)>5 else ''}")

        st.markdown("**Preview data (5 baris pertama):**")
        st.dataframe(df.head(5), use_container_width=True)
        st.markdown("")

        if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):

            with st.spinner("⏳ Memproses prediksi..."):
                hasil, proba = siapkan_dan_prediksi(df)

            # Hasil
            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")

            jml_normal = int((hasil == 1).sum())
            jml_attack = int((hasil == 0).sum())
            pct_normal = jml_normal / len(hasil) * 100
            pct_attack = jml_attack / len(hasil) * 100

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Total Data", f"{len(hasil):,}")
            col_r2.metric("✅ Normal",  f"{jml_normal:,}", f"{pct_normal:.1f}%")
            col_r3.metric("🚨 Attack",  f"{jml_attack:,}", f"{pct_attack:.1f}%",
                          delta_color="inverse")

            st.markdown("**Proporsi Normal vs Attack:**")
            st.progress(pct_normal / 100)
            st.caption(f"Normal: {pct_normal:.1f}%  |  Attack: {pct_attack:.1f}%")

            # Tabel detail
            df_hasil = pd.DataFrame({
                "No"             : range(1, len(hasil) + 1),
                "Prediksi"       : ["✅ Normal" if p == 1 else "🚨 Attack" for p in hasil],
                "Confidence (%)" : (np.max(proba, axis=1) * 100).round(2),
                "P(Normal) %"    : (proba[:, 1] * 100).round(2),
                "P(Attack) %"    : (proba[:, 0] * 100).round(2),
            })

            st.markdown("**Detail per baris:**")
            st.dataframe(df_hasil.head(100), use_container_width=True)

            if len(df_hasil) > 100:
                st.caption(f"Menampilkan 100 dari {len(df_hasil):,} baris.")

            # Download
            csv_out = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil (CSV)",
                data=csv_out,
                file_name="hasil_prediksi.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.info("⬆️ Silakan upload file CSV di atas untuk memulai prediksi.")

# ============================================================
# MENU 2 — INPUT MANUAL
# ============================================================
else:

    st.markdown("### ✏️ Input Nilai Fitur Jaringan")
    st.markdown("""
    <div class="info-card">
    <b>📋 Petunjuk:</b> Masukkan nilai fitur lalu lintas jaringan,
    lalu klik <b>Prediksi Sekarang</b>.
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_manual"):

        st.markdown("#### 🔹 Fitur Paket & Port")
        c1, c2, c3 = st.columns(3)
        with c1:
            dst_port         = st.number_input("Destination Port",           min_value=0,   max_value=65535, value=80)
            flow_duration    = st.number_input("Flow Duration",               min_value=0,   value=1000000)
            total_fwd        = st.number_input("Total Fwd Packets",           min_value=0,   value=5)
            total_len_fwd    = st.number_input("Total Length of Fwd Packets", min_value=0,   value=500)
        with c2:
            fwd_pkt_max      = st.number_input("Fwd Packet Length Max",       min_value=0,   value=200)
            fwd_pkt_min      = st.number_input("Fwd Packet Length Min",       min_value=0,   value=20)
            fwd_pkt_mean     = st.number_input("Fwd Packet Length Mean",      min_value=0.0, value=60.0,   format="%.4f")
            fwd_pkt_std      = st.number_input("Fwd Packet Length Std",       min_value=0.0, value=10.0,   format="%.4f")
        with c3:
            bwd_pkt_max      = st.number_input("Bwd Packet Length Max",       min_value=0,   value=100)
            bwd_pkt_min      = st.number_input("Bwd Packet Length Min",       min_value=0,   value=0)
            bwd_pkt_mean     = st.number_input("Bwd Packet Length Mean",      min_value=0.0, value=50.0,   format="%.4f")
            bwd_pkt_std      = st.number_input("Bwd Packet Length Std",       min_value=0.0, value=10.0,   format="%.4f")

        st.markdown("#### 🔹 Flow & IAT")
        c4, c5, c6 = st.columns(3)
        with c4:
            flow_bytes       = st.number_input("Flow Bytes/s",                min_value=0.0, value=5000.0,    format="%.4f")
            flow_pkts        = st.number_input("Flow Packets/s",              min_value=0.0, value=10.0,      format="%.4f")
            flow_iat_mean    = st.number_input("Flow IAT Mean",               min_value=0.0, value=100000.0,  format="%.4f")
            flow_iat_std     = st.number_input("Flow IAT Std",                min_value=0.0, value=50000.0,   format="%.4f")
        with c5:
            flow_iat_max     = st.number_input("Flow IAT Max",                min_value=0.0, value=500000.0,  format="%.4f")
            flow_iat_min     = st.number_input("Flow IAT Min",                min_value=0.0, value=1000.0,    format="%.4f")
            fwd_iat_total    = st.number_input("Fwd IAT Total",               min_value=0.0, value=500000.0,  format="%.4f")
            fwd_iat_mean     = st.number_input("Fwd IAT Mean",                min_value=0.0, value=100000.0,  format="%.4f")
        with c6:
            fwd_iat_std      = st.number_input("Fwd IAT Std",                 min_value=0.0, value=50000.0,   format="%.4f")
            fwd_iat_max      = st.number_input("Fwd IAT Max",                 min_value=0.0, value=300000.0,  format="%.4f")
            fwd_iat_min      = st.number_input("Fwd IAT Min",                 min_value=0.0, value=10000.0,   format="%.4f")
            bwd_iat_total    = st.number_input("Bwd IAT Total",               min_value=0.0, value=200000.0,  format="%.4f")

        st.markdown("#### 🔹 Header & Flag")
        c7, c8, c9 = st.columns(3)
        with c7:
            bwd_iat_mean     = st.number_input("Bwd IAT Mean",                min_value=0.0, value=100000.0,  format="%.4f")
            bwd_iat_std      = st.number_input("Bwd IAT Std",                 min_value=0.0, value=50000.0,   format="%.4f")
            bwd_iat_max      = st.number_input("Bwd IAT Max",                 min_value=0.0, value=300000.0,  format="%.4f")
            bwd_iat_min      = st.number_input("Bwd IAT Min",                 min_value=0.0, value=0.0,       format="%.4f")
        with c8:
            fwd_header       = st.number_input("Fwd Header Length",           min_value=0,   value=20)
            bwd_header       = st.number_input("Bwd Header Length",           min_value=0,   value=20)
            fwd_pkts_s       = st.number_input("Fwd Packets/s",               min_value=0.0, value=5.0,       format="%.4f")
            bwd_pkts_s       = st.number_input("Bwd Packets/s",               min_value=0.0, value=3.0,       format="%.4f")
        with c9:
            fin_flag         = st.number_input("FIN Flag Count",              min_value=0,   value=0)
            psh_flag         = st.number_input("PSH Flag Count",              min_value=0,   value=0)
            ack_flag         = st.number_input("ACK Flag Count",              min_value=0,   value=1)
            min_pkt_len      = st.number_input("Min Packet Length",           min_value=0,   value=20)

        st.markdown("#### 🔹 Statistik Paket & Lainnya")
        c10, c11, c12 = st.columns(3)
        with c10:
            max_pkt_len      = st.number_input("Max Packet Length",           min_value=0,   value=200)
            pkt_len_mean     = st.number_input("Packet Length Mean",          min_value=0.0, value=60.0,      format="%.4f")
            pkt_len_std      = st.number_input("Packet Length Std",           min_value=0.0, value=20.0,      format="%.4f")
            pkt_len_var      = st.number_input("Packet Length Variance",      min_value=0.0, value=400.0,     format="%.4f")
        with c11:
            avg_pkt_size     = st.number_input("Average Packet Size",         min_value=0.0, value=60.0,      format="%.4f")
            subflow_fwd_bytes= st.number_input("Subflow Fwd Bytes",           min_value=0,   value=500)
            init_win_fwd     = st.number_input("Init Win Bytes Forward",      min_value=-1,  value=65535)
            init_win_bwd     = st.number_input("Init Win Bytes Backward",     min_value=-1,  value=65535)
        with c12:
            act_data_pkt     = st.number_input("act_data_pkt_fwd",           min_value=0,   value=1)
            min_seg_fwd      = st.number_input("min_seg_size_forward",        min_value=0,   value=20)
            active_mean      = st.number_input("Active Mean",                 min_value=0.0, value=0.0,       format="%.4f")
            active_max       = st.number_input("Active Max",                  min_value=0.0, value=0.0,       format="%.4f")

        c13, c14 = st.columns(2)
        with c13:
            active_min       = st.number_input("Active Min",                  min_value=0.0, value=0.0,       format="%.4f")
            idle_mean        = st.number_input("Idle Mean",                   min_value=0.0, value=0.0,       format="%.4f")
        with c14:
            idle_max         = st.number_input("Idle Max",                    min_value=0.0, value=0.0,       format="%.4f")
            idle_min         = st.number_input("Idle Min",                    min_value=0.0, value=0.0,       format="%.4f")

        submitted = st.form_submit_button(
            "🔍 Prediksi Sekarang",
            type="primary",
            use_container_width=True
        )

    if submitted:

        # Buat dict sesuai urutan FITUR_MODEL
        nilai = {
            'Destination Port'            : float(dst_port),
            'Flow Duration'               : float(flow_duration),
            'Total Fwd Packets'           : float(total_fwd),
            'Total Length of Fwd Packets' : float(total_len_fwd),
            'Fwd Packet Length Max'       : float(fwd_pkt_max),
            'Fwd Packet Length Min'       : float(fwd_pkt_min),
            'Fwd Packet Length Mean'      : float(fwd_pkt_mean),
            'Fwd Packet Length Std'       : float(fwd_pkt_std),
            'Bwd Packet Length Max'       : float(bwd_pkt_max),
            'Bwd Packet Length Min'       : float(bwd_pkt_min),
            'Bwd Packet Length Mean'      : float(bwd_pkt_mean),
            'Bwd Packet Length Std'       : float(bwd_pkt_std),
            'Flow Bytes/s'                : float(flow_bytes),
            'Flow Packets/s'              : float(flow_pkts),
            'Flow IAT Mean'               : float(flow_iat_mean),
            'Flow IAT Std'                : float(flow_iat_std),
            'Flow IAT Max'                : float(flow_iat_max),
            'Flow IAT Min'                : float(flow_iat_min),
            'Fwd IAT Total'               : float(fwd_iat_total),
            'Fwd IAT Mean'                : float(fwd_iat_mean),
            'Fwd IAT Std'                 : float(fwd_iat_std),
            'Fwd IAT Max'                 : float(fwd_iat_max),
            'Fwd IAT Min'                 : float(fwd_iat_min),
            'Bwd IAT Total'               : float(bwd_iat_total),
            'Bwd IAT Mean'                : float(bwd_iat_mean),
            'Bwd IAT Std'                 : float(bwd_iat_std),
            'Bwd IAT Max'                 : float(bwd_iat_max),
            'Bwd IAT Min'                 : float(bwd_iat_min),
            'Fwd Header Length'           : float(fwd_header),
            'Bwd Header Length'           : float(bwd_header),
            'Fwd Packets/s'               : float(fwd_pkts_s),
            'Bwd Packets/s'               : float(bwd_pkts_s),
            'Min Packet Length'           : float(min_pkt_len),
            'Max Packet Length'           : float(max_pkt_len),
            'Packet Length Mean'          : float(pkt_len_mean),
            'Packet Length Std'           : float(pkt_len_std),
            'Packet Length Variance'      : float(pkt_len_var),
            'FIN Flag Count'              : float(fin_flag),
            'PSH Flag Count'              : float(psh_flag),
            'ACK Flag Count'              : float(ack_flag),
            'Average Packet Size'         : float(avg_pkt_size),
            'Subflow Fwd Bytes'           : float(subflow_fwd_bytes),
            'Init_Win_bytes_forward'      : float(init_win_fwd),
            'Init_Win_bytes_backward'     : float(init_win_bwd),
            'act_data_pkt_fwd'            : float(act_data_pkt),
            'min_seg_size_forward'        : float(min_seg_fwd),
            'Active Mean'                 : float(active_mean),
            'Active Max'                  : float(active_max),
            'Active Min'                  : float(active_min),
            'Idle Mean'                   : float(idle_mean),
            'Idle Max'                    : float(idle_max),
            'Idle Min'                    : float(idle_min),
        }

        input_df = pd.DataFrame([nilai])[FITUR_MODEL]

        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label      = "Normal" if pred == 1 else "Attack"
        confidence = max(proba) * 100
        p_normal   = proba[1] * 100
        p_attack   = proba[0] * 100

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        if label == "Normal":
            st.markdown("""
            <div class="hasil-normal">
                <h2>✅ NORMAL TRAFFIC</h2>
                <p>Lalu lintas jaringan terdeteksi <b>aman</b> dan tidak mengandung serangan</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hasil-attack">
                <h2>🚨 TERDETEKSI SERANGAN!</h2>
                <p>Lalu lintas jaringan teridentifikasi sebagai <b>Port Scanning Attack</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown(f"**Tingkat Keyakinan Model: {confidence:.2f}%**")
            st.progress(confidence / 100)
        with col_c2:
            st.metric("Prediksi", label)

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
