# ============================================================
# APLIKASI DETEKSI INTRUSI JARINGAN
# Algoritma XGBoost - Dataset CICIDS2017
# ============================================================
# Cara menjalankan:
#   streamlit run app_streamlit.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# ── Konfigurasi halaman ─────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Intrusi Jaringan - XGBoost",
    page_icon="🛡️",
    layout="centered"
)

# ── Judul aplikasi ──────────────────────────────────────────
st.title("🛡️ Klasifikasi Intrusi Jaringan")
st.caption("Menggunakan Algoritma XGBoost | Dataset CICIDS2017")
st.divider()

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
    st.success("✅ Model XGBoost berhasil dimuat")
except Exception as e:
    st.error(f"❌ Model gagal dimuat: {e}")
    st.info("Pastikan file **xgboost_cicids2017.json** ada di folder yang sama dengan app ini.")
    st.stop()

# Ambil daftar fitur dari model
fitur_model = model.get_booster().feature_names

# ============================================================
# PILIH CARA PREDIKSI
# ============================================================
st.subheader("Pilih Cara Prediksi")

cara = st.radio(
    "Pilih metode input data:",
    options=["📁 Upload File CSV", "✏️ Input Manual"],
    horizontal=True
)

st.divider()

# ============================================================
# CARA 1: UPLOAD FILE CSV
# ============================================================
if cara == "📁 Upload File CSV":

    st.markdown("### Upload File CSV")
    st.markdown("""
    **Ketentuan file CSV:**
    - File berisi data trafik jaringan (tanpa kolom label)
    - Nama kolom harus sesuai dengan fitur yang digunakan saat training
    """)

    file = st.file_uploader("Pilih file CSV", type=["csv"])

    if file is not None:
        # Baca file
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        # Hapus kolom label jika ada (jaga-jaga)
        for kolom_label in ["Label", "label", "Attack Type", "attack type"]:
            if kolom_label in df.columns:
                df.drop(columns=[kolom_label], inplace=True)

        st.success(f"✅ File berhasil dibaca: **{len(df):,} baris**, **{df.shape[1]} kolom**")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption("Menampilkan 5 baris pertama")

        # Tombol prediksi
        if st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True):

            with st.spinner("Sedang memproses prediksi..."):

                # Sesuaikan kolom dengan fitur model
                df_pred = pd.DataFrame(columns=fitur_model)
                for fitur in fitur_model:
                    if fitur in df.columns:
                        df_pred[fitur] = df[fitur]
                    else:
                        df_pred[fitur] = 0  # isi 0 jika kolom tidak ada

                # Bersihkan data
                df_pred = df_pred.apply(pd.to_numeric, errors="coerce").fillna(0)
                df_pred = df_pred.replace([np.inf, -np.inf], 0)

                # Prediksi
                hasil = model.predict(df_pred)
                proba = model.predict_proba(df_pred)

                # Buat kolom hasil
                df["Prediksi"]       = ["Normal" if p == 1 else "Attack" for p in hasil]
                df["Confidence (%)"] = (np.max(proba, axis=1) * 100).round(2)

            # ── Tampilkan ringkasan ──
            st.divider()
            st.markdown("### Hasil Prediksi")

            jml_normal = (df["Prediksi"] == "Normal").sum()
            jml_attack = (df["Prediksi"] == "Attack").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Data", f"{len(df):,}")
            col2.metric("✅ Normal", f"{jml_normal:,}")
            col3.metric("🚨 Attack", f"{jml_attack:,}")

            # Tabel hasil (100 baris pertama)
            st.dataframe(
                df[["Prediksi", "Confidence (%)"]].head(100),
                use_container_width=True
            )
            if len(df) > 100:
                st.caption(f"Menampilkan 100 dari {len(df):,} baris. Download untuk data lengkap.")

            # Tombol download
            csv_hasil = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil Prediksi (CSV)",
                data=csv_hasil,
                file_name="hasil_prediksi.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================================
# CARA 2: INPUT MANUAL
# ============================================================
else:

    st.markdown("### Input Nilai Fitur Jaringan")
    st.markdown("Masukkan nilai fitur lalu lintas jaringan di bawah ini:")

    # ── Form input fitur utama ──
    col1, col2 = st.columns(2)

    with col1:
        dst_port         = st.number_input("Destination Port",          min_value=0,   max_value=65535, value=80)
        flow_duration    = st.number_input("Flow Duration",             min_value=0,   value=1000000)
        total_fwd        = st.number_input("Total Fwd Packets",         min_value=0,   value=5)
        total_bwd        = st.number_input("Total Backward Packets",    min_value=0,   value=3)
        flow_bytes       = st.number_input("Flow Bytes/s",              min_value=0.0, value=5000.0,  format="%.2f")
        flow_packets     = st.number_input("Flow Packets/s",            min_value=0.0, value=10.0,    format="%.2f")
        fwd_iat_min      = st.number_input("Fwd IAT Min",               min_value=0.0, value=100000.0, format="%.2f")

    with col2:
        pkt_len_mean     = st.number_input("Packet Length Mean",        min_value=0.0, value=60.0,   format="%.2f")
        pkt_len_std      = st.number_input("Packet Length Std",         min_value=0.0, value=20.0,   format="%.2f")
        fwd_pkt_len_mean = st.number_input("Fwd Packet Length Mean",    min_value=0.0, value=60.0,   format="%.2f")
        bwd_pkt_len_std  = st.number_input("Bwd Packet Length Std",     min_value=0.0, value=15.0,   format="%.2f")
        psh_flag         = st.number_input("PSH Flag Count",            min_value=0,   value=0)
        fwd_header_len   = st.number_input("Fwd Header Length",         min_value=0,   value=20)
        flow_duration2   = st.number_input("Flow Duration (alt)",       min_value=0,   value=500000)

    # ── Tombol prediksi ──
    if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):

        # Buat dataframe kosong sesuai fitur model
        input_df = pd.DataFrame(0.0, index=[0], columns=fitur_model)

        # Isi nilai dari form
        nilai_input = {
            "Destination Port"        : dst_port,
            "Flow Duration"           : flow_duration,
            "Total Fwd Packets"       : total_fwd,
            "Total Backward Packets"  : total_bwd,
            "Flow Bytes/s"            : flow_bytes,
            "Flow Packets/s"          : flow_packets,
            "Fwd IAT Min"             : fwd_iat_min,
            "Packet Length Mean"      : pkt_len_mean,
            "Packet Length Std"       : pkt_len_std,
            "Fwd Packet Length Mean"  : fwd_pkt_len_mean,
            "Bwd Packet Length Std"   : bwd_pkt_len_std,
            "PSH Flag Count"          : psh_flag,
            "Fwd Header Length"       : fwd_header_len,
            "Flow Duration"           : flow_duration2,
        }

        for nama_fitur, nilai in nilai_input.items():
            if nama_fitur in input_df.columns:
                input_df[nama_fitur] = float(nilai)

        # Prediksi
        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label      = "Normal" if pred == 1 else "Attack"
        confidence = max(proba) * 100

        # ── Tampilkan hasil ──
        st.divider()
        st.markdown("### Hasil Prediksi")

        if label == "Normal":
            st.success(f"## ✅ NORMAL")
            st.success(f"Trafik jaringan terdeteksi **aman** (Normal Traffic)")
        else:
            st.error(f"## 🚨 ATTACK")
            st.error(f"Terdeteksi **serangan jaringan** (Port Scanning)!")

        # Confidence
        st.markdown(f"**Confidence: {confidence:.2f}%**")
        st.progress(confidence / 100)

        # Detail probabilitas
        st.markdown("**Probabilitas per kelas:**")
        col_a, col_b = st.columns(2)
        col_a.metric("Normal", f"{proba[1]*100:.2f}%")
        col_b.metric("Attack", f"{proba[0]*100:.2f}%")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Klasifikasi Intrusi Jaringan Menggunakan Algoritma XGBoost")
