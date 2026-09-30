"""IndoToxic 2024 — Sistem Analisis Toksisitas Teks Bahasa Indonesia.
Dashboard formal tanpa sidebar dengan visualisasi flowchart, kontras tinggi,
highlighting kata toksik/positif, toxicity meter, wordcloud kata populer,
dan pengurutan topik interaktif (sorting topic).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Pastikan root proyek terdaftar dalam sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.services.charts import (
    COLORS,
    generate_wordcloud_image,
    get_topic_summary_table,
    plot_confidence_gauge,
    plot_confusion_matrix,
    plot_regional_lexicon_distribution,
    plot_sorted_topics_distribution,
    plot_subcategories_distribution,
    plot_subtopics_bar,
    plot_token_length_distribution,
    plot_top_words_bar,
    plot_toxicity_distribution,
    plot_toxicity_meter_bar,
)
from dashboard.services.inference import IndoToxicInferenceService

# Konfigurasi Halaman (Sidebar disembunyikan sepenuhnya)
st.set_page_config(
    page_title="IndoToxic 2024: Sistem Analisis Toksisitas Teks",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-Contrast Professional Styling
st.markdown(
    """
    <style>
        /* Sembunyikan sidebar sepenuhnya */
        [data-testid="stSidebar"], 
        section[data-testid="stSidebar"], 
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Tipografi & Warna Dasar Kontras Tinggi */
        html, body, [class*="css"] {
            font-family: Arial, Helvetica, sans-serif !important;
            color: #FFFFFF !important;
            background-color: #0B1120 !important;
        }

        .main .block-container {
            max-width: 1400px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Header Formal */
        .app-header {
            background-color: #1E293B;
            border: 2px solid #3B82F6;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }
        .app-title {
            font-size: 24px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .app-subtitle {
            font-size: 14px;
            color: #E2E8F0;
            line-height: 1.5;
        }

        /* Tabs Navigasi Horizontal Formal */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid #334155;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E293B !important;
            border: 1px solid #475569 !important;
            border-radius: 6px !important;
            color: #E2E8F0 !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            padding: 10px 18px !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563EB !important;
            border: 1px solid #60A5FA !important;
            color: #FFFFFF !important;
        }

        /* Kartu Metrik Kontras Tinggi */
        .metric-card {
            background-color: #1E293B;
            border: 2px solid #475569;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #FFFFFF;
        }
        .metric-label {
            font-size: 12px;
            font-weight: 700;
            color: #CBD5E1;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-top: 4px;
        }

        /* Kontainer Konten Formal */
        .formal-card {
            background-color: #1E293B;
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 14px;
        }
        .formal-card-title {
            font-size: 15px;
            font-weight: 800;
            color: #FFFFFF;
            border-bottom: 2px solid #334155;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }

        /* Highlight Kotak Teks */
        .highlight-container {
            background-color: #0F172A;
            border: 2px solid #475569;
            border-radius: 8px;
            padding: 18px;
            font-size: 15px;
            line-height: 1.8;
            color: #FFFFFF;
            margin: 12px 0;
            word-wrap: break-word;
        }

        /* Badge Status Kontras Penuh */
        .badge-toxic {
            background-color: #DC2626;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 13px;
            padding: 5px 12px;
            border-radius: 4px;
            display: inline-block;
        }
        .badge-safe {
            background-color: #16A34A;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 13px;
            padding: 5px 12px;
            border-radius: 4px;
            display: inline-block;
        }
        .badge-review {
            background-color: #D97706;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 13px;
            padding: 5px 12px;
            border-radius: 4px;
            display: inline-block;
        }
        .badge-gambling {
            background-color: #BE185D;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 13px;
            padding: 5px 12px;
            border-radius: 4px;
            display: inline-block;
        }

        /* Kotak Notifikasi Formal */
        .alert-review {
            background-color: #78350F;
            border: 2px solid #F59E0B;
            color: #FFFFFF;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 12px 0;
            font-size: 13px;
            line-height: 1.5;
        }
        .alert-gambling {
            background-color: #831843;
            border: 2px solid #EC4899;
            color: #FFFFFF;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 12px 0;
            font-size: 13px;
            line-height: 1.5;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #475569;
            border-radius: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model inferensi IndoToxic...")
def get_inference_service() -> IndoToxicInferenceService:
    """Inisialisasi service inferensi berbasis model yang telah disimpan."""
    return IndoToxicInferenceService(models_dir=REPO_ROOT / "models")


@st.cache_data(show_spinner="Memuat dataset korpus...")
def get_processed_data() -> pd.DataFrame:
    """Muat korpus data yang telah diproses."""
    csv_path = REPO_ROOT / "data" / "processed" / "data_cleaned_before_encode.csv"
    if not csv_path.exists():
        st.error(f"Berkas data tidak ditemukan: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    def parse_topics(val):
        if pd.isna(val):
            return []
        try:
            p = ast.literal_eval(val)
            if isinstance(p, list):
                return [str(item).strip() for item in p if str(item).strip()]
        except Exception:
            pass
        return []

    if "topic_list" in df.columns:
        df["topic_list_parsed"] = df["topic_list"].apply(parse_topics)
    return df


@st.cache_data
def get_lexicon_data() -> list[dict[str, Any]]:
    """Muat data leksikon umpatan daerah."""
    json_path = REPO_ROOT / "data" / "umpatan.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f).get("entries", [])
    return []


def main():
    service = get_inference_service()
    df_data = get_processed_data()
    lexicon_entries = get_lexicon_data()
    metrics = service.eval_metrics

    # Header Dashboard Formal
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">IndoToxic 2024: Sistem Analisis Toksisitas Teks Bahasa Indonesia</div>
            <div class="app-subtitle">
                Platform intelijen moderasi konten berbasis ensemble TriModel Random Forest, Calibrated LinearSVC, 
                aturan leksikon umpatan daerah, serta deteksi promosi judi dan togel.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigasi Utama Tab Horizontal
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Inspeksi Teks & Toxicity Meter",
        "Eksplorasi Data & Wordcloud",
        "Ringkasan & Arsitektur",
        "Moderasi Massal (Batch)",
        "Evaluasi & Kalibrasi Model",
        "Leksikon & Pengetahuan Regional",
    ])

    with tab1:
        render_inspector_tab(service)

    with tab2:
        render_eda_tab(df_data)

    with tab3:
        render_overview_tab(df_data, metrics, lexicon_entries)

    with tab4:
        render_batch_tab(service)

    with tab5:
        render_diagnostics_tab(metrics)

    with tab6:
        render_lexicon_tab(lexicon_entries)


# ==============================================================================
# TAB 1: INSPEKSI TEKS & TOXICITY METER (HIGHLIGHTING & REAL-TIME PREDICTION)
# ==============================================================================
def render_inspector_tab(service: IndoToxicInferenceService):
    st.markdown("<div class='formal-card-title'>Input Teks & Deteksi Toksisitas Interaktif</div>", unsafe_allow_html=True)

    preset_map = {
        "— Pilih Contoh Kalimat Standar —": "",
        "Komentar Netral (Diskusi Publik)": "Harga bahan pokok naik lagi, pemerintah perlu menjelaskan rencana pengendaliannya.",
        "Ujaran Kebencian / SARA & Politik": "pendukung prabowo gibran mengatakan jangan memancing amarah dasar bajingan",
        "Umpatan Eksak Regional": "goblok sekali kamu anjing tidak tahu diri perusak bangsa",
        "Dugaan Promosi Judi Online / Togel": "Ayo buruan gabung menang 88 pasti wd togel sidney 23 malam ini",
        "Kombinasi Kalimat Campuran (Kasar & Pujian)": "Dasar bajingan goblok tidak tahu diri, tapi program bantuan ini sangat membantu warga",
    }

    selected_preset = st.selectbox("Pilih Contoh Presets Pengujian:", list(preset_map.keys()))
    default_text = preset_map[selected_preset] if selected_preset != "— Pilih Contoh Kalimat Standar —" else ""

    input_text = st.text_area(
        "Ketik teks komentar untuk diuji:",
        value=default_text,
        height=95,
        placeholder="Masukkan kalimat komentar berbahasa Indonesia di sini...",
    )

    col_btn, _ = st.columns([1.5, 5])
    with col_btn:
        btn_eval = st.button("Jalankan Analisis Teks", type="primary", use_container_width=True)

    if input_text or btn_eval:
        res = service.analyze_single_text(input_text)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 1. TOXICITY METER & STATUS
        st.markdown("<div class='formal-card-title'>Toxicity Meter & Status Klasifikasi</div>", unsafe_allow_html=True)

        m_col1, m_col2, m_col3 = st.columns([1.2, 1.4, 1.4])

        with m_col1:
            st.markdown("<div style='font-size: 12px; font-weight: 700; color: #CBD5E1; margin-bottom: 6px;'>STATUS TOKSISITAS</div>", unsafe_allow_html=True)
            if res["is_toxic"]:
                st.markdown(
                    f"""
                    <div style="background-color: #1E293B; border: 2px solid #DC2626; border-radius: 8px; padding: 18px; text-align: center;">
                        <span class="badge-toxic">TOKSIK</span>
                        <div style="font-size: 13px; color: #FCA5A5; margin-top: 10px; font-weight: 700;">
                            {res['severity_label']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #1E293B; border: 2px solid #16A34A; border-radius: 8px; padding: 18px; text-align: center;">
                        <span class="badge-safe">NON-TOKSIK</span>
                        <div style="font-size: 13px; color: #86EFAC; margin-top: 10px; font-weight: 700;">
                            {res['severity_label']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with m_col2:
            st.markdown("<div style='font-size: 12px; font-weight: 700; color: #CBD5E1; margin-bottom: 6px;'>SKOR TOXICITY METER</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_toxicity_meter_bar(res["class_probabilities"]["toxic"]), use_container_width=True)

        with m_col3:
            st.markdown("<div style='font-size: 12px; font-weight: 700; color: #CBD5E1; margin-bottom: 6px;'>KEYAKINAN MODEL (CONFIDENCE)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_confidence_gauge(res["confidence"], res["is_toxic"]), use_container_width=True)

        # 2. HIGHLIGHTING KATA TOKSIK / POSITIF
        st.markdown("<div class='formal-card-title'>Highlight Kata & Istilah Terdeteksi</div>", unsafe_allow_html=True)

        # Legend Highlighting
        st.markdown(
            """
            <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12px; font-weight: 700;">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="background-color: #DC2626; color: #FFFFFF; padding: 2px 8px; border-radius: 4px;">Merah</span> Kata / Frasa Terindikasi Toksik
                </span>
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="background-color: #166534; color: #DCFCE7; padding: 2px 8px; border-radius: 4px;">Hijau</span> Kata Positif / Konstruktif
                </span>
                <span style="display:flex; align-items:center; gap:6px; color: #CBD5E1;">
                    Teks Polos: Konteks Netral
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Box Teks Ter-highlight
        st.markdown(
            f"""
            <div class="highlight-container">
                {res['highlighted_html']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tabel Token Terdeteksi
        if res["detected_tokens"]:
            token_records = [
                {
                    "Kata / Frasa Terdeteksi": d["token"],
                    "Klasifikasi Token": d["kategori"],
                    "Sumber Referensi": d["sumber"],
                }
                for d in res["detected_tokens"]
            ]
            st.dataframe(pd.DataFrame(token_records), hide_index=True, use_container_width=True)
        else:
            st.caption("ℹ️ Tidak ditemukan kata kasar atau istilah leksikon spesifik dalam teks ini (Kalimat Netral).")

        # Alert Kasus Review Manual (Discrepancy)
        if res["needs_review"]:
            st.markdown(
                """
                <div class="alert-review">
                    <b>[PERHATIAN] Sinyal Tinjauan Manual Aktif (Human-in-the-Loop):</b><br/>
                    Aturan leksikon menemukan kata kasar dalam teks, namun Calibrated LinearSVC mengklasifikasikannya sebagai <b>Non-Toksik</b>.
                    Kondisi ini umumnya terjadi pada kata informal/pujian gaul atau sarkasme. Sistem merekomendasikan verifikasi manual oleh moderator.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Alert Deteksi Iklan Judi
        if res["gambling_promo"]:
            g_matches = ", ".join([f"{m['term']} {m['number']}" for m in res["gambling_matches"]])
            st.markdown(
                f"""
                <div class="alert-gambling">
                    <b>[INDIKASI PROMOSI JUDI ONLINE / TOGEL]:</b><br/>
                    Teks memenuhi pola promosi nomor judi: <b>{g_matches}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Tab Rincian Diagnostik Lanjutan
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        sub1, sub2, sub3 = st.tabs(["Estimasi TriModel & Subtopik", "Pencocokan Leksikon Daerah", "Prapemrosesan Teks"])

        with sub1:
            c_tm1, c_tm2 = st.columns([1, 1.2])
            with c_tm1:
                st.markdown("<div style='font-size: 13px; font-weight: 700; margin-bottom: 6px;'>Distribusi Probabilitas Mood & Sentimen</div>", unsafe_allow_html=True)
                m_df = pd.DataFrame(list(res["mood_proba"].items()), columns=["Kategori Mood", "Probabilitas"])
                s_df = pd.DataFrame(list(res["sentiment_proba"].items()), columns=["Kategori Sentimen", "Probabilitas"])
                st.dataframe(m_df, hide_index=True, use_container_width=True)
                st.dataframe(s_df, hide_index=True, use_container_width=True)

            with c_tm2:
                if res["subtopics"]:
                    st.plotly_chart(plot_subtopics_bar(res["subtopics"]), use_container_width=True)

        with sub2:
            st.markdown("<div style='font-size: 13px; font-weight: 700; margin-bottom: 6px;'>Hasil Audit Leksikon Umpatan</div>", unsafe_allow_html=True)
            if res["rule_matches"]:
                records = [
                    {"Istilah Terdeteksi": m["expression"], "Wilayah / Dialek": ", ".join(m["regions"]) if m["regions"] else "Nasional / Indonesia"}
                    for m in res["rule_matches"]
                ]
                st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
            else:
                st.info("Tidak ada istilah leksikon umpatan yang terdeteksi.")

        with sub3:
            st.markdown("<div style='font-size: 13px; font-weight: 700; margin-bottom: 6px;'>Teks Hasil Normalisasi</div>", unsafe_allow_html=True)
            st.code(res["cleaned_text"], language="text")
            st.caption(f"Persentase kosakata yang dikenali model saat pelatihan: {res['vocabulary_coverage']*100:.1f}%")


# ==============================================================================
# TAB 2: EKSPLORASI DATA, WORDCLOUD & PENGURUTAN TOPIK (SORTING TOPIC)
# ==============================================================================
def render_eda_tab(df: pd.DataFrame):
    if df.empty:
        st.warning("Data korpus tidak tersedia.")
        return

    # 1. PENGURUTAN TOPIK INTERAKTIF (SORTING TOPIC)
    st.markdown("<div class='formal-card-title'>Analisis & Pengurutan Topik (Sorting Topic)</div>", unsafe_allow_html=True)

    st_col1, st_col2 = st.columns([2, 1])
    with st_col1:
        sort_choice = st.selectbox(
            "Urutkan Data Topik Berdasarkan:",
            [
                "Jumlah Komentar Terbanyak (Volume Tertinggi)",
                "Jumlah Komentar Tersedikit (Volume Terendah)",
                "Persentase Toksisitas Tertinggi (% Toksik Terbanyak)",
                "Persentase Toksisitas Terendah (% Toksik Tersedikit)",
                "Nama Topik (A - Z)",
                "Nama Topik (Z - A)",
            ],
            index=0,
        )
    with st_col2:
        top_k = st.slider("Tampilkan Jumlah Topik:", min_value=5, max_value=min(25, df["topic"].nunique()), value=10)

    # Mapping sorting key
    sort_key_map = {
        "Jumlah Komentar Terbanyak (Volume Tertinggi)": "volume_desc",
        "Jumlah Komentar Tersedikit (Volume Terendah)": "volume_asc",
        "Persentase Toksisitas Tertinggi (% Toksik Terbanyak)": "toxic_rate_desc",
        "Persentase Toksisitas Terendah (% Toksik Tersedikit)": "toxic_rate_asc",
        "Nama Topik (A - Z)": "name_asc",
        "Nama Topik (Z - A)": "name_desc",
    }
    active_sort_key = sort_key_map[sort_choice]

    # Visualisasi & Tabel Topik Terurut
    st.plotly_chart(plot_sorted_topics_distribution(df, sort_by=active_sort_key, top_n=top_k), use_container_width=True)

    with st.expander("Lihat Rincian Tabel Statistik Topik Terurut", expanded=False):
        topic_table = get_topic_summary_table(df, sort_by=active_sort_key).head(top_k)
        st.dataframe(topic_table, hide_index=True, use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 2. WORDCLOUD & KATA POPULER
    st.markdown("<div class='formal-card-title'>Wordcloud & Frekuensi Kata Populer Korpus</div>", unsafe_allow_html=True)

    wc_choice = st.radio(
        "Pilih Subset Data untuk Wordcloud & Kata Populer:",
        ["Kata Populer pada Komentar Toksik", "Kata Populer pada Komentar Non-Toksik", "Kata Populer Keseluruhan Korpus"],
        horizontal=True,
    )

    wc_cat_map = {
        "Kata Populer pada Komentar Toksik": "toxic",
        "Kata Populer pada Komentar Non-Toksik": "non_toxic",
        "Kata Populer Keseluruhan Korpus": "all",
    }
    active_wc_cat = wc_cat_map[wc_choice]

    wc_col1, wc_col2 = st.columns([1.2, 1])

    with wc_col1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; margin-bottom: 6px;'>Visualisasi Wordcloud (Stopwords Bahasa Indonesia Dieliminasi)</div>", unsafe_allow_html=True)
        wc_img = generate_wordcloud_image(df, category=active_wc_cat, max_words=100)
        st.image(wc_img, use_container_width=True)

    with wc_col2:
        st.plotly_chart(plot_top_words_bar(df, category=active_wc_cat, top_n=15), use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 3. DISTRIBUSI KORPUS UMUM
    st.markdown("<div class='formal-card-title'>Distribusi Prevalensi & Subkategori Toksisitas</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(plot_toxicity_distribution(df), use_container_width=True)
    with g2:
        st.plotly_chart(plot_subcategories_distribution(df), use_container_width=True)

    st.plotly_chart(plot_token_length_distribution(df), use_container_width=True)


# ==============================================================================
# TAB 3: RINGKASAN & ARSITEKTUR SISTEM
# ==============================================================================
def render_overview_tab(df: pd.DataFrame, metrics: dict[str, Any], lexicon: list[dict[str, Any]]):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">23.009</div>
                <div class="metric-label">Total Sampel Korpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        tox_rate = (df["toxicity"].mean() * 100) if not df.empty else 9.3
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #EF4444;">{tox_rate:.1f}%</div>
                <div class="metric-label">Prevalensi Toksisitas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        acc = metrics.get("toxicity_svm", {}).get("accuracy", 0.913) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #22C55E;">{acc:.1f}%</div>
                <div class="metric-label">Akurasi Calibrated SVM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        brier = metrics.get("toxicity_svm", {}).get("brier_score_loss", 0.069)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #60A5FA;">{brier:.3f}</div>
                <div class="metric-label">Brier Score Loss</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Diagram Alir Sistem (Graphviz Vectorized Flowchart)
    st.markdown("<div class='formal-card-title'>Diagram Alur Pemrosesan Sistem Cerdas (Pipeline Flowchart)</div>", unsafe_allow_html=True)

    flowchart_dot = """
    digraph IndoToxicFlowchart {
        graph [rankdir=TB, bgcolor="#1E293B", fontname="Arial", nodesep=0.4, ranksep=0.5];
        node [fontname="Arial", fontsize=11, fontcolor="#FFFFFF", shape=box, style="filled,rounded", penwidth=2.0];
        edge [color="#94A3B8", penwidth=2.0, fontname="Arial", fontsize=10, fontcolor="#CBD5E1", arrowhead=vee];

        DATA [label="Dataset Anotasi 2024\\n(23.009 Komentar)", fillcolor="#0F172A", color="#3B82F6"];
        PREP [label="Prapemrosesan & Normalisasi\\n(Pembersihan Teks, Slang, Emoji)", fillcolor="#0F172A", color="#3B82F6"];
        SPLIT [label="Stratified Group K-Fold Split\\n(80% Latih / 20% Uji, Bebas Leakage)", fillcolor="#0F172A", color="#3B82F6"];

        TFIDF [label="Ekstraksi Fitur TF-IDF\\n(n-gram 1-2, 2.500 Fitur)", fillcolor="#1E293B", color="#8B5CF6"];
        TRI [label="TriModel Random Forest\\n- Mood Proxy (Angry/Happy)\\n- Sentimen Proxy (Neg/Pos)\\n- 10 Subtopik Multilabel", fillcolor="#1E293B", color="#8B5CF6"];
        PROTO [label="Out-of-Fold Proto-Features\\n(Probabilitas Prediktif)", fillcolor="#1E293B", color="#8B5CF6"];
        
        LEX [label="Leksikon Umpatan Daerah\\n(umpatan.json: 175 Istilah)", fillcolor="#1E293B", color="#D97706"];
        RULES [label="Fitur Aturan Leksikon\\n(Hit Flag & Match Count)", fillcolor="#1E293B", color="#D97706"];

        SVM [label="Calibrated LinearSVC (ToxicOrNot)\\n(Ensemble Fitur Teks, Proto, dan Rules)", fillcolor="#065F46", color="#10B981"];

        DECISION [label="Keputusan Toksisitas Primer\\n& Skor Probabilitas Terkalibrasi", fillcolor="#047857", color="#10B981"];
        REVIEW [label="Sinyal Review Manual (Human-in-the-Loop)\\n(Pemicu: Aturan=Toksik & SVM=Non-Toksik)", fillcolor="#78350F", color="#F59E0B"];
        GAMBLE [label="Detektor Pola Promosi Judi/Togel\\n(Keyword + 2-3 Digit Nomor)", fillcolor="#831843", color="#EC4899"];

        DATA -> PREP;
        PREP -> SPLIT;
        SPLIT -> TFIDF;
        TFIDF -> TRI;
        TRI -> PROTO;

        TFIDF -> SVM;
        PROTO -> SVM;
        LEX -> RULES;
        RULES -> SVM;

        SVM -> DECISION;
        RULES -> REVIEW [style=dashed, color="#F59E0B", label="Discrepancy Check"];
        SVM -> REVIEW [style=dashed, color="#F59E0B"];
        DATA -> GAMBLE;
    }
    """

    st.graphviz_chart(flowchart_dot, use_container_width=True)


# ==============================================================================
# TAB 4: MODERASI MASSAL (BATCH)
# ==============================================================================
def render_batch_tab(service: IndoToxicInferenceService):
    st.markdown("<div class='formal-card-title'>Pemindaian Data Komentar Massal</div>", unsafe_allow_html=True)

    up_file = st.file_uploader("Unggah Berkas Komentar (.CSV atau .XLSX):", type=["csv", "xlsx"])

    sample_text = "text\n" "Program bantuan sosial ini sangat bermanfaat bagi warga.\n" "dasar bajingan goblok perusak negara\n" "Ayo menang 88 pasti wd togel sidney 23\n" "Kebijakan transportasi publik perlu dievaluasi berkala.\n"
    st.download_button("Unduh Template CSV Contoh", sample_text, "template_komentar.csv", "text/csv")

    if up_file is not None:
        try:
            df_in = pd.read_excel(up_file) if up_file.name.endswith(".xlsx") else pd.read_csv(up_file)
            st.success(f"Berkas berhasil dimuat: {len(df_in):,} baris data.")
        except Exception as e:
            st.error(f"Gagal memuat berkas: {e}")
            return

        col_text = st.selectbox("Pilih kolom teks komentar:", list(df_in.columns))
        limit_n = st.slider("Jumlah baris yang diproses:", min_value=5, max_value=min(1000, len(df_in)), value=min(100, len(df_in)))

        if st.button("Jalankan Pemindaian Batch", type="primary"):
            texts = df_in[col_text].dropna().astype(str).head(limit_n).tolist()

            p_bar = st.progress(0)
            res_df = service.analyze_batch(texts)
            p_bar.progress(100)

            k1, k2, k3, k4 = st.columns(4)
            n_total = len(res_df)
            n_toxic = sum(res_df["Prediksi"] == "Toksik")
            n_rev = sum(res_df["Status Review"] == "Perlu Review")
            n_gamb = sum(res_df["Iklan Judi"] == "Terindikasi")

            with k1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{n_total}</div><div class='metric-label'>Total Dipindai</div></div>", unsafe_allow_html=True)
            with k2:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#EF4444;'>{n_toxic}</div><div class='metric-label'>Terdeteksi Toksik</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#F59E0B;'>{n_rev}</div><div class='metric-label'>Perlu Tinjauan</div></div>", unsafe_allow_html=True)
            with k4:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#EC4899;'>{n_gamb}</div><div class='metric-label'>Indikasi Judi</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='formal-card-title'>Tabel Hasil Klasifikasi Batch</div>", unsafe_allow_html=True)

            f_choice = st.radio("Filter Tampilan Tabel:", ["Semua Data", "Hanya Toksik", "Hanya Butuh Review", "Hanya Indikasi Judi"], horizontal=True)

            out_view = res_df.copy()
            if f_choice == "Hanya Toksik":
                out_view = out_view[out_view["Prediksi"] == "Toksik"]
            elif f_choice == "Hanya Butuh Review":
                out_view = out_view[out_view["Status Review"] == "Perlu Review"]
            elif f_choice == "Hanya Indikasi Judi":
                out_view = out_view[out_view["Iklan Judi"] == "Terindikasi"]

            st.dataframe(out_view, use_container_width=True, height=340)

            csv_dl = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh Hasil Anotasi Lengkap (CSV)", csv_dl, "hasil_moderasi_indotoxic.csv", "text/csv")


# ==============================================================================
# TAB 5: EVALUASI & KALIBRASI MODEL
# ==============================================================================
def render_diagnostics_tab(metrics: dict[str, Any]):
    st.markdown("<div class='formal-card-title'>Evaluasi Kinerja Model pada Partisi Uji (Held-Out Test 20%)</div>", unsafe_allow_html=True)

    svm_m = metrics.get("toxicity_svm", {})
    rule_m = metrics.get("toxicity_rule_only", {})
    or_m = metrics.get("toxicity_or_experiment", {})

    b_data = [
        {
            "Metode Klasifikasi": "Calibrated LinearSVC (Primer)",
            "Akurasi": f"{svm_m.get('accuracy', 0.913)*100:.2f}%",
            "Precision Macro": f"{svm_m.get('precision_macro', 0.770)*100:.2f}%",
            "Recall Macro": f"{svm_m.get('recall_macro', 0.538)*100:.2f}%",
            "F1 Macro": f"{svm_m.get('f1_macro', 0.548)*100:.2f}%",
            "Brier Score Loss": f"{svm_m.get('brier_score_loss', 0.069):.4f}",
            "Keterangan": "Ensemble TF-IDF, Proto-Features, dan Fitur Aturan",
        },
        {
            "Metode Klasifikasi": "Rule-Based Lexicon (Umpatan Saja)",
            "Akurasi": f"{rule_m.get('accuracy', 0.759)*100:.2f}%",
            "Precision Macro": f"{rule_m.get('precision_macro', 0.529)*100:.2f}%",
            "Recall Macro": f"{rule_m.get('recall_macro', 0.559)*100:.2f}%",
            "F1 Macro": f"{rule_m.get('f1_macro', 0.525)*100:.2f}%",
            "Brier Score Loss": "-",
            "Keterangan": "Pencocokan kata kasar eksak tanpa kesadaran konteks",
        },
        {
            "Metode Klasifikasi": "Eksperimen Hybrid (SVM OR Rule)",
            "Akurasi": f"{or_m.get('accuracy', 0.761)*100:.2f}%",
            "Precision Macro": f"{or_m.get('precision_macro', 0.537)*100:.2f}%",
            "Recall Macro": f"{or_m.get('recall_macro', 0.575)*100:.2f}%",
            "F1 Macro": f"{or_m.get('f1_macro', 0.533)*100:.2f}%",
            "Brier Score Loss": "-",
            "Keterangan": "Menggabungkan kedua sinyal (recall tinggi, presisi turun)",
        },
    ]
    st.dataframe(pd.DataFrame(b_data), hide_index=True, use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col_cm, col_info = st.columns([1, 1.2])

    with col_cm:
        cm_matrix = svm_m.get("confusion_matrix", [[4166, 20], [380, 33]])
        st.plotly_chart(plot_confusion_matrix(cm_matrix), use_container_width=True)

    with col_info:
        st.markdown("<div class='formal-card-title'>Temuan Kritis Evaluasi Model</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="formal-card">
                <b>1. Keandalan Kalibrasi Probabilitas (Brier Score = {svm_m.get('brier_score_loss', 0.069):.3f}):</b><br/>
                Skor Brier di bawah 0.10 mengindikasikan bahwa probabilitas model merefleksikan peluang aktual secara reliabel, bukan sekadar prediksi biner tanpa dasar keyakinan.
            </div>
            <div class="formal-card">
                <b>2. Analisis Kasus Discrepancy ({svm_m.get('disagreements_count', 923)} Kasus):</b><br/>
                Terdapat 923 kasus di mana aturan kamus menemukan kata umpatan, namun model SVM mengklasifikasikannya sebagai non-toksik. Hal ini mengonfirmasi bahwa <b>aturan leksikon tidak dapat dijadikan sebagai override otomatis</b>, melainkan harus diarahkan ke antrean peninjauan manusia.
            </div>
            <div class="formal-card">
                <b>3. Akurasi Multilabel Subtopik (Micro F1 = {metrics.get('subtopics', {}).get('micro_f1', 0.88)*100:.1f}%):</b><br/>
                Model Random Forest multilabel berhasil mengidentifikasi 10 subtopik sensitif dengan <i>Hamming Loss</i> sebesar {metrics.get('subtopics', {}).get('hamming_loss', 0.027):.3f}.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 6: LEKSIKON & PENGETAHUAN REGIONAL
# ==============================================================================
def render_lexicon_tab(entries: list[dict[str, Any]]):
    st.markdown("<div class='formal-card-title'>Basis Pengetahuan Leksikon Umpatan Daerah</div>", unsafe_allow_html=True)

    if not entries:
        st.warning("Data leksikon tidak ditemukan.")
        return

    c_g, c_t = st.columns([1.3, 1])
    with c_g:
        st.plotly_chart(plot_regional_lexicon_distribution(entries), use_container_width=True)

    with c_t:
        st.markdown(
            f"""
            <div class="formal-card">
                <div class="formal-card-title">Audit Transparansi Aturan Leksikon</div>
                <p style="font-size: 13px; color: #CBD5E1; line-height: 1.6;">
                    • <b>Total Istilah Terdaftar:</b> {len(entries)} entri.<br/>
                    • <b>Mekanisme Pencocokan:</b> Menggunakan regex dengan batas kata (<i>word-boundary</i>):<br/>
                    <code>(?&lt;![a-z0-9])istilah(?![a-z0-9])</code><br/>
                    • <b>Tujuan:</b> Menghindari pencocokan keliru pada kata yang hanya mengandung substring istilah kasar tanpa konteks yang sesuai.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='formal-card-title'>Daftar Kata Kasar Berdasarkan Wilayah / Dialek</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        q_search = st.text_input("Pencarian Istilah:", placeholder="Ketik kata untuk mencari...")
    with sc2:
        reg_list = ["Semua Wilayah"] + sorted({e.get("region", "Nasional / Indonesia") for e in entries})
        s_reg = st.selectbox("Saring Berdasarkan Wilayah:", reg_list)

    f_entries = entries
    if q_search:
        f_entries = [e for e in f_entries if q_search.lower() in e.get("expression", "").lower()]
    if s_reg != "Semua Wilayah":
        f_entries = [e for e in f_entries if e.get("region") == s_reg]

    t_lex = pd.DataFrame(f_entries).rename(
        columns={"expression": "Istilah Kata / Frasa", "region": "Wilayah / Bahasa Daerah"}
    )
    st.dataframe(t_lex, use_container_width=True, height=320)


if __name__ == "__main__":
    main()
