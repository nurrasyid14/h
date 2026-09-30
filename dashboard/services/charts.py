"""Plotly chart and WordCloud generation service for IndoToxic Dashboard.
Provides formal, high-contrast, publication-grade data visualizations,
interactive topic sorting, and corpus word clouds.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# High Contrast Formal Palette
COLORS = {
    "primary": "#3B82F6",      # Sharp Blue
    "secondary": "#8B5CF6",    # Purple
    "toxic": "#DC2626",        # Sharp Crimson Red
    "safe": "#16A34A",         # Sharp Green
    "warning": "#D97706",      # Sharp Amber
    "info": "#0284C7",         # Sky Blue
    "dark_bg": "#0F172A",      # Slate 900
    "card_bg": "#1E293B",      # Slate 800
    "border": "#475569",       # Slate 600
    "text": "#FFFFFF",         # Pure White
    "muted": "#E2E8F0",        # Light Slate
}

# Standard Indonesian Stopwords
STOPWORDS_ID = {
    "yang", "dan", "di", "ini", "itu", "untuk", "dari", "dengan", "ke", "adalah",
    "saya", "kamu", "dia", "mereka", "kita", "kami", "anda", "tidak", "bisa", "ada",
    "pada", "oleh", "sudah", "akan", "juga", "atau", "karena", "hanya", "harus", "saat",
    "jika", "lagi", "lebih", "tapi", "tetapi", "kalau", "bukan", "jadi", "udah", "aja",
    "nya", "saja", "tp", "yg", "dgn", "bgt", "gak", "ga", "nggak", "yah", "kok", "sih",
    "deh", "lah", "dong", "dr", "krn", "sy", "lu", "gue", "gua", "nih", "tuh", "kan",
    "sama", "tentang", "banyak", "orang", "masih", "belum", "apa", "kenapa", "gimana"
}


def apply_custom_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply consistent formal high-contrast styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": COLORS["text"], "family": "Arial, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        height=height,
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        margin=dict(l=30, r=30, t=50 if title else 25, b=30),
        font=dict(color=COLORS["text"], size=12, family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=COLORS["text"]),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#334155",
            zeroline=False,
            color=COLORS["text"],
            tickfont=dict(color=COLORS["text"], size=11),
            titlefont=dict(color=COLORS["text"], size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#334155",
            zeroline=False,
            color=COLORS["text"],
            tickfont=dict(color=COLORS["text"], size=11),
            titlefont=dict(color=COLORS["text"], size=12),
        ),
    )
    return fig


def plot_toxicity_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot formal donut chart of toxicity prevalence."""
    counts = df["toxicity"].value_counts().reset_index()
    counts.columns = ["Status", "Jumlah"]
    counts["Label"] = counts["Status"].map({0: "Non-Toksik (0)", 1: "Toksik (1)"})
    colors = [COLORS["safe"], COLORS["toxic"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Label"],
                values=counts["Jumlah"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1.5)),
                textinfo="label+percent",
                textfont=dict(color="#FFFFFF", size=13),
                hoverinfo="label+value+percent",
            )
        ]
    )
    return apply_custom_layout(fig, "Distribusi Label Toksisitas (23.009 Sampel)")


def plot_subcategories_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot horizontal bar chart of the 5 toxicity subcategories."""
    cols = [
        ("identity_attack", "Serangan Identitas (SARA)"),
        ("insults", "Penghinaan / Ujaran Kebencian"),
        ("profanity_obscenity", "Kekasaran / Kata Tidak Pantas"),
        ("threat_incitement_to_violence", "Ancaman / Hasutan Kekerasan"),
        ("sexually_explicit", "Konten Eksplisit / Asusila"),
    ]
    data = []
    for col, name in cols:
        if col in df.columns:
            data.append({"Kategori": name, "Frekuensi": int(df[col].sum())})

    sub_df = pd.DataFrame(data).sort_values(by="Frekuensi", ascending=True)

    fig = px.bar(
        sub_df,
        x="Frekuensi",
        y="Kategori",
        orientation="h",
        color="Frekuensi",
        color_continuous_scale=["#1D4ED8", "#7C3AED", "#DC2626"],
        text="Frekuensi",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=12),
        cliponaxis=False,
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_custom_layout(fig, "Frekuensi 5 Subkategori Toksisitas")


def plot_sorted_topics_distribution(df: pd.DataFrame, sort_by: str = "volume_desc", top_n: int = 10) -> go.Figure:
    """Plot topics and their toxicity proportions with dynamic sorting."""
    topic_tox = (
        df.groupby(["topic", "toxicity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    topic_tox.columns = ["Topik", "Non_Toksik", "Toksik"]
    topic_tox["Total"] = topic_tox["Non_Toksik"] + topic_tox["Toksik"]
    topic_tox["Persen_Toksik"] = (topic_tox["Toksik"] / topic_tox["Total"]) * 100

    # Sorting options
    if sort_by == "volume_desc":
        topic_tox = topic_tox.sort_values(by="Total", ascending=False)
        sort_title = "Volume Tertinggi"
    elif sort_by == "volume_asc":
        topic_tox = topic_tox.sort_values(by="Total", ascending=True)
        sort_title = "Volume Terendah"
    elif sort_by == "toxic_rate_desc":
        topic_tox = topic_tox.sort_values(by="Persen_Toksik", ascending=False)
        sort_title = "Persentase Toksik Tertinggi"
    elif sort_by == "toxic_rate_asc":
        topic_tox = topic_tox.sort_values(by="Persen_Toksik", ascending=True)
        sort_title = "Persentase Toksik Terendah"
    elif sort_by == "name_asc":
        topic_tox = topic_tox.sort_values(by="Topik", ascending=True)
        sort_title = "Nama (A - Z)"
    elif sort_by == "name_desc":
        topic_tox = topic_tox.sort_values(by="Topik", ascending=False)
        sort_title = "Nama (Z - A)"
    else:
        topic_tox = topic_tox.sort_values(by="Total", ascending=False)
        sort_title = "Volume Tertinggi"

    display_topics = topic_tox.head(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Non-Toksik",
            x=display_topics["Topik"],
            y=display_topics["Non_Toksik"],
            marker_color=COLORS["safe"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Toksik",
            x=display_topics["Topik"],
            y=display_topics["Toksik"],
            marker_color=COLORS["toxic"],
            text=display_topics["Persen_Toksik"].apply(lambda v: f"{v:.1f}% toksik"),
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=10),
        )
    )
    fig.update_layout(barmode="stack")
    return apply_custom_layout(fig, f"Topik Berdasarkan {sort_title} (Top {top_n})")


def get_topic_summary_table(df: pd.DataFrame, sort_by: str = "volume_desc") -> pd.DataFrame:
    """Return formatted summary table for topics with dynamic sorting."""
    topic_tox = (
        df.groupby(["topic", "toxicity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    topic_tox.columns = ["Topik", "Non_Toksik", "Toksik"]
    topic_tox["Total_Komentar"] = topic_tox["Non_Toksik"] + topic_tox["Toksik"]
    topic_tox["Rasio_Toksik"] = (topic_tox["Toksik"] / topic_tox["Total_Komentar"]) * 100

    if sort_by == "volume_desc":
        topic_tox = topic_tox.sort_values(by="Total_Komentar", ascending=False)
    elif sort_by == "volume_asc":
        topic_tox = topic_tox.sort_values(by="Total_Komentar", ascending=True)
    elif sort_by == "toxic_rate_desc":
        topic_tox = topic_tox.sort_values(by="Rasio_Toksik", ascending=False)
    elif sort_by == "toxic_rate_asc":
        topic_tox = topic_tox.sort_values(by="Rasio_Toksik", ascending=True)
    elif sort_by == "name_asc":
        topic_tox = topic_tox.sort_values(by="Topik", ascending=True)
    elif sort_by == "name_desc":
        topic_tox = topic_tox.sort_values(by="Topik", ascending=False)

    topic_tox["Rasio_Toksik_Format"] = topic_tox["Rasio_Toksik"].apply(lambda v: f"{v:.2f}%")
    return topic_tox.rename(
        columns={
            "Topik": "Nama Topik",
            "Non_Toksik": "Jumlah Non-Toksik",
            "Toksik": "Jumlah Toksik",
            "Total_Komentar": "Total Komentar",
            "Rasio_Toksik_Format": "Tingkat Toksisitas (%)",
        }
    )[["Nama Topik", "Total Komentar", "Jumlah Toksik", "Jumlah Non-Toksik", "Tingkat Toksisitas (%)"]]


def generate_wordcloud_image(df: pd.DataFrame, category: str = "all", max_words: int = 100) -> Any:
    """Generate WordCloud PIL image for toxic, non-toxic, or full corpus."""
    if category == "toxic":
        subset = df[df["toxicity"] == 1]
        colormap = "Reds"
    elif category == "non_toxic":
        subset = df[df["toxicity"] == 0]
        colormap = "Greens"
    else:
        subset = df
        colormap = "Blues"

    texts = subset["text_clean"].dropna().astype(str).tolist()
    all_tokens = []
    for t in texts:
        for w in t.split():
            clean_w = w.strip().lower()
            if len(clean_w) > 2 and clean_w not in STOPWORDS_ID and clean_w.isalpha():
                all_tokens.append(clean_w)

    counts = Counter(all_tokens)
    if not counts:
        counts = {"korpus": 1, "kosong": 1}

    wc = WordCloud(
        width=850,
        height=380,
        background_color="#1E293B",
        colormap=colormap,
        max_words=max_words,
        random_state=42,
    ).generate_from_frequencies(counts)

    return wc.to_image()


def plot_top_words_bar(df: pd.DataFrame, category: str = "all", top_n: int = 20) -> go.Figure:
    """Plot interactive bar chart of top N most frequent words."""
    if category == "toxic":
        subset = df[df["toxicity"] == 1]
        title_cat = "Komentar Toksik"
        bar_color = COLORS["toxic"]
    elif category == "non_toxic":
        subset = df[df["toxicity"] == 0]
        title_cat = "Komentar Non-Toksik"
        bar_color = COLORS["safe"]
    else:
        subset = df
        title_cat = "Keseluruhan Korpus"
        bar_color = COLORS["primary"]

    texts = subset["text_clean"].dropna().astype(str).tolist()
    all_tokens = []
    for t in texts:
        for w in t.split():
            clean_w = w.strip().lower()
            if len(clean_w) > 2 and clean_w not in STOPWORDS_ID and clean_w.isalpha():
                all_tokens.append(clean_w)

    counts = Counter(all_tokens).most_common(top_n)
    if not counts:
        counts = [("kosong", 1)]

    word_df = pd.DataFrame(counts, columns=["Kata", "Frekuensi"]).sort_values(by="Frekuensi", ascending=True)

    fig = px.bar(
        word_df,
        x="Frekuensi",
        y="Kata",
        orientation="h",
        color="Frekuensi",
        color_continuous_scale="Viridis",
        text="Frekuensi",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
        cliponaxis=False,
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_custom_layout(fig, f"Top {top_n} Kata Paling Populer ({title_cat})", height=420)


def plot_token_length_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot comment token length distribution for toxic vs non-toxic."""
    sample = df.sample(min(4000, len(df)), random_state=42).copy()
    sample["token_count"] = sample["text_clean"].fillna("").astype(str).str.split().str.len()
    sample["Kategori"] = sample["toxicity"].map({0: "Non-Toksik", 1: "Toksik"})

    fig = px.histogram(
        sample,
        x="token_count",
        color="Kategori",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"Non-Toksik": COLORS["safe"], "Toksik": COLORS["toxic"]},
        opacity=0.75,
    )
    fig.update_xaxes(range=[0, 70], title="Jumlah Kata (Token)", titlefont=dict(color="#FFFFFF"))
    fig.update_yaxes(title="Frekuensi", titlefont=dict(color="#FFFFFF"))
    return apply_custom_layout(fig, "Distribusi Panjang Komentar (Token/Kata)")


def plot_confusion_matrix(cm: list[list[int]]) -> go.Figure:
    """Plot formal confusion matrix heatmap."""
    labels = ["Non-Toksik (0)", "Toksik (1)"]
    z = cm
    annotations = []
    names = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{names[i][j]}</b><br>{z[i][j]:,} sampel",
                    showarrow=False,
                    font=dict(size=14, color="#FFFFFF"),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, "#0F172A"], [0.5, "#1E40AF"], [1, "#3B82F6"]],
            showscale=False,
        )
    )
    fig.update_layout(
        annotations=annotations,
        xaxis_title="Prediksi Model (Predicted)",
        yaxis_title="Label Aktual (Ground Truth)",
        yaxis=dict(autorange="reversed"),
    )
    return apply_custom_layout(fig, "Confusion Matrix pada Data Uji (Held-Out Test)", height=340)


def plot_confidence_gauge(confidence: float, is_toxic: bool) -> go.Figure:
    """Formal gauge for calibrated prediction confidence."""
    bar_color = COLORS["toxic"] if is_toxic else COLORS["safe"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FFFFFF", "tickfont": {"color": "#FFFFFF", "size": 11}},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "#0F172A",
                "borderwidth": 1,
                "bordercolor": "#475569",
                "steps": [
                    {"range": [0, 50], "color": "#1E293B"},
                    {"range": [50, 75], "color": "#334155"},
                    {"range": [75, 100], "color": "#475569"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 3},
                    "thickness": 0.8,
                    "value": 85,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=25, r=25, t=30, b=10),
        paper_bgcolor="#1E293B",
        font=dict(color="#FFFFFF", family="Arial, sans-serif"),
    )
    return fig


def plot_toxicity_meter_bar(p_toxic: float) -> go.Figure:
    """Horizontal stacked toxicity meter indicator."""
    p_percent = p_toxic * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_percent,
            number={"suffix": "%", "font": {"size": 32, "color": "#FFFFFF"}},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100], "tickcolor": "#FFFFFF", "tickfont": {"color": "#FFFFFF"}},
                "bar": {"color": "#DC2626" if p_percent >= 50 else "#16A34A", "thickness": 0.5},
                "bgcolor": "#0F172A",
                "steps": [
                    {"range": [0, 25], "color": "#166534"},
                    {"range": [25, 50], "color": "#1E293B"},
                    {"range": [50, 75], "color": "#9A3412"},
                    {"range": [75, 100], "color": "#7F1D1D"},
                ],
            },
        )
    )
    fig.update_layout(
        height=130,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1E293B",
        font=dict(color="#FFFFFF", family="Arial, sans-serif"),
    )
    return fig


def plot_subtopics_bar(subtopics: list[dict[str, Any]]) -> go.Figure:
    """Bar chart for predicted subtopic probabilities."""
    df_sub = pd.DataFrame(subtopics).head(8).sort_values(by="probability", ascending=True)
    df_sub["Persen"] = df_sub["probability"] * 100

    fig = px.bar(
        df_sub,
        x="Persen",
        y="topic",
        orientation="h",
        color="Persen",
        color_continuous_scale=["#1E293B", "#3B82F6", "#8B5CF6"],
        text=df_sub["Persen"].apply(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
        cliponaxis=False,
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    fig.update_xaxes(range=[0, 105], title="Probabilitas (%)", titlefont=dict(color="#FFFFFF"))
    return apply_custom_layout(fig, "Prediksi Multilabel Subtopik (TriModel RF)", height=300)


def plot_regional_lexicon_distribution(entries: list[dict[str, Any]]) -> go.Figure:
    """Bar chart of regional profanity expressions."""
    regions_count = {}
    for entry in entries:
        r = entry.get("region", "Nasional / Indonesia")
        regions_count[r] = regions_count.get(r, 0) + 1

    reg_df = (
        pd.DataFrame(list(regions_count.items()), columns=["Wilayah / Bahasa", "Jumlah Kata"])
        .sort_values(by="Jumlah Kata", ascending=True)
    )

    fig = px.bar(
        reg_df,
        x="Jumlah Kata",
        y="Wilayah / Bahasa",
        orientation="h",
        color="Jumlah Kata",
        color_continuous_scale="Blues",
        text="Jumlah Kata",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=12),
        cliponaxis=False,
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_custom_layout(fig, "Distribusi Leksikon Umpatan Berdasarkan Wilayah", height=320)
