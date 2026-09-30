"""Inference service for the IndoToxic Streamlit Dashboard.
Handles model loading, single-text analysis, text highlighting,
toxicity meter calculations, and batch inference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.classif.gambling_ad_detector import GamblingAdDetector
from src.classif.rule_toxicity import RuleToxicityClassifier
from src.classif.toxicity_meter import ToxicityMeter
from src.preprocessing import clean_text
from src.utils import load_pickle, project_root

# Vocabulary of known positive/constructive terms in Indonesian
POSITIVE_TERMS = {
    "membantu", "terima kasih", "bagus", "bermanfaat", "baik", "ramah",
    "damai", "sehat", "sukses", "maju", "positif", "hebat", "senang",
    "bahagia", "suka", "setuju", "keren", "berkualitas", "peduli",
    "mendukung", "menghargai", "mengapresiasi", "terpuji", "adil"
}

# Vocabulary of offensive/abusive terms
OFFENSIVE_TERMS = {
    "benci", "marah", "menghina", "sakit", "hancurkan", "serang", "bunuh",
    "mencela", "menyerang", "ancam", "bajingan", "brengsek", "kontol", "jawir",
    "goblok", "tolol", "anjing", "babi", "bangsat", "kampret", "perusak",
    "jancuk", "jancok", "pantek", "bodoh", "setan", "iblis", "pelacur",
    "lonte", "biadab", "hina", "kafir", "najis", "keparat", "mampus"
}


class IndoToxicInferenceService:
    """Manages loaded IndoToxic models and inference pipelines."""

    def __init__(self, models_dir: Path | None = None):
        if models_dir is None:
            models_dir = project_root() / "models"
        self.models_dir = Path(models_dir)
        self.rule_classifier = RuleToxicityClassifier()
        self.gambling_detector = GamblingAdDetector()
        self.toxicity_meter = ToxicityMeter()

        self.vectorizer = None
        self.trimodel = None
        self.toxic_model = None
        self.all_topics: list[str] = []
        self.eval_metrics: dict[str, Any] = {}
        self.is_loaded = False
        self._load_models()

    def _load_models(self):
        try:
            vec_path = self.models_dir / "vectorizer_model.pkl"
            tri_path = self.models_dir / "trimodel_model.pkl"
            tox_path = self.models_dir / "toxic_or_not_model.pkl"
            topics_path = self.models_dir / "all_topics.json"
            metrics_path = self.models_dir / "evaluation_metrics.json"

            if vec_path.exists() and tri_path.exists() and tox_path.exists():
                self.vectorizer = load_pickle(vec_path)
                self.trimodel = load_pickle(tri_path)
                self.toxic_model = load_pickle(tox_path)
                if topics_path.exists():
                    with topics_path.open("r", encoding="utf-8") as f:
                        self.all_topics = json.load(f)
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8") as f:
                        self.eval_metrics = json.load(f)
                self.is_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load serialized models: {e}")
            self.is_loaded = False

    def _make_proto_features(self, trimodel_pred: dict[str, Any]) -> np.ndarray:
        mood_classes = self.trimodel.mood_model.classes_
        sentiment_classes = self.trimodel.sentiment_model.classes_

        mood_feat = np.column_stack([
            trimodel_pred["mood"] == label for label in mood_classes
        ])
        sentiment_feat = np.column_stack([
            trimodel_pred["sentiment"] == label for label in sentiment_classes
        ])
        return np.column_stack((mood_feat, sentiment_feat, trimodel_pred["subtopic"]))

    def _make_rule_features(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rule_results = [self.rule_classifier.classify(text) for text in texts]
        rule_features = np.column_stack((
            [int(res["toxic"]) for res in rule_results],
            [len(res["matches"]) for res in rule_results],
        ))
        return rule_features, rule_results

    def get_vocabulary_coverage(self, cleaned_text: str) -> float:
        if not self.vectorizer or not hasattr(self.vectorizer, "vocabulary_"):
            return 1.0
        analyzer = self.vectorizer.build_analyzer()
        tokens = analyzer(cleaned_text)
        if not tokens:
            return 0.0
        vocab = self.vectorizer.vocabulary_
        known = sum(1 for tok in tokens if tok in vocab)
        return known / len(tokens)

    def highlight_text_tokens(self, raw_text: str, rule_matches: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Highlight words/phrases categorized as toxic (red) or positive (green)."""
        if not raw_text or not raw_text.strip():
            return "", []

        # Collect all toxic phrases & words
        lexicon_phrases = {m["expression"].strip().lower() for m in rule_matches if m.get("expression")}
        all_toxic = lexicon_phrases.union(OFFENSIVE_TERMS)
        all_positive = set(POSITIVE_TERMS)

        term_map = {}
        for term in sorted(all_toxic, key=len, reverse=True):
            matched_rule = next((m for m in rule_matches if m["expression"].lower() == term), None)
            region_info = ", ".join(matched_rule["regions"]) if (matched_rule and matched_rule.get("regions")) else "Kamus Umpatan Nasional"
            term_map[term.lower()] = {
                "category": "toxic",
                "label": "Toksik",
                "source": f"Leksikon Umpatan ({region_info})",
                "badge": "badge-toxic",
            }
        for term in sorted(all_positive, key=len, reverse=True):
            if term.lower() not in term_map:
                term_map[term.lower()] = {
                    "category": "positive",
                    "label": "Positif / Konstruktif",
                    "source": "Leksikon Sentimen Positif",
                    "badge": "badge-safe",
                }

        all_sorted_terms = sorted(term_map.keys(), key=len, reverse=True)
        if not all_sorted_terms:
            return raw_text, []

        pattern = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in all_sorted_terms) + r")\b")
        detected_tokens: list[dict[str, Any]] = []

        def replacer(match: re.Match) -> str:
            val = match.group(0)
            low = val.lower()
            info = term_map.get(low)
            if not info:
                return val

            if not any(d["token"].lower() == low for d in detected_tokens):
                detected_tokens.append({
                    "token": val,
                    "kategori": info["label"],
                    "sumber": info["source"],
                    "badge": info["badge"],
                })

            if info["category"] == "toxic":
                return f'<mark style="background-color: #DC2626; color: #FFFFFF; padding: 2px 6px; border-radius: 4px; font-weight: 800; border: 1px solid #B91C1C;">{val}</mark>'
            else:
                return f'<mark style="background-color: #166534; color: #DCFCE7; padding: 2px 6px; border-radius: 4px; font-weight: 800; border: 1px solid #15803D;">{val}</mark>'

        highlighted = pattern.sub(replacer, raw_text)
        return highlighted, detected_tokens

    def analyze_single_text(self, text: str) -> dict[str, Any]:
        """Perform full multi-stage AI analysis, toxicity meter calculation, and token highlighting."""
        raw_text = str(text or "")
        cleaned = clean_text(raw_text)

        rule_res = self.rule_classifier.classify(raw_text)
        gambling_res = self.gambling_detector.classify(raw_text)
        meter_score = self.toxicity_meter.score(raw_text)

        if not self.is_loaded or not cleaned:
            is_toxic = rule_res["toxic"] or (meter_score >= 0.2)
            p_toxic = 0.85 if is_toxic else 0.10
            highlighted_html, detected_tokens = self.highlight_text_tokens(raw_text, rule_res["matches"])

            severity_label = "Sangat Tinggi (Toksisitas Berat)" if p_toxic >= 0.75 else ("Sedang (Toksik)" if p_toxic >= 0.50 else "Sangat Rendah (Aman)")

            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "highlighted_html": highlighted_html,
                "detected_tokens": detected_tokens,
                "is_toxic": is_toxic,
                "toxicity_label": "Toksik" if is_toxic else "Non-Toksik",
                "confidence": 0.85 if is_toxic else 0.90,
                "toxicity_percent": round(p_toxic * 100, 1),
                "severity_label": severity_label,
                "class_probabilities": {"non_toxic": 1.0 - p_toxic, "toxic": p_toxic},
                "mood": "angry" if is_toxic else "happy",
                "mood_proba": {"angry": 0.8 if is_toxic else 0.2, "happy": 0.2 if is_toxic else 0.8},
                "sentiment": "negative" if is_toxic else "positive",
                "sentiment_proba": {"negative": 0.8 if is_toxic else 0.2, "positive": 0.2 if is_toxic else 0.8},
                "subtopics": [],
                "rule_toxic": rule_res["toxic"],
                "rule_matches": rule_res["matches"],
                "needs_review": False,
                "gambling_promo": gambling_res["suspected_gambling_promo"],
                "gambling_matches": gambling_res["matches"],
                "vocabulary_coverage": 1.0,
                "meter_score": meter_score,
            }

        # 1. Feature extraction
        text_features = self.vectorizer.transform([cleaned])

        # 2. TriModel predictions
        tri_pred = self.trimodel.transform(text_features)
        mood = str(tri_pred["mood"][0])
        mood_probs = dict(zip(self.trimodel.mood_model.classes_, tri_pred["mood_proba"][0]))
        sentiment = str(tri_pred["sentiment"][0])
        sentiment_probs = dict(zip(self.trimodel.sentiment_model.classes_, tri_pred["sentiment_proba"][0]))

        subtopics = []
        labels = tri_pred["subtopic_labels"]
        probs = tri_pred["subtopic_proba"][0]
        for label, prob in zip(labels, probs):
            subtopics.append({"topic": str(label), "probability": float(prob), "active": bool(prob >= 0.5)})
        subtopics = sorted(subtopics, key=lambda x: x["probability"], reverse=True)

        # 3. Proto and Rule features
        proto_features = self._make_proto_features(tri_pred)
        rule_features, _ = self._make_rule_features([raw_text])
        engineered = np.column_stack((proto_features, rule_features))

        # 4. Calibrated SVM Prediction
        svm_res = self.toxic_model.predict([raw_text], engineered_features=engineered)
        final_class = int(svm_res["final_prediction"][0])
        confidence = float(svm_res["confidence"][0])
        class_probs = svm_res["class_probabilities"][0]

        p_toxic = float(class_probs[1])

        rule_hit = bool(rule_res["toxic"])
        svm_toxic = bool(final_class == 1)
        needs_review = bool(rule_hit and not svm_toxic)

        # Highlighting logic
        highlighted_html, detected_tokens = self.highlight_text_tokens(raw_text, rule_res["matches"])

        # Determine severity level
        p_percent = round(p_toxic * 100, 1)
        if p_percent <= 25.0:
            severity_label = "Sangat Rendah (Aman & Konstruktif)"
        elif p_percent <= 50.0:
            severity_label = "Rendah (Informal / Wajar)"
        elif p_percent <= 75.0:
            severity_label = "Sedang (Toksik / Ujaran Kebencian)"
        else:
            severity_label = "Sangat Tinggi (Toksisitas Berat / SARA / Ancaman)"

        vocab_cov = self.get_vocabulary_coverage(cleaned)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "highlighted_html": highlighted_html,
            "detected_tokens": detected_tokens,
            "is_toxic": svm_toxic,
            "toxicity_label": "Toksik" if svm_toxic else "Non-Toksik",
            "confidence": confidence,
            "toxicity_percent": p_percent,
            "severity_label": severity_label,
            "class_probabilities": {
                "non_toxic": float(class_probs[0]),
                "toxic": float(class_probs[1]),
            },
            "mood": mood,
            "mood_proba": {k: float(v) for k, v in mood_probs.items()},
            "sentiment": sentiment,
            "sentiment_proba": {k: float(v) for k, v in sentiment_probs.items()},
            "subtopics": subtopics,
            "rule_toxic": rule_hit,
            "rule_matches": rule_res["matches"],
            "needs_review": needs_review,
            "gambling_promo": gambling_res["suspected_gambling_promo"],
            "gambling_matches": gambling_res["matches"],
            "vocabulary_coverage": vocab_cov,
            "meter_score": meter_score,
        }

    def analyze_batch(self, texts: list[str]) -> pd.DataFrame:
        """Run batch analysis for a list of texts and return a structured DataFrame."""
        records = []
        for text in texts:
            res = self.analyze_single_text(text)
            top_topics = [s["topic"] for s in res["subtopics"] if s["active"]][:3]
            if not top_topics and res["subtopics"]:
                top_topics = [f"{res['subtopics'][0]['topic']} ({res['subtopics'][0]['probability']:.2f})"]

            matched_words = [m["expression"] for m in res["rule_matches"]]
            matched_regions = sorted({r for m in res["rule_matches"] for r in m["regions"]})
            gambling_notes = [f"{m['term']} {m['number']}" for m in res["gambling_matches"]]

            records.append({
                "Teks Asli": res["raw_text"],
                "Prediksi": res["toxicity_label"],
                "Toxicity Meter (%)": res["toxicity_percent"],
                "Tingkat Keparahan": res["severity_label"],
                "Keyakinan (Confidence)": round(res["confidence"], 3),
                "Mood": res["mood"],
                "Sentimen": res["sentiment"],
                "Topik Relevan": ", ".join(top_topics) if top_topics else "-",
                "Leksikon Umpatan": "Ya" if res["rule_toxic"] else "Tidak",
                "Kata Terdeteksi": ", ".join(matched_words) if matched_words else "-",
                "Asal Daerah": ", ".join(matched_regions) if matched_regions else "-",
                "Status Review": "Perlu Review" if res["needs_review"] else "Aman",
                "Iklan Judi": "Terindikasi" if res["gambling_promo"] else "Bukan",
                "Pola Judi": ", ".join(gambling_notes) if gambling_notes else "-",
            })
        return pd.DataFrame(records)


__all__ = ["IndoToxicInferenceService"]
