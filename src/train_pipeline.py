"""Complete training script for IndoToxic 2024 classifier suite.
Fits vectorizer, TriModel RF, out-of-fold proto-features, rule features,
calibrated ToxicOrNot LinearSVC, computes held-out evaluation metrics,
and saves all artifacts to the models/ directory.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.classif.Toxic_or_not import ToxicOrNot
from src.classif.mood_rf import MoodRF
from src.classif.sentiment_rf import SentimentRF
from src.classif.subtopic_rf import SubtopicRF
from src.trimodel_rf import TriModelRF
from src.utils import ensure_directory, project_root, save_pickle


def parse_topic_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return []


def make_proto_features(prediction, mood_classes, sentiment_classes):
    mood_features = np.column_stack([
        prediction["mood"] == label for label in mood_classes
    ])
    sentiment_features = np.column_stack([
        prediction["sentiment"] == label for label in sentiment_classes
    ])
    return np.column_stack((mood_features, sentiment_features, prediction["subtopic"]))


def make_rule_features(texts, classifier):
    rule_results = [classifier.classify(text) for text in texts]
    return np.column_stack((
        [int(result["toxic"]) for result in rule_results],
        [len(result["matches"]) for result in rule_results],
    ))


def run_training():
    start_time = time.time()
    root = project_root()
    data_path = root / "data" / "processed" / "data_cleaned_before_encode.csv"
    models_dir = root / "models"
    ensure_directory(models_dir)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    df["topic_list_parsed"] = df["topic_list"].apply(parse_topic_list)
    all_topics = sorted({topic for topics in df["topic_list_parsed"] for topic in topics})
    print(f"Detected {len(all_topics)} subtopics: {all_topics}")

    with (models_dir / "all_topics.json").open("w", encoding="utf-8") as f:
        json.dump(all_topics, f, indent=2)

    X_text = df["text_clean"].fillna("").astype(str)
    y_toxic = df["toxicity"].fillna(0).astype(int)

    # Proxy targets
    mood_labels = np.where(y_toxic == 1, "angry", "happy")
    sentiment_labels = np.where(y_toxic == 1, "negative", "positive")

    # Subtopic multi-label matrix
    subtopic_matrix = np.array(
        [[1 if topic in topics else 0 for topic in all_topics] for topics in df["topic_list_parsed"]],
        dtype=int,
    )

    # 80/20 train/test split with StratifiedGroupKFold to prevent leakage
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X_text, y_toxic, groups=X_text))

    X_train_text = X_text.iloc[train_idx]
    X_test_text = X_text.iloc[test_idx]
    ytox_train, ytox_test = y_toxic.iloc[train_idx], y_toxic.iloc[test_idx]
    mood_train, mood_test = mood_labels[train_idx], mood_labels[test_idx]
    sent_train, sent_test = sentiment_labels[train_idx], sentiment_labels[test_idx]
    sub_train, sub_test = subtopic_matrix[train_idx], subtopic_matrix[test_idx]

    print(f"Train samples: {len(X_train_text)}, Test samples: {len(X_test_text)}")

    # TF-IDF feature extraction
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Individual classifiers
    print("Training individual TriModel heads...")
    mood_model = MoodRF(n_estimators=60, random_state=42, n_jobs=-1)
    sentiment_model = SentimentRF(n_estimators=60, random_state=42, n_jobs=-1)
    subtopic_model = SubtopicRF(n_estimators=60, random_state=42, n_jobs=-1)

    mood_model.fit(X_train, mood_train)
    sentiment_model.fit(X_train, sent_train)
    subtopic_model.fit(X_train, sub_train, labels=all_topics)

    # Orchestrated TriModel
    trimodel = TriModelRF(
        mood_kwargs={"n_estimators": 60, "n_jobs": -1},
        sentiment_kwargs={"n_estimators": 60, "n_jobs": -1},
        subtopic_kwargs={"n_estimators": 60, "n_jobs": -1},
        random_state=42,
    )
    trimodel.fit(X_train, mood_train, sent_train, sub_train, labels=all_topics)

    # Out-of-fold proto-features for toxic classifier
    print("Generating out-of-fold proto-features...")
    proto_width = len(np.unique(mood_train)) + len(np.unique(sent_train)) + sub_train.shape[1]
    train_proto_features = np.zeros((len(X_train_text), proto_width), dtype=float)
    proto_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=123)

    for fold_num, (f_train, f_val) in enumerate(proto_splitter.split(X_train_text, ytox_train, groups=X_train_text)):
        fold_trimodel = TriModelRF(
            mood_kwargs={"n_estimators": 40, "n_jobs": -1},
            sentiment_kwargs={"n_estimators": 40, "n_jobs": -1},
            subtopic_kwargs={"n_estimators": 40, "n_jobs": -1},
            random_state=42 + fold_num,
        )
        fold_trimodel.fit(
            X_train[f_train],
            mood_train[f_train],
            sent_train[f_train],
            sub_train[f_train],
            labels=all_topics,
        )
        fold_pred = fold_trimodel.predict(X_train[f_val])
        train_proto_features[f_val] = make_proto_features(
            fold_pred,
            fold_trimodel.mood_model.classes_,
            fold_trimodel.sentiment_model.classes_,
        )

    # ToxicOrNot classifier
    print("Fitting calibrated ToxicOrNot SVM with engineered features...")
    toxicity_model = ToxicOrNot(random_state=42)
    train_rule_features = make_rule_features(X_train_text, toxicity_model.rule_classifier)
    toxicity_train_engineered = np.column_stack((train_proto_features, train_rule_features))

    toxicity_model.fit(
        X_train_text,
        ytox_train,
        engineered_features=toxicity_train_engineered,
    )

    # Evaluation on held-out test partition
    print("Evaluating held-out test predictions...")
    test_tri_prediction = trimodel.predict(X_test)
    test_proto_features = make_proto_features(
        test_tri_prediction,
        trimodel.mood_model.classes_,
        trimodel.sentiment_model.classes_,
    )
    test_rule_features = make_rule_features(X_test_text, toxicity_model.rule_classifier)
    toxicity_test_engineered = np.column_stack((test_proto_features, test_rule_features))

    toxicity_test_preds = toxicity_model.predict(
        X_test_text,
        engineered_features=toxicity_test_engineered,
    )

    mood_test_preds = mood_model.predict(X_test)
    sent_test_preds = sentiment_model.predict(X_test)
    sub_test_preds = subtopic_model.predict(X_test)

    # Metrics computation
    pos_idx = list(toxicity_model.svm.classes_).index(1)
    pos_conf = toxicity_test_preds["class_probabilities"][:, pos_idx]

    cm = confusion_matrix(ytox_test, toxicity_test_preds["final_prediction"]).tolist()

    eval_metrics = {
        "dataset_rows": len(df),
        "train_rows": len(X_train_text),
        "test_rows": len(X_test_text),
        "topics_count": len(all_topics),
        "toxicity_prevalence": float(y_toxic.mean()),
        "toxicity_svm": {
            "accuracy": float(accuracy_score(ytox_test, toxicity_test_preds["final_prediction"])),
            "precision_macro": float(precision_score(ytox_test, toxicity_test_preds["final_prediction"], average="macro", zero_division=0)),
            "recall_macro": float(recall_score(ytox_test, toxicity_test_preds["final_prediction"], average="macro", zero_division=0)),
            "f1_macro": float(f1_score(ytox_test, toxicity_test_preds["final_prediction"], average="macro", zero_division=0)),
            "precision_toxic": float(precision_score(ytox_test, toxicity_test_preds["final_prediction"], pos_label=1, zero_division=0)),
            "recall_toxic": float(recall_score(ytox_test, toxicity_test_preds["final_prediction"], pos_label=1, zero_division=0)),
            "f1_toxic": float(f1_score(ytox_test, toxicity_test_preds["final_prediction"], pos_label=1, zero_division=0)),
            "brier_score_loss": float(brier_score_loss(ytox_test, pos_conf)),
            "confusion_matrix": cm,
            "disagreements_count": int(toxicity_test_preds["rule_review"].sum()),
        },
        "toxicity_rule_only": {
            "accuracy": float(accuracy_score(ytox_test, toxicity_test_preds["rule_prediction"])),
            "precision_macro": float(precision_score(ytox_test, toxicity_test_preds["rule_prediction"], average="macro", zero_division=0)),
            "recall_macro": float(recall_score(ytox_test, toxicity_test_preds["rule_prediction"], average="macro", zero_division=0)),
            "f1_macro": float(f1_score(ytox_test, toxicity_test_preds["rule_prediction"], average="macro", zero_division=0)),
        },
        "toxicity_or_experiment": {
            "accuracy": float(accuracy_score(ytox_test, toxicity_test_preds["or_experimental_prediction"])),
            "precision_macro": float(precision_score(ytox_test, toxicity_test_preds["or_experimental_prediction"], average="macro", zero_division=0)),
            "recall_macro": float(recall_score(ytox_test, toxicity_test_preds["or_experimental_prediction"], average="macro", zero_division=0)),
            "f1_macro": float(f1_score(ytox_test, toxicity_test_preds["or_experimental_prediction"], average="macro", zero_division=0)),
        },
        "mood_proxy": {
            "accuracy": float(accuracy_score(mood_test, mood_test_preds)),
            "f1_macro": float(f1_score(mood_test, mood_test_preds, average="macro", zero_division=0)),
        },
        "sentiment_proxy": {
            "accuracy": float(accuracy_score(sent_test, sent_test_preds)),
            "f1_macro": float(f1_score(sent_test, sent_test_preds, average="macro", zero_division=0)),
        },
        "subtopics": {
            "exact_match_accuracy": float(accuracy_score(sub_test, sub_test_preds)),
            "micro_f1": float(f1_score(sub_test, sub_test_preds, average="micro", zero_division=0)),
            "hamming_loss": float(hamming_loss(sub_test, sub_test_preds)),
        },
    }

    # Save models
    print("Saving model artifacts...")
    artifacts = {
        "mood_model.pkl": mood_model,
        "sentiment_model.pkl": sentiment_model,
        "subtopic_model.pkl": subtopic_model,
        "trimodel_model.pkl": trimodel,
        "vectorizer_model.pkl": vectorizer,
        "toxic_or_not_model.pkl": toxicity_model,
    }
    for filename, model_obj in artifacts.items():
        save_pickle(model_obj, models_dir / filename)
        print(f"Saved: {models_dir / filename}")

    with (models_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Saved metrics: {models_dir / 'evaluation_metrics.json'}")

    elapsed = time.time() - start_time
    print(f"All models trained and saved successfully in {elapsed:.1f} seconds!")
    return eval_metrics


if __name__ == "__main__":
    run_training()
