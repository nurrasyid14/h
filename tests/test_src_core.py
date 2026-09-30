import pandas as pd

from src.classif.toxicity_meter import ToxicityMeter
from src.classif import GamblingAdDetector
from src.predict import predict_single_text
from src.preprocessing import clean_text, normalize_text
from src.trimodel_rf import TriModelRF
from src.utils import splitter


def test_clean_and_normalize_text_helpers():
    text = "  Halo!!!  saya   sangat marah  "
    cleaned = clean_text(text)
    normalized = normalize_text(text)

    assert isinstance(cleaned, str)
    assert isinstance(normalized, str)
    assert "halo" in cleaned.lower()
    assert "saya" in normalized.lower()


def test_trimodel_rf_train_and_predict():
    X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 1.0]]
    y_mood = ["happy", "angry", "happy", "angry"]
    y_sentiment = ["positive", "negative", "positive", "negative"]
    y_subtopic = [[1, 0], [0, 1], [1, 0], [0, 1]]

    model = TriModelRF(random_state=42)
    model.fit(X, y_mood, y_sentiment, y_subtopic, labels=["topic_a", "topic_b"])
    result = model.predict(X)

    assert set(result.keys()) == {"mood", "sentiment", "subtopic", "subtopic_labels"}
    assert len(result["mood"]) == len(X)
    assert len(result["sentiment"]) == len(X)
    assert result["subtopic"].shape[1] == 2


def test_split_8_2_ratio():
    X = list(range(10))
    y = list(range(10, 20))

    X_train, X_test, y_train, y_test = splitter(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_toxicity_meter_and_predict_text():
    meter = ToxicityMeter()
    score = meter.score("aku sangat marah dan ingin menyerang orang")
    assert 0.0 <= score <= 1.0

    prediction = predict_single_text("aku benci kamu dan ingin menghancurkan semuanya")
    assert set(prediction.keys()) >= {"mood", "sentiment", "subtopic", "toxicity", "gambling_ad"}


def test_gambling_detector_is_exported_and_used_in_prediction():
    assert GamblingAdDetector().classify("Menang88")["suspected_gambling_promo"] is True

    promotion_prediction = predict_single_text("Menang88")
    empty_prediction = predict_single_text("")

    assert promotion_prediction["gambling_ad"]["suspected_gambling_promo"] is True
    assert empty_prediction["gambling_ad"]["suspected_gambling_promo"] is False
