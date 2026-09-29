import numpy as np

from src.classif.Toxic_or_not import ToxicOrNot


def test_toxic_or_not_returns_svm_rule_and_combined_predictions():
    texts = [
        "kebijakan ini membantu warga",
        "layanan publik semakin baik",
        "goblok sekali kamu",
        "dasar brengsek",
        "saya mendukung program pemerintah",
        "bajingan tidak tahu diri",
    ]
    labels = [0, 0, 1, 1, 0, 1]
    classifier = ToxicOrNot(max_features=100, min_df=1)
    classifier.fit(texts, labels)

    result = classifier.predict(["goblok", "layanan publik"])

    assert result["svm_prediction"].shape == (2,)
    assert result["rule_prediction"].tolist() == [1, 0]
    assert result["final_prediction"].shape == (2,)
    assert np.array_equal(result["final_prediction"], result["svm_prediction"])
    assert np.array_equal(
        result["or_experimental_prediction"],
        np.maximum(result["svm_prediction"], result["rule_prediction"]),
    )
    assert "goblok" in [match["expression"] for match in result["rule_matches"][0]]


def test_toxic_or_not_requires_both_classes_and_fit_before_predict():
    classifier = ToxicOrNot(max_features=20, min_df=1)

    try:
        classifier.predict(["test"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("prediction before fit should fail")

    try:
        classifier.fit(["clean text", "another clean text"], [0, 0])
    except ValueError as error:
        assert "both toxicity classes" in str(error)
    else:
        raise AssertionError("single-class training should fail")


def test_toxic_or_not_accepts_proto_features_and_returns_calibrated_confidence():
    texts = [
        "clear public policy",
        "helpful public service",
        "goblok sekali kamu",
        "dasar brengsek",
    ] * 3
    labels = [0, 0, 1, 1] * 3
    proto_features = np.array([[0, 1], [0, 1], [1, 0], [1, 0]] * 3, dtype=float)
    classifier = ToxicOrNot(max_features=100, min_df=1, calibration_cv=2)
    classifier.fit(texts, labels, engineered_features=proto_features)

    result = classifier.predict(
        ["ordinary policy", "goblok"],
        engineered_features=np.array([[0, 1], [1, 0]], dtype=float),
    )

    assert result["class_probabilities"].shape == (2, 2)
    assert result["confidence"].shape == (2,)
    assert np.all((result["confidence"] >= 0) & (result["confidence"] <= 1))
    assert np.allclose(
        result["confidence"],
        result["class_probabilities"][np.arange(2), result["final_prediction"]],
    )