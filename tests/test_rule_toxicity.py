from src.classif.rule_toxicity import RuleToxicityClassifier


def test_rule_toxicity_detects_words_and_phrases():
    classifier = RuleToxicityClassifier()

    word_result = classifier.classify("Kamu ANJING!")
    phrase_result = classifier.classify("Kamu sok tahu.")

    assert word_result["label"] == "toxic"
    assert "anjing" in [match["expression"] for match in word_result["matches"]]
    assert phrase_result["label"] == "toxic"
    assert "sok tahu" in [match["expression"] for match in phrase_result["matches"]]


def test_rule_toxicity_uses_expression_boundaries_and_handles_empty_text():
    classifier = RuleToxicityClassifier()

    assert classifier.classify("anjingnya lucu")["label"] == "non_toxic"
    assert classifier.classify(None) == {
        "toxic": False,
        "label": "non_toxic",
        "matches": [],
    }