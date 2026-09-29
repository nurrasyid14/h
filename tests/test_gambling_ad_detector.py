import json

from src.classif.gambling_ad_detector import GamblingAdDetector


def test_gambling_detector_matches_examples():
    detector = GamblingAdDetector()

    for text in ("Menang88", "Suka138", "Juara 77", "Singapore88", "Sidney49"):
        result = detector.classify(text)
        assert result["suspected_gambling_promo"] is True
        assert len(result["matches"]) == 1


def test_gambling_detector_rejects_partial_terms_and_wrong_digit_counts():
    detector = GamblingAdDetector()

    for text in ("menangis88", "menang8", "menang1234", "Singapore news"):
        assert detector.classify(text)["suspected_gambling_promo"] is False


def test_gambling_detector_handles_empty_input():
    assert GamblingAdDetector().classify(None)["label"] == "no_pattern"


def test_gambling_detector_includes_only_verified_number_meanings(tmp_path):
    reference_path = tmp_path / "togelnumbers.json"
    reference_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "number": "88",
                        "meaning": "verified paraphrase",
                        "meaning_status": "contextual",
                        "review_status": "verified",
                        "source_ids": ["licensed-source"],
                    },
                    {
                        "number": "77",
                        "meaning": "pending paraphrase",
                        "meaning_status": "unknown",
                        "review_status": "pending",
                        "source_ids": ["unreviewed-source"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    detector = GamblingAdDetector(number_reference_path=reference_path)

    verified_result = detector.classify("Menang88")
    pending_result = detector.classify("Juara77")

    assert verified_result["matches"][0]["meanings"] == [
        {
            "meaning": "verified paraphrase",
            "meaning_status": "contextual",
            "source_ids": ["licensed-source"],
        }
    ]
    assert pending_result["matches"][0]["meanings"] == []