from mbcdisasm.string_analysis import (
    StringClassifier,
    summarise_insights,
    token_histogram,
)


def test_classifier_identifies_camel_case_identifier() -> None:
    classifier = StringClassifier()
    insight = classifier.classify("SetPlayerName")
    assert insight.classification == "identifier"
    assert insight.tokens[:3] == ("set", "player", "name")
    assert insight.confidence >= 0.7
    assert insight.entropy > 0.0
    assert insight.case_style == "mixed"
    data = insight.as_dict()
    assert data["classification"] == "identifier"
    assert data["case_style"] == "mixed"
    assert 0.0 <= data["printable_ratio"] <= 1.0
    assert data["token_density"] > 0.0


def test_classifier_detects_paths_and_formats() -> None:
    classifier = StringClassifier()
    path_insight = classifier.classify("/assets/scripts/init.lua")
    assert path_insight.classification == "path"
    assert "path" in path_insight.hints

    format_insight = classifier.classify("Hello %s! Score: %d")
    assert format_insight.classification == "format"
    assert "format" in format_insight.hints


def test_classifier_low_printable_detection() -> None:
    classifier = StringClassifier()
    noisy_text = "Hello\x01World"
    insight = classifier.classify(noisy_text)
    assert insight.printable_ratio < 1.0
    assert insight.entropy > 0.0


def test_token_histogram_counts_tokens() -> None:
    classifier = StringClassifier()
    insights = [
        classifier.classify("AlphaBeta"),
        classifier.classify("AlphaGamma"),
        classifier.classify("Beta"),
    ]
    histogram = token_histogram(insights, limit=3)
    assert histogram[0][0] == "alpha"
    assert histogram[0][1] == 2
    assert any(token == "beta" for token, _ in histogram)


def test_summarise_insights_orders_histogram() -> None:
    classifier = StringClassifier()
    insights = [
        classifier.classify("Quest complete"),
        classifier.classify("/tmp/output.txt"),
        classifier.classify("PlayerName"),
        classifier.classify("Do you accept the quest?")
    ]
    total, histogram = summarise_insights(insights)
    assert total == 4
    assert histogram[0][0] in {"identifier", "path", "sentence", "dialogue", "unknown"}
