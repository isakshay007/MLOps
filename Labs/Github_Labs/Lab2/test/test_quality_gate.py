from quality_gate import check_quality

GOOD = {"f1": 0.95, "roc_auc": 0.99, "recall": 0.93}


def test_passes_first_model():
    assert check_quality(GOOD) == []


def test_fails_below_minimum():
    failures = check_quality({**GOOD, "roc_auc": 0.90})
    assert len(failures) == 1
    assert "roc_auc" in failures[0] and "below the minimum" in failures[0]


def test_fails_when_worse_than_latest_model():
    previous = {"f1": 0.98, "roc_auc": 0.99, "recall": 0.97}
    failures = check_quality(GOOD, previous)
    assert {f.split()[0] for f in failures} == {"f1", "recall"}
    assert all("dropped" in f for f in failures)


def test_allows_small_drop_within_tolerance():
    previous = {"f1": 0.955, "roc_auc": 0.995, "recall": 0.935}
    assert check_quality(GOOD, previous) == []


def test_custom_thresholds():
    assert check_quality(GOOD, min_metrics={"f1": 0.96}) != []
    assert check_quality(GOOD, min_metrics={"f1": 0.90}) == []
