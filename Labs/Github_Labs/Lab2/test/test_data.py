import numpy as np

from data import FEATURE_NAMES, load_data, split_data


def test_dataset_shape():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert len(FEATURE_NAMES) == 30
    assert set(np.unique(y)) == {0, 1}


def test_malignant_is_positive_class():
    # The dataset has 212 malignant and 357 benign tumours; malignant must be labelled 1
    _, y = load_data()
    assert y.sum() == 212


def test_split_is_stratified_and_reproducible():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(y_train) == 455 and len(y_test) == 114
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert np.array_equal(split_data(X, y)[1], X_test)
