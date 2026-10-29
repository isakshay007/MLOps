from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

DATASET_NAME = "Breast Cancer Wisconsin (Diagnostic)"
FEATURE_NAMES = list(load_breast_cancer().feature_names)
# sklearn encodes malignant as 0 and benign as 1. We flip it so 1 = malignant, which makes
# malignant the positive class for precision, recall and F1 (the class we must not miss).
CLASS_NAMES = ["benign", "malignant"]
RANDOM_STATE = 42


def load_data():
    """
    Load the Breast Cancer Wisconsin dataset.
    Returns:
        X (numpy.ndarray): 569 tumours x 30 cell-nucleus measurements.
        y (numpy.ndarray): 1 if the tumour is malignant, 0 if benign.
    """
    cancer = load_breast_cancer()
    return cancer.data, 1 - cancer.target


def split_data(X, y):
    """
    Split the data into training and testing sets.
    Stratified so both sets keep the same malignant/benign ratio (about 37% malignant).
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
