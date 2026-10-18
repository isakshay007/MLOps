from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# sklearn names the last feature "od280/od315_of_diluted_wines"; "/" isn't valid in a
# Python/JSON field name, so the API uses underscores. Order matches the dataset columns.
FEATURE_NAMES = [name.replace("/", "_") for name in load_wine().feature_names]
CLASS_NAMES = ["cultivar_1", "cultivar_2", "cultivar_3"]


def load_data():
    """
    Load the Wine dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): 178 samples x 13 chemical measurements.
        y (numpy.ndarray): Cultivar label (0, 1 or 2) for each sample.
    """
    wine = load_wine()
    return wine.data, wine.target


def split_data(X, y):
    """
    Split the data into training and testing sets.
    Stratified so all three cultivars keep the same proportions in both sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
