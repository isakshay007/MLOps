from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

DATASET_NAME = "Diabetes (Efron et al., 2004)"
# The 10 baseline measurements, in dataset order. We load the unscaled values, so a
# prediction request uses real units (age in years, BMI, blood pressure in mm Hg, ...).
FEATURES = {
    "age": "Age in years",
    "sex": "Sex (1 or 2)",
    "bmi": "Body mass index",
    "bp": "Average blood pressure (mm Hg)",
    "s1": "Total serum cholesterol (tc)",
    "s2": "Low-density lipoproteins (ldl)",
    "s3": "High-density lipoproteins (hdl)",
    "s4": "Total cholesterol / HDL (tch)",
    "s5": "Log of serum triglycerides (ltg)",
    "s6": "Blood sugar level (glu)",
}
FEATURE_NAMES = list(FEATURES)


def load_data():
    """
    Load the Diabetes dataset with its original (unscaled) values.
    Returns:
        X (numpy.ndarray): 442 patients x 10 baseline measurements.
        y (numpy.ndarray): Disease progression one year after baseline (25 to 346).
    """
    diabetes = load_diabetes(scaled=False)
    return diabetes.data, diabetes.target


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
        test_size (float): Fraction of patients held out for testing.
        random_state (int): Seed, so the same split is used on every run.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
