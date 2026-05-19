"""
Data preprocessing module for ML pipeline.
Uses Iris dataset and applies train-test split.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    """
    Loads and splits Iris dataset.

    Returns:
        X_train, X_test, y_train, y_test
    """
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
