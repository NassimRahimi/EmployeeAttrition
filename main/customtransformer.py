import numpy as np
import pandas as pd
import logging
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "main")))
from main.data_preprocessing import remove_outliers, preprocess_employee_data


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OutlierRemover(TransformerMixin, BaseEstimator):
    """
    Custom Transformer to remove outliers based on Z-score.
    """

    def __init__(self, z_score_threshold=3, numerical_columns=None):
        self.z_score_threshold = z_score_threshold
        self.numerical_columns = numerical_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert to pandas DataFrame if it's a NumPy array, to preserve column names
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Call the remove_outliers function and return the cleaned DataFrame
        return remove_outliers(
            X,
            z_score_threshold=self.z_score_threshold,
            numerical_columns=self.numerical_columns,
        )


class CorrelationFilter(BaseEstimator, SelectorMixin, MetaEstimatorMixin):
    """
    Correlation Filter.
    Filters out features that are highly correlated based on a given threshold.
    Input: Numpy array
    Output: Numpy array
    """

    def __init__(self, threshold=0.9):
        """
        Initialize the CorrelationFilter.
        :param threshold: Correlation value that determines which features to eliminate. Default is 0.9.
        """
        if threshold < 0 or threshold > 1:
            raise AttributeError("Threshold must be between 0 and 1!")
        self.threshold = threshold
        self.corr_matrix = None
        self.correlated_features = []
        self.kept_features = []
        self.y_shape = None

    def fit(self, X, y=None):
        self.y_shape = X.shape[1]

        # Compute the correlation matrix
        self.corr_matrix = np.corrcoef(X, rowvar=False)

        # Create an upper triangular matrix with absolute values, excluding the diagonal
        corr_matrix_upper = np.abs(np.triu(self.corr_matrix, k=1))

        # Find the features to remove (those with correlation higher than the threshold)
        f_x, f_y = np.where(corr_matrix_upper > self.threshold)

        # Save the indices of correlated features
        self.correlated_features = list(np.unique(f_y))

        # Keep features that are not highly correlated
        self.kept_features = [
            ind for ind in range(X.shape[1]) if ind not in self.correlated_features
        ]
        logging.info(
            f"CorrelationFilter selected {len(self.kept_features)} features out of {X.shape[1]}."
        )

        return self

    def _get_support_mask(self):
        """
        Returns a boolean mask of features to keep based on the correlation filter.
        """
        return np.array([x in self.kept_features for x in range(self.y_shape)])


# Custom Threshold Classifier
class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, threshold=0.5):
        self.base_estimator = base_estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
