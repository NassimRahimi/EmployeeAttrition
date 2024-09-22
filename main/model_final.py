import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

import sys
import os

# Add the main directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.customtransformer import OutlierRemover, CorrelationFilter, ThresholdClassifier
from main import data_preprocessing

from sklearn.decomposition import PCA
from data_preprocessing import preprocess_employee_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the configuration from a JSON file
def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config

# Load the config at the start of your application
config = load_config("config/config.json")
custom_threshold = config.get("custom_threshold", 0.14)
corr_threshold = config.get("corr_threshold", 0.90)
lasso_cv = config.get("lasso_cv", 5)
z_score_threshold = config.get("z_score_threshold", 4.8)
pca_variance = config.get("pca_variance", 0.95)
# Define the known categories for OneHotEncoder
department_categories = ["sales", "accounting", "hr", "technical", "support", "management", "product_mng", "marketing"]
salary_categories = ["low", "medium", "high"]
# Function to preprocess data and train the model
def preprocess_and_train_model(df, custom_threshold=custom_threshold):
    """
    Preprocess the data and train a model using RandomForest and XGBoost with a custom threshold.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        custom_threshold (float): The threshold for classification (default=0.14).
    
    Returns:
        Tuple: training/test datasets, predictions, and trained pipeline.
    """

    # Drop columns not needed for training
    y = df["left"].astype(int)
    X = df.drop(columns=["left", "Employee_ID"])
    # Check if the columns are properly removed
    assert "Employee_ID" not in X.columns, "Employee_ID should have been removed!"
    assert "left" not in X.columns, "left should have been removed!"

    # Define the numerical columns to pass into OutlierRemover
    numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 
                         'time_spend_company', 'work_accident', 'promotion_last_5years']

    # Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X.copy()
    y_train = y.copy()

    # Apply outlier removal on X_train and synchronize y_train
    outlier_remover = OutlierRemover(z_score_threshold=z_score_threshold, numerical_columns=numerical_columns)
    X_train_cleaned = outlier_remover.fit_transform(X_train)
    
    # Align y_train by removing rows corresponding to removed outliers
    y_train_cleaned = y_train.loc[X_train_cleaned.index]

    # Separate numerical and categorical features
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = ["department", "salary"] 

    # Define preprocessing steps for numerical and categorical features
    numeric_pipeline = Pipeline([
        ("scale", StandardScaler())  # Scale numeric features
    ])

    # Categorical pipeline with explicitly defined categories for OneHotEncoder
    categorical_pipeline = Pipeline([
        ("one_hot", OneHotEncoder(categories=[department_categories, salary_categories], 
                                  drop=None, sparse_output=False))  # Using known categories
    ])

    # ColumnTransformer to apply preprocessing to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
    remainder='drop'  # Drop any untransformed features
)
    # Feature selection using LassoCV
    SelectLassoCV = SelectFromModel(
        estimator=LassoCV(cv=lasso_cv, random_state=42),
        threshold=None
    )

    # Apply Correlation Filter before the feature selection (Lasso)
    correlation_filter = CorrelationFilter(threshold=corr_threshold)
    # PCA (Principal Component Analysis)
    pca = PCA(n_components=pca_variance)  # Retain 95% variance
    

    # Models with class weight balancing
    rf_model = RandomForestClassifier(class_weight="balanced")
    xgb_model = xgb.XGBClassifier(scale_pos_weight=(y_train_cleaned == 0).sum() / (y_train_cleaned == 1).sum())

    # Wrap XGBoost model in a threshold classifier
    threshold_adjusted_xgb_model = ThresholdClassifier(base_estimator=xgb_model, threshold=custom_threshold)

    # Final pipeline
    ModellingPipeline = Pipeline([
        ("preprocessor", preprocessor),  # Preprocessing numeric/categorical
        #("correlation_filter", correlation_filter),  # Correlation filter to remove highly correlated features
        #("pca", pca),
        #("lasso", SelectLassoCV),  # Lasso-based feature selection
        ("estimator", threshold_adjusted_xgb_model)  # Final estimator
    ])

    # Fit the pipeline
    logging.info("Fitting the ModellingPipeline!")
    trained_pipeline = ModellingPipeline.fit(X_train_cleaned, y_train_cleaned)

    # Predict probabilities and class labels
    logging.info("Predicting probabilities and labels!")
    y_train_hat = trained_pipeline.predict_proba(X_train_cleaned)[:, 1]
    #y_test_hat = trained_pipeline.predict_proba(X_test)[:, 1]
    y_train_pred = trained_pipeline.predict(X_train_cleaned)
    #y_test_pred = trained_pipeline.predict(X_test)

    return {
        "X_train": X_train_cleaned,
        #"X_test": X_test,
        "y_train": y_train_cleaned,
        #"y_test": y_test,
        "y_train_hat": y_train_hat,
        #"y_test_hat": y_test_hat,
        "y_train_pred": y_train_pred,
        #"y_test_pred": y_test_pred,
        "trained_pipeline": trained_pipeline
    }


if __name__ == "__main__":
    df = pd.read_excel("data/Worksheet_in_Case_Study.xlsx")
    df = preprocess_employee_data(df)
    print(df.columns)

    # Call the function to preprocess and train the model
    results = preprocess_and_train_model(df)

    X_train = results["X_train"]
    #X_test = results["X_test"]
    y_train = results["y_train"]
    #y_test = results["y_test"]
    y_train_hat = results["y_train_hat"]
    #y_test_hat = results["y_test_hat"]
    y_train_pred = results["y_train_pred"]
    #y_test_pred = results["y_test_pred"]
    trained_pipeline = results["trained_pipeline"]

    # Log model success
    logging.info("Model training and prediction complete!")
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
     # Save the trained pipeline using pickle
    pipeline_filename_pickle = f"trained_pipeline_xgb_alldata_noreduction_{current_date}.pkl"
    with open(pipeline_filename_pickle, "wb") as file:
        pickle.dump(trained_pipeline, file)
    logging.info(f"Trained pipeline saved to {pipeline_filename_pickle} using pickle")

    # Ensure numeric and categorical features are defined based on X_train
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = ["department", "salary"]  # Only these two categorical columns

    # Extract feature names from the preprocessor (ColumnTransformer)
    preprocessor = trained_pipeline.named_steps['preprocessor']

    # Extract numeric and one-hot encoded feature names
    ohe_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)

    # Combine numeric and one-hot encoded feature names
    feature_names = np.hstack([numeric_features, ohe_feature_names])

    # Extract feature importances from the trained model
    feature_importances = trained_pipeline.named_steps['estimator'].base_estimator.feature_importances_

    # Match the selected feature names with the model's selected features
    #feature_names_after = feature_names[:len(feature_importances)]  # Adjust for PCA or Lasso

    # Create a DataFrame for feature importance and feature names
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the feature importances in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the top features
    plt.figure(figsize=(18, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='magenta')
    plt.xlabel('Feature Importance')
    plt.title('Top Features by Importance')
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    plt.show()
