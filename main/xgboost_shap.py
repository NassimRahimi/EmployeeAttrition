import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_employee_data
import sys
import os
import json
# Add the main directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the configuration from a JSON file
def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config

# Load the config at the start of your application
config = load_config("config/config.json")
custom_threshold = config.get("custom_threshold", 0.14)


from main.customtransformer import OutlierRemover, CorrelationFilter, ThresholdClassifier
from main import data_preprocessing


# Load the trained pipeline (replace the filename with the one you saved)
pipeline_filename_pickle = "data/trained_pipeline_xgb_alldata_noreduction_2024-09-21.pkl"
with open(pipeline_filename_pickle, "rb") as file:
    trained_pipeline = pickle.load(file)

# Load your data (in this case, X_train should be the same dataset you used to train the model)
df = pd.read_excel("data/Worksheet_in_Case_Study.xlsx")
df = preprocess_employee_data(df)

# Drop columns that are not needed for predictions
X = df.drop(columns=["left", "Employee_ID"])

# Extract feature names from the preprocessor (ColumnTransformer)
preprocessor = trained_pipeline.named_steps['preprocessor']

# Ensure numeric and categorical features are defined based on X_train
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = ["department", "salary"] 
# Extract feature names from one-hot encoding (with no categories dropped)
ohe_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)

# Combine numeric and one-hot encoded feature names
feature_names = np.hstack([numeric_features, ohe_feature_names])

# Transform the dataset with the pipeline (get preprocessed data)
X_preprocessed = pd.DataFrame(trained_pipeline.named_steps['preprocessor'].transform(X), columns=feature_names)
print(X_preprocessed.columns)
# Extract the XGBoost model from the pipeline
xgb_model = trained_pipeline.named_steps['estimator'].base_estimator

# Initialize the SHAP TreeExplainer with the trained XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Apply SHAP explainer on the preprocessed data
shap_values = explainer.shap_values(X_preprocessed)

# Plot the SHAP summary plot with all features
plt.figure(figsize=(12, 8))  # Adjust figure size for better readability
shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names, plot_type="dot", max_display=len(feature_names), show=True)

# Save the SHAP summary plot to a file
output_file = "shap_summary_plot_all_features.png"  # You can change the file format to .pdf if needed
plt.savefig(output_file, format="png", bbox_inches="tight")  # Save the figure
plt.close()  # Close the plot after saving

print(f"SHAP summary plot saved to {output_file}")
plt.title('SHAP Summary Plot (All Features)', fontsize=14)  # Add a title to the plot
plt.tight_layout()  # Ensure the layout fits well
plt.show()
