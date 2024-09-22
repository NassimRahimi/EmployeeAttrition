import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import sys
import os


def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config


# Load the config at the start of your application
config = load_config("config/config.json")
custom_threshold = config.get("custom_threshold", 0.5)


# Load the trained XGBoost model
def load_trained_model():
    with open(
        "data/trained_pipeline_xgb_alldata_noreduction_2024-09-21.pkl", "rb"
    ) as file:
        trained_pipeline = pickle.load(file)
    return trained_pipeline


model = load_trained_model()


# Initialize FastAPI app
app = FastAPI()


# Define a data model for validation
class EmployeeData(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    average_monthly_hours: int
    time_spend_company: int
    number_project: int
    salary: str
    department: str
    work_accident: int
    promotion_last_5years: int


# Define a route for predictions
@app.post("/predict")
def predict(data: EmployeeData):
    try:
        # Convert the input data into a pandas DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Predict using the loaded model
        prediction_proba = model.predict_proba(input_data)[0][1]

        # Convert numpy.float32 to a standard Python float
        prediction_proba = float(prediction_proba)

        # Generate the prediction label based on the custom threshold
        prediction_label = "Leave" if prediction_proba > custom_threshold else "Stay"

        # Return the prediction result
        return {
            "prediction": prediction_label,
            "probability": round(prediction_proba * 100, 2),  # Percentage probability
        }
    except Exception as e:
        return {"error": str(e)}


# You can also add a root route for testing
@app.get("/")
def read_root():
    return {"message": "Employee Attrition Prediction API"}
