# 📊 Employee Attrition Case Study
This project aims to model the probability of employee attrition and provide actionable insights by identifying the key variables that contribute to attrition. The solution involves building an exploratory data analysis (EDA) and running a Machine Learning (ML) model to predict attrition. The final deliverable will offer insights into which variables need immediate attention and provide a roadmap for future ML pipeline implementation.

# 📖 Overview
The goal of this project is to model and understand employee attrition using a dataset provided, focusing on the following key points:

1. Exploratory Data Analysis (EDA) to visualize key features and correlations with attrition.
2. Predictive Modeling to forecast the likelihood of employee attrition.
3. ML Pipeline Proposal outlining how to deploy this solution in production using SQL and a scalable pipeline.
4. Insights & Recommendations based on the model's outcomes to guide future actions in mitigating attrition.

# 🚀 Getting Started

Prerequisites
Before you start, ensure that you have the following installed:

Python 3.11 (or higher)
Required Python packages listed in the requirements.txt file (e.g., streamlit, matplotlib, flask, pandas, scipy, astral)

Navigate to the project directory on your local machine:

## 📦 Installation

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd EmployeeAttrition
```

To install the Python package for local development, first create a
virtualenv, then install requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

To activate the environment whenever you open a new terminal, type:

```bash
source venv/bin/activate
```
## 🖥️ Streamlit App

The Streamlit app provides an interactive interface with four key functionalities:

# 🔑 App Structure
    1.	Exploratory Data Analysis (EDA):Department Metrics, Attrition Analysis, Feature Distributions, Correlation Heatmap, Boxplots
    2.	Model Pipeline:Visualize the machine learning pipeline architecture, including data preprocessing, feature generation, and model training steps, using a pipeline diagram.
    3.	Model Evaluation
    4.	Attrition Prediction Simulation: Simulate attrition by adjusting employee inputs like satisfaction, hours, and salary. Predict the likelihood of an employee staying or leaving.View SHAP explanations to understand key feature impacts.

Running the Streamlit App
To run the Streamlit app, execute the following command from the project directory:

```bash

streamlit run streamlit_app.py
```
By default, the app will be available at http://localhost:8501.
This app provides an interactive and visual way to analyze, evaluate, and simulate employee attrition patterns and trends.


# 🚀 API for Employee Attrition Prediction (FastAPI)

This project includes an API built using FastAPI to predict employee attrition. The API accepts employee data as input and returns a prediction on whether the employee is likely to "Stay" or "Leave," along with the probability.

# 📦 Installation
To set up the FastAPI app, follow these steps:

Install FastAPI and Uvicorn (an ASGI server):
```bash
pip install fastapi uvicorn
```
## 🖥️ Running the FastAPI App
To run the FastAPI app, execute the following command from the project directory:

```bash
uvicorn fast_api_main:app --reload --host 0.0.0.0 --port 8000
```
This will start the API, and it will be available at http://0.0.0.0:8000 by default.
Navigate to the API server: http://localhost:8000/docs in a browser.

## 📊 API Endpoints
1. Root Endpoint (GET request)
    URL: /
        Description: A simple root endpoint to verify if the API is running.
    Example Response:
    ```bash
        {
        "message": "Employee Attrition Prediction API"
        }
    ```

2. Prediction Endpoint (POST request)
    URL: /predict
    Description: This endpoint accepts employee data and returns a prediction on whether the employee will "Stay" or "Leave" along with a probability percentage.
    Example Request Payload:
    ```bash
    {
    "satisfaction_level": 0.5,
    "last_evaluation": 0.8,
    "average_monthly_hours": 200,
    "time_spend_company": 3,
    "number_project": 4,
    "salary": "medium",
    "department": "sales",
    "work_accident": 0,
    "promotion_last_5years": 1
    }
    ```
    Example Response:
    ```bash
    {
    "prediction": "Stay",
    "probability": 64.0
    }
    ```
## 🔧 Example Request Using Curl
To make a request to the /predict endpoint using curl, use the following command:

```bash
    curl -X 'POST' \
    'http://0.0.0.0:8000/predict' \
    -H 'Content-Type: application/json' \
    -d '{
    "satisfaction_level": 0.5,
    "last_evaluation": 0.8,
    "average_monthly_hours": 200,
    "time_spend_company": 3,
    "number_project": 4,
    "salary": "medium",
    "department": "sales",
    "work_accident": 0,
    "promotion_last_5years": 1
    }'
```
Example Response:
```bash
    {
    "prediction": "Stay",
    "probability": 64.0
    }
```

## 🔄 ML Pipeline Proposal
If the PoC is successful, the next step is to build a production-ready Machine Learning pipeline:
    1.	Data Ingestion: Extract data from an SQL server using SQLAlchemy or Pandas.
    2.	Data Preprocessing: Clean, encode categorical features, and scale numerical data through an automated pipeline.
    3.	Feature Engineering: Create new features from existing data to enhance model performance and capture underlying patterns.
    4.	Model Training & Deployment: Train models using Scikit-learn or TensorFlow, and manage deployment with MLflow or Kubeflow.
    5.	Monitoring & Retraining: Continuously monitor model performance and retrain with updated data to ensure accuracy.

# 💾 Data Storage and Retrieval
In a production environment, employee data could be stored in a SQL Server database. To establish a connection and retrieve data, you would need to install the necessary ODBC drivers for SQL Server.

# Install SQL Server ODBC Driver
Before querying the database, install the appropriate driver:
    
[Windows Download](https://learn.microsoft.com/en-us/sql/connect/odbc/microsoft-odbc-driver-for-sql-server?view=sql-server-ver16).
[Linux and Mac Download](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver16&tabs=alpine18-install%2Calpine17-install%2Cdebian8-install%2Credhat7-13-install%2Crhel7-offline).

# Set Up Environment Variables
Create a .env file in the project's root directory and fill it with the following details:
```bash 
        SQL_SERVER=your-sql-server-hostname
        SQL_DATABASE=your-database-name
        SQL_USERNAME=your-username
        SQL_PASSWORD=your-password
    ```
# Establish SQL Server Connection
Using a Python library like pyodbc, you can establish a connection to the SQL Server to fetch data directly into the ML pipeline:
```bash 
        import pyodbc
        import pandas as pd

        # Load environment variables
        import os
        from dotenv import load_dotenv
        load_dotenv()
        # Database connection details
        server = os.getenv("SQL_SERVER")
        database = os.getenv("SQL_DATABASE")
        username = os.getenv("SQL_USERNAME")
        password = os.getenv("SQL_PASSWORD")
        # Establish connection
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=' + server + ';'
            'DATABASE=' + database + ';'
            'UID=' + username + ';'
            'PWD=' + password
        )
        # Query employee data
        query = "SELECT * FROM EmployeeAttritionData"
        df = pd.read_sql(query, connection)
```

# 💾 SQL Server Setup and Querying

To query employee attrition data from SQL Server, follow these steps based on your operating system:

Windows:
```bash
        SET file_search_path = 'C:\path\to\your\files\';
        CREATE TABLE EmployeeAttritionData AS SELECT * FROM 'EmployeeAttritionData.parquet';
        SELECT * FROM 'EmployeeAttritionData.parquet' LIMIT 10;
```
    
    Mac/Linux:
```bash
        SET file_search_path = '/path/to/your/files/';
        CREATE TABLE EmployeeAttritionData AS SELECT * FROM 'EmployeeAttritionData.parquet';
        SELECT * FROM 'EmployeeAttritionData.parquet' LIMIT 10;
```
This allows you to quickly load data into memory and begin exploring it in the SQL editor.
## 📊 Insights & Recommendations
Once the model is trained and tested, the insights can be translated into actionable recommendations:
•	High Attrition Risk Departments: Departments with high attrition risk, as identified by the model, should be investigated for root causes.
•	Impactful Features: Features such as low satisfaction levels, high working hours, or lack of promotions should be addressed as they contribute significantly to attrition.
•	Predictive Monitoring: Set up an automated system to flag employees at high risk of leaving, allowing HR to intervene proactively.

## 📋 Notes
•	The dataset used in this project is cleaned_df.xlsx.
•	Streamlit app provides interactive features for easy data exploration and visualization.

## 🧪 Testing
Run the following command to execute automated tests:
```bash 

pytest
```
Ensure all tests pass before deploying or committing significant changes.

## 🧹 Linting
Use Ruff and Black to lint and format the code:
•	Black: Automatically formats Python code to comply with PEP 8.
•	Ruff: A linter to identify any issues with your code.
Run Black to format your code:
```bash

black .
```
Run Ruff to lint your code:
```bash

sruff .
```

## ❓ Troubleshooting
If you encounter any issues:
•	Missing dependencies: Ensure that all required dependencies are installed by running pip install -r requirements.txt.
•	Virtual environment issues: Make sure your virtual environment is activated before running the app.

