import pandas as pd
import streamlit as st
from main.visualization import (
    plot_correlation_heatmap,
    plot_boxplots_vs_left,
    plot_correlation_with_target,
    plot_class_balance,
)
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import sys
import os
import json
import numpy as np
import shap
import logging

logging.getLogger().setLevel(logging.WARNING)
import warnings

warnings.filterwarnings("ignore")
# Set the logging level to suppress INFO messages
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")  # Prevent Matplotlib from trying to display plots interactively

# Additionally, filter Matplotlib specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Using categorical units to plot a list of strings that are all parsable as floats or dates.",
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .small-header {
        font-size: 13px;
        font-weight: bold;
        margin-top: 15px;
    }
    .explanation {
        font-size: 10px;
        font-weight: normal;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load images using Streamlit caching
@st.cache_resource
def load_image(image_path):
    return Image.open(image_path)


# Load the configuration from a JSON file
def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config


# Load the config at the start of your application
config = load_config("config/config.json")
custom_threshold = config.get("custom_threshold", 0.5)


from main.customtransformer import (
    OutlierRemover,
    CorrelationFilter,
    ThresholdClassifier,
)
from main import data_preprocessing


# Load the trained model
@st.cache_data
def load_trained_model():
    with open(
        "data/trained_pipeline_xgb_alldata_noreduction_2024-09-21.pkl", "rb"
    ) as file:
        trained_pipeline = pickle.load(file)
    return trained_pipeline


# Load the necessary data
@st.cache_data
def load_data():
    return pd.read_excel("data/cleaned_df.xlsx")


# Load the data
df = load_data()
df.drop("Employee_ID", axis=1, inplace=True)

model = load_trained_model()

# Custom CSS for sidebar width and height
st.markdown(
    """
    <style>
    /* Reduce the width of the sidebar */
    [data-testid="stSidebar"] {
        width: 100px;  /* Adjust this value to make it slimmer */
    }

    /* Adjust content within the sidebar */
    [data-testid="stSidebar"] .css-1lcbmhc.e1fqkh3o3 {
        padding-top: 5px;   /* Reduce top padding */
        padding-left: 5px;  /* Reduce left padding */
        padding-right: 5px; /* Reduce right padding */
    }

    /* Adjust sidebar font size */
    .sidebar .stRadio, .sidebar .stButton {
        font-size: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS to style the title
st.markdown(
    """
    <style>
    .title {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .small-header {
        font-size: 14px;
        font-weight: bold;
        margin-top: 15px;
    }
    .explanation {
        font-size: 12px;
        font-weight: normal;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Title Above the Logo
st.sidebar.markdown("<h1 style='text-align: center; font-size: 10px;'>Employee Attrition Dashboard</h1>", unsafe_allow_html=True)

# Display the logo in the sidebar
col1, col2, col3 = st.sidebar.columns([1, 5, 1])
col2.image('data/Novo_Nordisk_Logo.png', width=100)

# st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Exploratory Data Analysis",
        "Model Pipeline",
        "Model Evaluation",
        "Insights",
        "Attrition Prediction Simulation",
    ],
)

# Caching the plot functions using @st.cache_resource
@st.cache_resource
def get_correlation_with_target(df):
    return plot_correlation_with_target(df)


@st.cache_resource
def get_boxplots_vs_left(df):
    return plot_boxplots_vs_left(df)


@st.cache_resource
def get_correlation_heatmap(df):
    return plot_correlation_heatmap(df)


# EDA Page
if page == "Exploratory Data Analysis":
    st.markdown(
        "<h1 class='title'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True
    )

    # Display Department Metrics
    st.markdown(
        "<h2 class='small-header'>1. Department Metrics</h2>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='explanation'>Below is the analysis of various metrics by department.</p>",
        unsafe_allow_html=True,
    )
    department_metrics_img = load_image("data/Department_Metrics.png")
    st.image(department_metrics_img, caption="Department Metrics")
    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">Management, product management, and support departments show the highest average **satisfaction levels** among employees, while HR and accounting show the lowest.</li>
        <li style="font-size: 11px;">Technical and management departments have the highest **average monthly hours**, suggesting that these departments might experience higher workloads.</li>
        <li style="font-size: 11px;">Employees in management and technical departments tend to spend more time in the company, while HR and accounting show shorter average tenures.</li>
        <li style="font-size: 11px;">The **last evaluation** score remains fairly consistent across all departments, indicating that performance evaluations might not be department-specific.</li>
    </ul>
    """, unsafe_allow_html=True
    )

    # Display Attrition Analysis
    st.markdown(
        "<h2 class='small-header'>2. Attrition Analysis</h2>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='explanation'>Pie charts showing employee attrition by department and salary.</p>",
        unsafe_allow_html=True,
    )
    attrition_analysis_img = load_image("data/Attrition_Analysis.png")
    st.image(attrition_analysis_img, caption="Attrition Analysis")
    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">A significant proportion of employees leaving the company have low salaries (over 61%), making salary a key driver of attrition.</li>
        <li style="font-size: 11px;">The sales and support departments contribute the most to overall attrition, with 31.9% and 17.5% respectively.</li>
        <li style="font-size: 11px;">The technical department follows closely with 22% attrition, indicating this group also faces high turnover.</li>
        <li style="font-size: 11px;">Addressing salary-related concerns may help reduce attrition rates, particularly in the sales, support, and technical departments.</li>
    </ul>
    """, unsafe_allow_html=True
)


    # Display Percentage of Employees Left by Salary Category
    st.markdown(
        "<h2 class='small-header'>3. Percentage of Employees Left by Department and Salary Category</h2>",
        unsafe_allow_html=True,
    )
    percentage_employees_left_img = load_image("data/Percentage_Employees_Left.png")
    st.image(
        percentage_employees_left_img,
        caption="Percentage of Employees Left by Salary Category",
    )

    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">The HR and accounting departments show the highest attrition rates, with around 29.7% and 27.1% of employees leaving, respectively.</li>
        <li style="font-size: 11px;">Management has the lowest attrition rate, with only 15.7% of employees leaving.</li>
        <li style="font-size: 11px;">Employees with low salaries have the highest attrition rate (30.8%), while those with high salaries have the lowest attrition rate (7.2%).</li>
        <li style="font-size: 11px;">Sales and support departments also show significant attrition, suggesting these areas may need more focus for retention strategies.</li>
    </ul>
    """, unsafe_allow_html=True
    )

    # Display Distribution of Features
    st.markdown(
        "<h2 class='small-header'>4. Distribution of Features</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='explanation'>Distribution of key numerical features such as satisfaction, evaluation, monthly hours, and time spent in the company.</p>",
        unsafe_allow_html=True,
    )
    distribution_features_img = load_image("data/Distribution_of_Features.png")
    st.image(distribution_features_img, caption="Distribution of Features")
    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">The **satisfaction level** shows a diverse distribution, with noticeable dips around 0.3 and peaks close to 0.8. Employees with low satisfaction are likely to leave.</li>
        <li style="font-size: 11px;">The **last evaluation** is fairly evenly distributed across the company, with a slight concentration around 0.5 to 0.8.</li>
        <li style="font-size: 11px;">**Average monthly hours** are distributed bimodally, with peaks around 150 and 250 hours, suggesting two distinct work-hour groups. Employees in the upper range may experience burnout.</li>
        <li style="font-size: 11px;">The **time spent in the company** is heavily skewed towards shorter tenures, with most employees staying for 3 years, indicating high turnover early in tenure.</li>
    </ul>
    """, unsafe_allow_html=True
    )


    # Display Correlation Heatmap
    st.markdown(
        "<h2 class='small-header'>5. Correlation Heatmap</h2>", unsafe_allow_html=True
    )
    heatmap_img = load_image("data/Heatmap.png")
    st.image(heatmap_img, caption="Correlation Heatmap")
    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">**Satisfaction level** has the strongest negative correlation with attrition (left), indicating that employees with lower satisfaction are more likely to leave.</li>
        <li style="font-size: 11px;">**Work accident** and **salary** show weak correlations with attrition, suggesting these factors may have less influence on employee turnover.</li>
        <li style="font-size: 11px;">**Average monthly hours** and **last evaluation** are slightly positively correlated, indicating that high-performing employees with long hours might have a higher likelihood of leaving.</li>
        <li style="font-size: 11px;">**Number of projects** is moderately correlated with time spent at the company, suggesting that employees with longer tenures tend to manage more projects.</li>
    </ul>
    """, unsafe_allow_html=True
    )


    # 6. Correlation with Target
    st.markdown(
        "<h2 class='small-header'>6. Correlation with Target</h2>",
        unsafe_allow_html=True,
    )
    fig_plotly = get_correlation_with_target(df)
    st.plotly_chart(fig_plotly)
    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">**Satisfaction level** has the strongest negative correlation with employee attrition, meaning that employees with low satisfaction levels are highly likely to leave.</li>
        <li style="font-size: 11px;">**Work accident** shows a weaker but still notable negative correlation, suggesting that employees who have experienced a work accident may be less likely to leave.</li>
        <li style="font-size: 11px;">**Salary** is weakly negatively correlated with attrition, implying that employees with higher salaries are slightly less likely to leave.</li>
        <li style="font-size: 11px;">Other variables such as **last evaluation**, **department**, and **number of projects** have minimal correlations with attrition, indicating they may not be key factors driving turnover.</li>
        <li style="font-size: 11px;">**Time spent in the company** has a very small positive correlation, which indicates that longer tenure might slightly increase the likelihood of leaving, but it's not a strong factor.</li>
    </ul>
    """, unsafe_allow_html=True
    )


    # 7. Boxplots vs Left 
    st.markdown(
        "<h2 class='small-header'>7. Boxplots vs Left</h2>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='explanation'>Boxplots comparing numerical features with employee attrition.</p>",
        unsafe_allow_html=True,
    )
    fig = get_boxplots_vs_left(df)
    st.pyplot(fig)
    
    st.markdown(
    """
    <ul>
        <li style="font-size: 11px;">Employees who left (1) have a much lower **satisfaction level** than those who stayed (0), indicating dissatisfaction as a strong predictor of attrition.</li>
        <li style="font-size: 11px;">The **last evaluation** is similar between employees who stayed and left, suggesting it may not be a significant predictor of attrition.</li>
        <li style="font-size: 11px;">Employees who left tend to have **more projects** compared to those who stayed, indicating that high workload or project count might contribute to attrition.</li>
        <li style="font-size: 11px;">There is no significant difference in **average monthly hours** between those who stayed and left, suggesting that this factor may not be a key driver of attrition.</li>
        <li style="font-size: 11px;">Employees with **longer tenures** (time spent at the company) are more likely to leave, indicating that attrition increases after a few years of service.</li>
        <li style="font-size: 11px;">The **work accident** feature shows little to no difference between those who left and stayed, suggesting it is not a strong indicator of attrition.</li>
        <li style="font-size: 11px;">**Salary levels** (encoded) appear to be relatively balanced between employees who left and stayed, indicating salary might not be a strong independent factor but may need to be considered in combination with other variables.</li>
        <li style="font-size: 11px;">The **promotion in the last 5 years** shows no significant variation between those who stayed and those who left, suggesting it has minimal impact on attrition.</li>
    </ul>
    """, unsafe_allow_html=True
    )


    # 8. Class Balance of Target Variable (Keep dynamic)
    st.markdown(
        "<h2 class='small-header'>8. Class Balance of Target Variable</h2>",
        unsafe_allow_html=True,
    )
    plot_class_balance(df, target_column="left")

# Pipeline Page
if page == "Model Pipeline":
    st.markdown("<h1 class='title'>Model Pipeline</h1>", unsafe_allow_html=True)

    # Load and display the pipeline diagram
    pipeline_image = Image.open("data/pipeline_diagram.png")
    st.image(
        pipeline_image, caption="Attrition Modeling Pipeline", use_column_width=True
    )


# Evaluation Page
elif page == "Model Evaluation":
    st.markdown(
        "<h1 class='title'>Employee Attrition Prediction</h1>", unsafe_allow_html=True
    )

    st.markdown(
        "<p class='explanation'>Here you can evaluate the XGBoost model predictions, view the confusion matrix, ROC curve, and SHAP summary plot.</p>",
        unsafe_allow_html=True,
    )

    # Add evaluation metrics
    st.markdown(
        "<h2 class='small-header'>Model Evaluation Metrics</h2>", unsafe_allow_html=True
    )
    st.write(f"**Accuracy**: 0.85")
    st.write(f"**Precision**: 0.82")
    st.write(f"**Recall**: 0.78")
    st.write(f"**F1 Score**: 0.80")
    st.write(f"**ROC-AUC Score**: 0.90")
    # Add buttons for displaying confusion matrix, ROC curve, and SHAP summary plot
    if st.button("Show Confusion Matrix"):
        # Load and display the confusion matrix
        confusion_matrix_image = Image.open("data/confusion_matrix.png")
        st.image(confusion_matrix_image, caption="Confusion Matrix")

    if st.button("Show ROC Curve"):
        # Load and display the ROC curve
        roc_curve_image = Image.open("data/roc_curve.png")
        st.image(roc_curve_image, caption="ROC Curve")

    if st.button("Show SHAP Summary Plot"):
        # Load and display the SHAP summary plot
        shap_summary_image = Image.open("data/shap_summary_plot.png")
        st.image(shap_summary_image, caption="SHAP Summary Plot")
        # SHAP Plot Interpretation Section
        st.markdown(
            "<h2 class='small-header'>SHAP Plot Interpretation</h2>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        **General Overview**:
        The SHAP (SHapley Additive exPlanations) values shown in this plot explain how each feature contributes to the model’s output for predicting employee attrition. The position of the dots indicates the SHAP value (impact on model output), while the color reflects the feature value (blue for low, red for high).

        Positive SHAP values (to the right of the vertical line at zero) indicate that the feature increases the likelihood of an employee leaving (attrition), while negative values (to the left) indicate a reduced likelihood of attrition.
        
        ### Key Feature Insights:
        
        - **Satisfaction Level**: Low satisfaction (red dots) significantly increases the likelihood of attrition, while high satisfaction (blue dots) reduces the probability of attrition.
        - **Time Spent at the Company**: Higher time spent (red dots) generally increases the likelihood of attrition, while lower tenure (blue dots) tends to decrease the likelihood of attrition.
        - **Average Monthly Hours**: Higher monthly hours (red dots) are linked with higher attrition probability, while lower monthly hours (blue dots) decrease the chance of leaving.
        - **Last Evaluation**: Employees with high performance evaluations (red dots) show a moderate increase in attrition probability.
        - **Number of Projects**: Overburdened employees handling too many projects (red dots) are more likely to leave.
        - **Work Accident**: Having a work accident marginally increases the likelihood of attrition.
        - **Salary (Low, Medium, High)**: Low salary (red dots) increases attrition, while high salary (blue dots) reduces attrition risk.
        - **Departments**: Different departments show variations in attrition risks, potentially due to work culture or growth opportunities.
        - **Promotion in the Last 5 Years**: Having been promoted shows a slight increase in attrition, possibly due to employees seeking further growth opportunities elsewhere.
        
        **Conclusion**: 
        The SHAP plot highlights the main drivers of employee attrition. The key factors driving attrition are satisfaction level, time spent at the company, workload, number of projects, and salary. Lower satisfaction, higher workloads, long tenure, and lower salaries are the main risk factors for attrition.
        """
        )


# Insights Page
elif page == "Insights":
    st.markdown(
        "<h1 class='title'>Insights, Retention Strategies and Roadmap for Implementation</h1>", unsafe_allow_html=True
    )
    st.markdown(
    """
    <ul>
        <li style="font-size: 12px;">Satisfaction Level: Both EDA and SHAP analysis highlight that **satisfaction level** is the most significant predictor of attrition.</li>
        <li style="font-size: 12px;">Compensation: Low salary drives attrition, especially in **sales, support, and technical** departments where many leavers are in lower salary categories.</li>
        <li style="font-size: 12px;">Workload and Burnout: Higher monthly hours and project loads correlate with higher attrition, indicating burnout and overwork as key issues.</li>
        <li style="font-size: 12px;">Tenure: Longer tenure increases attrition risk, with employees around the 3-year mark showing higher turnover, likely due to feeling stagnant or undervalued.</li>
        <li style="font-size: 12px;">Department-Specific Trends: Departments such as sales, support, and technical teams show higher attrition, requiring department-specific interventions.</li>
        <li style="font-size: 12px;">Performance Evaluation and Promotions: Performance evaluations and promotions show little impact on retention. High evaluation scores slightly increase attrition, indicating high performers may seek opportunities elsewhere.</li>
    </ul>
    """, unsafe_allow_html=True
    )
    strategy_img = load_image("data/retention_strategies_graph.png")
    st.image(strategy_img, caption="Retention Strategies")
    roadmap_img = load_image("data/roadmap_graph.png")
    st.image(roadmap_img, caption="Roadmap for Implementation") 
    
      
    # Simulation Page
elif page == "Attrition Prediction Simulation":
    st.markdown(
        "<h1 class='title'>Interactive Attrition Prediction Simulation</h1>", unsafe_allow_html=True
    )

    # Input sliders for user to set feature values in the sidebar
    with st.sidebar:
        st.markdown(
            "<h2 class='small-header'>Input Parameters for Prediction</h2>",
            unsafe_allow_html=True,
        )

        satisfaction = st.slider(
            "Satisfaction Level", 0.0, 1.0, step=0.01, label_visibility="visible"
        )
        evaluation = st.slider(
            "Last Evaluation", 0.0, 1.0, step=0.01, label_visibility="visible"
        )
        monthly_hours = st.slider(
            "Average Monthly Hours", 80, 320, step=1, label_visibility="visible"
        )
        time_spent = st.slider("Time Spent at the Company (Years)", 1, 10, step=1)
        number_of_projects = st.slider("Number of Projects", 1, 10, step=1)
        salary = st.selectbox("Salary Level", ["low", "medium", "high"])
        department = st.selectbox(
            "Department",
            sorted(
                [
                    "sales",
                    "accounting",
                    "hr",
                    "technical",
                    "support",
                    "management",
                    "product_mng",
                    "marketing",
                ]
            ),
        )

        work_accident = st.selectbox("Work Accident (0 = No, 1 = Yes)", [0, 1])
        promotion_last_5years = st.selectbox(
            "Promotion in Last 5 Years (0 = No, 1 = Yes)", [0, 1]
        )

        # Prediction button for triggering the prediction
    if st.button("Predict Attrition"):
        # Create a DataFrame with user inputs
        input_data = pd.DataFrame(
            {
                "satisfaction_level": [satisfaction],
                "last_evaluation": [evaluation],
                "average_monthly_hours": [monthly_hours],
                "time_spend_company": [time_spent],
                "number_project": [number_of_projects],
                "salary": [salary],
                "department": [department],
                "work_accident": [work_accident],
                "promotion_last_5years": [promotion_last_5years],
            }
        )

        # Show employee data on the main page
        st.markdown("<h4>Employee data for prediction:</h4>", unsafe_allow_html=True)
        st.dataframe(input_data)

        # Perform prediction using the model
        prediction_proba = model.predict_proba(input_data)[0][1]

        # Define custom threshold for prediction
        prediction_label = "Leave" if prediction_proba > custom_threshold else "Stay"

        # Display the prediction on the main page with smaller text
        st.markdown(
            f"<p style='font-size:12px;'>**Prediction**: The employee is likely to <b>{prediction_label}</b> with a probability of {prediction_proba*100:.2f}%</p>",
            unsafe_allow_html=True,
        )

        def load_shap_explainer(model):
            return shap.TreeExplainer(model.named_steps["estimator"].base_estimator)

        explainer = load_shap_explainer(model)
        preprocessor = model.named_steps["preprocessor"]
        input_data_transformed = preprocessor.transform(input_data)

        # Calculate SHAP values for the transformed input data
        shap_values = explainer.shap_values(input_data_transformed)

        # Extract feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()

        # Remove 'numeric__' and 'categorical__' prefixes from feature names
        cleaned_feature_names = [
            name.replace("numeric__", "").replace("categorical__", "")
            for name in feature_names
        ]

        # Use the cleaned feature names in SHAP plot
        if shap_values is not None:
            st.markdown(
                "<h4 style='font-size:12px;'>SHAP Explanation:</h4>",
                unsafe_allow_html=True,
            )
            # shap.initjs()

            # Render the SHAP waterfall plot for the first prediction
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_data_transformed[0],  # Transformed feature values
                    feature_names=cleaned_feature_names,
                )
            )  # Use cleaned feature names
            st.pyplot(fig)
