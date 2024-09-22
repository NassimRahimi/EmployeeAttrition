from matplotlib import pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    fbeta_score,
    confusion_matrix,
)
import numpy as np
import seaborn as sns
import os
import pandas as pd
import datetime
import pickle

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from data_preprocessing import preprocess_employee_data
from modeling_pipeline import preprocess_and_train_model


class ModelEvaluator:
    """
    Base class for model evaluation.
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline


class BinaryModelEvaluator(ModelEvaluator):
    """
    Evaluator for binary classification models (0-1).
    """

    @staticmethod
    def build_roc_curve(input_frames=[]):
        fig, ax = plt.subplots(figsize=(10, 10))
        lw = 2

        for input_frame in input_frames:
            fpr, tpr, _ = roc_curve(
                y_true=input_frame[0], y_score=input_frame[1], pos_label=1
            )
            roc_auc = round(
                roc_auc_score(y_true=input_frame[0], y_score=input_frame[1]), 4
            )
            ax.plot(
                fpr, tpr, lw=lw, label=f"ROC curve {input_frame[2]} (AUC = {roc_auc})"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=lw,
            linestyle="--",
            label="Random estimator",
        )
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc=0)

        return fig

    @staticmethod
    def build_feature_importances(f_importances=[], f_names=[], top=50):
        fig, ax = plt.subplots(figsize=(10, 10))

        order_ind = np.argsort(-1 * f_importances)
        features = tuple(np.array(f_names)[order_ind][:top])
        y_pos = np.arange(len(features))
        importance = f_importances[order_ind][:top]

        ax.barh(
            y_pos,
            importance,
            align="center",
            color="magenta",
            ecolor="black",
            alpha=0.5,
        )
        ax.grid()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Feature importance")

        return fig

    @staticmethod
    def generate_model_report(y_actual, y_predicted):
        metrics = {
            "accuracy": accuracy_score(y_actual, y_predicted),
            "balanced_accuracy": balanced_accuracy_score(y_actual, y_predicted),
            "precision": precision_score(y_actual, y_predicted),
            "recall": recall_score(y_actual, y_predicted),
            "f1": f1_score(y_actual, y_predicted),
            "f1_macro": f1_score(y_actual, y_predicted, average="macro"),
            "f1_micro": f1_score(y_actual, y_predicted, average="micro"),
            "f1_weighted": f1_score(y_actual, y_predicted, average="weighted"),
            "fbeta": fbeta_score(y_actual, y_predicted, beta=1),
        }
        return metrics

    @staticmethod
    def build_confusion_matrix(y_true, y_pred, ax, title="Confusion Matrix"):
        """
        Generates and plots a confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(title)
        ax.xaxis.set_ticklabels(["False", "True"])
        ax.yaxis.set_ticklabels(["False", "True"])

    def generate_pdf_report(
        self,
        filepath,
        f_importances,
        f_names,
        input_frames,
        y_test_true,
        y_test_pred,
        y_train_true,
        y_train_pred,
    ):
        pdfmetrics.registerFont(TTFont("Arial", "Arial.ttf"))

        doc = SimpleDocTemplate(
            filepath,
            pagesize=landscape(letter),
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        styles = getSampleStyleSheet()
        story = []
        story.append(
            Paragraph(
                "Employee Attrition Prediction - Model Evaluation", styles["Title"]
            )
        )

        # Add Feature Importance
        story.append(Paragraph("Feature Importance", styles["Heading2"]))
        _fi_img = self.build_feature_importances(f_importances, f_names, 50)
        fi_img_path = "feature_importance.png"
        _fi_img.savefig(fi_img_path, bbox_inches="tight")
        story.append(Image(fi_img_path, height=4.2 * inch, width=5 * inch))

        # Add ROC Curve
        story.append(Paragraph("Receiver Operating Characteristic", styles["Heading2"]))
        _roc_img = self.build_roc_curve(input_frames=input_frames)
        roc_img_path = "roc_curve.png"
        _roc_img.savefig(roc_img_path, bbox_inches="tight")
        story.append(Image(roc_img_path, height=4 * inch, width=4 * inch))

        # Confusion Matrix for Train and Test
        story.append(Paragraph("Confusion Matrices", styles["Heading2"]))
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        self.build_confusion_matrix(
            y_train_true, y_train_pred, ax[0], "Confusion Matrix - Train"
        )
        self.build_confusion_matrix(
            y_test_true, y_test_pred, ax[1], "Confusion Matrix - Test"
        )
        cm_path = "confusion_matrix.png"
        fig.savefig(cm_path, bbox_inches="tight")
        story.append(Image(cm_path, height=3 * inch, width=6 * inch))

        # Add Performance Metrics
        story.append(Paragraph("Performance Metrics", styles["Heading2"]))

        # Training Metrics
        story.append(Paragraph("Train", styles["Heading3"]))
        train_metrics = self.generate_model_report(y_train_true, y_train_pred)
        train_data = [["Metric", "Value"]] + [[k, v] for k, v in train_metrics.items()]
        story.append(Table(train_data))

        # Test Metrics
        story.append(Paragraph("Test", styles["Heading3"]))
        test_metrics = self.generate_model_report(y_test_true, y_test_pred)
        test_data = [["Metric", "Value"]] + [[k, v] for k, v in test_metrics.items()]
        story.append(Table(test_data))

        doc.build(story)


if __name__ == "__main__":
    # Load the dataset and preprocess
    df = pd.read_excel("data/Worksheet_in_Case_Study.xlsx")
    df = preprocess_employee_data(df)

    # Define numeric and categorical features before the pipeline
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Train the model and get the results
    results = preprocess_and_train_model(df)

    X_train = results["X_train"]
    X_test = results["X_test"]
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_train_hat = results["y_train_hat"]
    y_test_hat = results["y_test_hat"]
    y_train_pred = results["y_train_pred"]
    y_test_pred = results["y_test_pred"]
    trained_pipeline = results["trained_pipeline"]

    # Log model success
    logging.info("Model training and prediction complete!")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # Save the trained pipeline using pickle
    pipeline_filename_pickle = f"trained_pipeline_xgb_{current_date}.pkl"
    with open(pipeline_filename_pickle, "wb") as file:
        pickle.dump(trained_pipeline, file)
    logging.info(f"Trained pipeline saved to {pipeline_filename_pickle} using pickle")

    # Ensure numeric and categorical features are defined based on X_train
    numeric_features = X_train.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Extract feature names from the preprocessor (ColumnTransformer)
    preprocessor = trained_pipeline.named_steps["preprocessor"]

    # Extract numeric and one-hot encoded feature names
    ohe_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(
        categorical_features
    )

    # Combine numeric and one-hot encoded feature names
    feature_names = np.hstack([numeric_features, ohe_feature_names])

    # Extract feature importances from the trained model
    feature_importances = trained_pipeline.named_steps[
        "estimator"
    ].base_estimator.feature_importances_

    # Match the selected feature names with the model's selected features
    feature_names_after_pca = feature_names[
        : len(feature_importances)
    ]  # Adjust for PCA or Lasso

    # Create a DataFrame for feature importance and feature names
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names_after_pca, "Importance": feature_importances}
    )

    # Sort the feature importances in descending order
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Plot the top features
    plt.figure(figsize=(15, 8))
    plt.barh(
        feature_importance_df["Feature"][:10],
        feature_importance_df["Importance"][:10],
        color="magenta",
    )
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Features by Importance")
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    plt.show()
    # Prepare input frames for the ROC curve
    input_frames = [[y_train, y_train_hat, "Train"], [y_test, y_test_hat, "Test"]]

    # Generate the PDF report
    evaluator = BinaryModelEvaluator()
    evaluator.generate_pdf_report(
        filepath=f"evaluation_report.pdf",
        f_importances=feature_importances,
        f_names=feature_names_after_pca,
        input_frames=input_frames,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        y_train_true=y_train,
        y_train_pred=y_train_pred,
    )

    logging.info("Evaluation report generated successfully!")
