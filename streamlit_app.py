import streamlit as st
import pandas as pd
import numpy as np
"""
ML Assignment 2
Breast Cancer Wisconsin Diagnostic Dataset

Models:
1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. SVM

Outputs:
- Saved Models
- Test Dataset CSV
- Model Comparison CSV
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ==================================================
# PATHS
# ==================================================

ROOT = Path.cwd()

MODEL_DIR = ROOT / "model"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# DATASET PATH
# ==================================================

DATASET_PATH = Path(
    DATASET_PATH = ROOT / "wdbc.csv"
)


# ==================================================
# DATASET COLUMNS
# ==================================================

columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean",
    "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se",
    "perimeter_se", "area_se",
    "smoothness_se", "compactness_se",
    "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def score_for_auc(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def calculate_metrics(model_name, y_true, y_pred, y_score):

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_score), 4),
        "Precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "F1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4)
    }


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    # Check dataset path
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Read dataset
    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        sep=","
    )

    df.columns = columns

    print("Dataset Shape:", df.shape)

    print("\nDataset Information:")
    df.info()

    print("\nFirst 3 Rows:")
    print(df.head(3))

    # ==================================================
    # TARGET CLEANING
    # ==================================================

    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("\nOriginal Diagnosis Values:")
    print(df["diagnosis"].unique())

    # B = 0, M = 1
    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    if df["diagnosis"].isna().any():
        raise ValueError("Invalid diagnosis values found.")

    print("\nEncoded Diagnosis Values:")
    print(df["diagnosis"].unique())

    # ==================================================
    # FEATURES AND TARGET
    # ==================================================

    X = df.drop(columns=["id", "diagnosis"]).copy()
    y = df["diagnosis"]

    # Convert features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    print("\nMissing Values:")
    print(X.isnull().sum().sum())

    # ==================================================
    # TRAIN TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # ==================================================
    # SAVE TEST DATASET
    # ==================================================

    test_df = X_test.copy()
    test_df["diagnosis"] = y_test

    test_df.to_csv(
        RESULT_DIR / "test_data.csv",
        index=False
    )

    print("\nSaved: test_data.csv")

    # ==================================================
    # MODELS
    # ==================================================

    models = {

        "logistic_regression": (
            "Logistic Regression",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=5000,
                    random_state=42
                ))
            ])
        ),

        "decision_tree": (
            "Decision Tree",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(
                    random_state=42
                ))
            ])
        ),

        "knn": (
            "KNN",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(
                    n_neighbors=5
                ))
            ])
        ),

        "naive_bayes": (
            "Naive Bayes",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GaussianNB())
            ])
        ),

       
        "svm": (
            "Support Vector Machine",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(
                    estimator=SVC(
                        kernel="rbf",
                        random_state=42
                    ),
                    method="sigmoid",
                    cv=5,
                    ensemble=False
                ))
            ])
        )
    }

    # ==================================================
    # TRAINING AND EVALUATION
    # ==================================================

    results = []

    for file_name, (display_name, model) in models.items():

        print("\n" + "=" * 60)
        print(f"Training: {display_name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_score = score_for_auc(
            model,
            X_test
        )

        metrics = calculate_metrics(
            display_name,
            y_test,
            y_pred,
            y_score
        )

        results.append(metrics)

        # Save trained model
        model_file = MODEL_DIR / f"{file_name}.joblib"

        joblib.dump(model, model_file)

        print("Saved Model:", model_file.name)
        print("Metrics:", metrics)

    # ==================================================
    # SAVE MODEL COMPARISON
    # ==================================================

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="F1",
        ascending=False
    )

    comparison_file = RESULT_DIR / "model_comparison.csv"

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(comparison_df.to_string(index=False))

    print("\n")
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print("Models Folder :", MODEL_DIR)
    print("Results Folder:", RESULT_DIR)

    print("\nGenerated Files:")
    print("- logistic_regression.joblib")
    print("- decision_tree.joblib")
    print("- knn.joblib")
    print("- naive_bayes.joblib")
    print("- svm.joblib")
    print("- test_data.csv")
    print("- model_comparison.csv")

    print("\nExecution Completed Successfully!")


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()
