from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

st.set_page_config(
    page_title="Breast Cancer Wisconsin Classification",
    layout="wide"
)

with st.sidebar:

    st.markdown("## 🎗️ Breast Cancer")
    st.caption("ML Assignment 2")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Evaluate Models",
            "Compare Models",
            "Dataset & Methodology"
        ]
    )

    st.divider()

    st.markdown("### Test Dataset")

    uploaded_file = st.file_uploader(
        "Upload a labelled test CSV",
        type=["csv"]
    )

    st.caption(
        "Expected target column: diagnosis"
    )

    st.divider()

    st.caption(
        "Models are loaded from saved joblib pipelines."
    )
if page == "Overview":

    st.markdown("""
    <div style="
    background:linear-gradient(90deg,#4FC3F7,#81D4FA);
    padding:25px;
    border-radius:15px;
    color:white;
    margin-bottom:15px;
    ">
    <h1>Breast Cancer Wisconsin Classification</h1>
    <p>
    Evaluate five machine-learning classifiers for breast cancer diagnosis using a labelled dataset.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithms", "5")
    c2.metric("Evaluation Metrics", "6")
    c3.metric("Retraining in app", "No")

    st.subheader("Project at a glance")

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown("""
        ### Objective

        Predict whether a tumour is:

        - Benign (0)
        - Malignant (1)

        using machine learning classification algorithms.

        ### Models included

        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Support Vector Machine
        """)

    with right:

        st.markdown("""
        ### Evaluation workflow

        1. Load dataset
        2. Train models
        3. Evaluate models
        4. Compare performance
        5. Save results
        """)

        st.info(
            "All models are evaluated using the same train-test split."
        )

    st.subheader("Current Dataset")

    try:

        df = pd.read_csv(
            "wdbc.data",
            header=None
        )

        st.write(
            f"{df.shape[0]:,} rows · {df.shape[1]} columns"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Dataset could not be loaded: {e}"
        )

if page == "Dataset & Methodology":

    st.header("Dataset and Methodology")

    st.markdown("""
    ## Dataset

    Breast Cancer Wisconsin Diagnostic Dataset

    ## Objective

    Predict whether a tumor is:

    - Benign (0)
    - Malignant (1)

    ## Dataset Characteristics

    - Total Samples: 569
    - Total Features: 30
    - Target Classes: 2

    ## Data Preparation

    - Median Imputation
    - Standard Scaling
    - Train Test Split (80/20)
    - Random State = 42

    ## Models Evaluated

    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Support Vector Machine

    ## Evaluation Metrics

    - Accuracy
    - ROC AUC
    - Precision
    - Recall
    - F1 Score
    - MCC
    """)

if page == "Compare Models":

    st.header("Model Comparison")

    file_path = Path("results/model_comparison.csv")

    if file_path.exists():

        df = pd.read_csv(file_path)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.success(
            "🏆 Best Model: Support Vector Machine "
            "(Accuracy = 97.37%, F1 = 0.9639, MCC = 0.9433)"
        )

        numeric_cols = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]

        st.bar_chart(
            df.set_index("Model")[numeric_cols]
        )

        st.download_button(
            "Download Comparison CSV",
            df.to_csv(index=False),
            "model_comparison.csv",
            "text/csv"
        )

    else:

        st.error(
            "model_comparison.csv not found."
        )

if page == "Evaluate Models":

    st.header("Model Evaluation")

    model_name = st.selectbox(
        "Choose a classifier",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "SVM"
        ]
    )

    if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    model = joblib.load(
        Path("model") /
        MODEL_FILES[model_name]
    )

    X = test_df.drop(
        columns=["diagnosis"]
    )

    y_true = test_df["diagnosis"]

    y_pred = model.predict(X)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    st.success(
        f"Loaded {len(test_df)} test records."
    )

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.4f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.4f}"
    )

    c4.metric(
        "F1",
        f"{f1:.4f}"
    )

    c5.metric(
        "MCC",
        f"{mcc:.4f}"
    )

        tab1, tab2, tab3 = st.tabs(
            [
                "Confusion Matrix",
                "Classification Report",
                "Test Data Preview"
            ]
        )

     with tab1:

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Benign",
            "Malignant"
        ]
    )

    disp.plot(
        ax=ax,
        cmap="Blues"
    )

    st.pyplot(fig)

        with tab2:

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).T

    st.dataframe(
        report_df,
        use_container_width=True
    )


        with tab3:

            st.dataframe(
                test_df.head(100),
                use_container_width=True
            )

    else:

        st.warning(
            "Upload a CSV file using the sidebar."
        )
