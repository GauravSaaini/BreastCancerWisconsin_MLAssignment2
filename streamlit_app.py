import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Breast Cancer Wisconsin Classification",
    layout="wide"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Evaluate Models",
        "Compare Models",
        "Dataset & Methodology"
    ]
)

if page == "Overview":

    st.title("Breast Cancer Wisconsin Classification")

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithms", "5")
    c2.metric("Metrics", "6")
    c3.metric("Retraining", "No")

    st.subheader("Project Overview")

    st.markdown("""
    ### Models Included

    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Gaussian Naive Bayes
    - Support Vector Machine
    """)

if page == "Dataset & Methodology":

    st.header("Dataset Information")

    st.write(
        "Breast Cancer Wisconsin Diagnostic Dataset"
    )

if page == "Compare Models":

    st.header("Model Comparison")

    file_path = Path("results/model_comparison.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)

        st.dataframe(df)

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

    else:
        st.error(
            "model_comparison.csv not found."
        )

if page == "Evaluate Models":

    st.header("Model Evaluation")

    st.info(
        "Upload a test dataset for evaluation."
    )

    st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )
