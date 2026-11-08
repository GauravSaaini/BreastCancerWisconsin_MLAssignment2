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
