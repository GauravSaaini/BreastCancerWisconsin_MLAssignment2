BreastCancerWisconsin_MLAssignment2
Breast Cancer Wisconsin Diagnostic Classification Using Machine Learning
a. Problem Statement

The objective of this project is to build and compare machine learning models that classify breast tumors as Benign (B) or Malignant (M) using diagnostic measurements from breast tissue samples.

This is a binary classification problem where:

B = Benign (non-cancerous)
M = Malignant (cancerous)

Five classification algorithms are evaluated and compared using multiple performance metrics.

b. Dataset Description

Dataset: Breast Cancer Wisconsin Diagnostic Dataset

Source: UCI Machine Learning Repository

URL:
 https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

The dataset contains 569 patient records and 30 numerical features computed from digitized images of breast mass biopsies.

Target Variable: diagnosis

B = Benign
M = Malignant

Features include measurements such as:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry
c. Data Preprocessing

The following preprocessing steps were applied:

Removed the ID column
Encoded diagnosis:
B → 0
M → 1
Missing values handled using median imputation
Standardized features using StandardScaler
Split data into:
Training Set: 80%
Testing Set: 20%
Random State = 42
d. Models Implemented
Logistic Regression
Decision Tree
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Support Vector Machine (SVM)
e. Evaluation Metrics

The models were evaluated using:

Accuracy
ROC-AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
f. Streamlit Application Features
Overview Dashboard
Model Evaluation
Model Comparison
Dataset & Methodology
Dataset Preview
Interactive Charts
CSV Upload Support
Download Results
g. Conclusion

This project demonstrates the use of machine learning techniques for breast cancer diagnosis. Multiple classification algorithms were compared, and performance was assessed using several evaluation metrics to identify the most effective model for predicting malignant and benign tumors.
