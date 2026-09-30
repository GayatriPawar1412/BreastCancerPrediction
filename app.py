import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎗️ Breast Cancer Prediction")
st.subheader("Machine Learning Research Prototype")

st.markdown(
    """
    This application demonstrates a machine-learning model
    for breast cancer classification using the Breast Cancer
    Wisconsin Diagnostic dataset.
    """
)

st.warning(
    "⚠️ Educational/Research Prototype only. "
    "This tool is NOT a medical diagnostic system and must not be "
    "used for real-world medical decisions."
)

st.divider()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

evaluation_model = model
evaluation_model.fit(X_train_scaled, y_train)

y_pred = evaluation_model.predict(X_test_scaled)
y_prob = evaluation_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


# =========================================================
# ROC CURVE
# =========================================================

st.subheader("📈 ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_data = {
    "False Positive Rate": fpr,
    "True Positive Rate": tpr
}

st.line_chart(
    roc_data,
    x="False Positive Rate",
    y="True Positive Rate"
)

st.caption(
    f"ROC-AUC Score: {roc_auc:.4f}"
)


# =========================================================
# MODEL PERFORMANCE DASHBOARD
# =========================================================
st.markdown(
    """
    ### 📊 Model Performance Dashboard
    The following metrics show the performance of the trained
    machine-learning model on the test dataset.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall / Sensitivity",
        f"{recall * 100:.2f}%"
    )


col4, col5 = st.columns(2)

with col4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📌 Confusion Matrix")

st.write(
    "The confusion matrix shows the actual and predicted "
    "classification results of the model."
)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)
st.subheader("🤖 Model Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("Dataset Samples", "569")

with info_col2:
    st.metric("Features", "30")

with info_col3:
    st.metric("Algorithm", "Logistic Regression")

st.info(
    "This project uses the Breast Cancer Wisconsin Diagnostic dataset "
    "and Logistic Regression for educational machine-learning research."
)
st.divider()


# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "Radius Error",
    "Texture Error",
    "Perimeter Error",
    "Area Error",
    "Smoothness Error",
    "Compactness Error",
    "Concavity Error",
    "Concave Points Error",
    "Symmetry Error",
    "Fractal Dimension Error",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension"
]
# =========================================================
# MODEL EXPLAINABILITY
# =========================================================

st.subheader("🔍 Model Explainability")

st.write(
    "The chart below shows the relative influence of each feature "
    "according to the Logistic Regression model."
)

feature_importance = np.abs(model.coef_[0])

importance_order = np.argsort(feature_importance)[::-1]

top_features = [
    feature_names[i]
    for i in importance_order[:10]
]

top_importance = [
    feature_importance[i]
    for i in importance_order[:10]
]

importance_data = {
    "Feature": top_features,
    "Importance": top_importance
}

st.bar_chart(
    importance_data,
    x="Feature",
    y="Importance"
)

st.caption(
    "Feature influence is based on the absolute Logistic Regression "
    "coefficient values."
)

# =========================================================
# PATIENT FEATURE INPUT
# =========================================================
st.markdown(
    """
    ### 👩‍⚕️ Patient Feature Input
    Enter the numerical feature values used by the machine-learning model.
    """
)

st.info(
    "Enter numerical values for the 30 features used by the "
    "Breast Cancer Wisconsin Diagnostic dataset."
)


# =========================================================
# DEMO PATIENT
# =========================================================

if st.button("📋 Load Demo Patient"):

    demo_data = load_breast_cancer()

    demo_values = demo_data.data[0]

    for i, feature in enumerate(feature_names):
        st.session_state[feature] = float(demo_values[i])

    st.success(
        "Demo patient data loaded successfully! ✅"
    )


# =========================================================
# FEATURE INPUT
# =========================================================

features = []

columns = st.columns(3)

for i, feature in enumerate(feature_names):

    with columns[i % 3]:

        value = st.number_input(
            feature,
            value=0.0,
            format="%.4f",
            key=feature
        )

        features.append(value)


st.divider()


# =========================================================
# PREDICTION
# =========================================================
st.markdown("---")

st.markdown(
    """
    ### 🔍 Run Prediction
    Click the button below to generate the model prediction
    from the entered feature values.
    """
)

if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array(features).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]


    # Prediction Result
    st.markdown("---")

    st.markdown(
       """
       ### 🎯 Prediction Result
       The model classification and probability scores are shown below.
       """
    )

    if prediction == 0:

        st.error(
            "🔴 Prediction: Malignant"
        )

    else:

        st.success(
            "🟢 Prediction: Benign"
        )


    # Model Probability

    st.markdown(
       """
       ### 📊 Prediction Probability
       The probabilities below represent the model's estimated
       classification probabilities for the entered features.
       """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Malignant Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Benign Probability",
            f"{probabilities[1] * 100:.2f}%"
        )


    # =====================================================
    # PREDICTION PROBABILITY CHART
    # =====================================================
    st.markdown(
        """
        #### 📊 Probability Visualization
        The chart below visualizes the model's predicted probabilities.
        """
    )

    probability_data = {
        "Malignant": [
            probabilities[0] * 100
        ],
        "Benign": [
            probabilities[1] * 100
        ]
    }

    st.bar_chart(
        probability_data
    )

 # =========================================================
# DOWNLOAD PREDICTION REPORT
# =========================================================

    report = f"""
Breast Cancer Prediction
========================

Prediction: {"Malignant" if prediction == 0 else "Benign"}

Malignant Probability: {probabilities[0] * 100:.2f}%
Benign Probability: {probabilities[1] * 100:.2f}%

Model: Logistic Regression
Dataset: Breast Cancer Wisconsin Diagnostic Dataset

IMPORTANT:
This is an educational/research prototype.
It is NOT a clinically validated diagnostic system.
This prediction must not be used for medical decisions.
"""

    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="breast_cancer_prediction_report.txt",
        mime="text/plain"
    )

# =========================================================
# RESEARCH NOTES & LIMITATIONS
# =========================================================

st.divider()

st.subheader("📝 Research Notes & Limitations")

st.markdown(
    """
    - This application is an educational and research prototype.
    - It is not a clinically validated diagnostic system.
    - Model performance is evaluated on a test split of the dataset.
    - Performance on this dataset does not represent real-world clinical accuracy.
    - The prediction should not be used for medical diagnosis or treatment decisions.
    - Real-world clinical use would require appropriate clinical validation,
      regulatory review, external validation, and expert oversight.
    """
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Dataset: Breast Cancer Wisconsin Diagnostic Dataset | "
    "Model: Logistic Regression"
)