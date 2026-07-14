"""
Companion live-prediction app for the Power BI dashboard.

Power BI is excellent for exploring historical data and model *results*, but it
can't run a Python model interactively per user input. This small Streamlit app
fills that one gap: type in a hypothetical customer's details and get a live
Decision Tree prediction, using the exact model trained in the notebook.

Run with:
    streamlit run streamlit_predict_app.py
(from inside the Dashboard/ folder, with Model/ and Dataset/ as siblings of Notebook/)
"""

import pickle
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopping Behaviour - Live Prediction", layout="centered")

MODEL_PATH = "../Model/decision_tree_model.pkl"
ENCODERS_PATH = "../Model/label_encoders.pkl"
METRICS_PATH = "../Model/model_metrics.json"
DATA_PATH = "../Dataset/shopping_behavior_cleaned.csv"


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    df = pd.read_csv(DATA_PATH)
    return model, encoders, metrics, df


model, encoders, metrics, df = load_artifacts()

st.title("🛍️ Shopping Behaviour — High Value Customer Predictor")
st.caption(
    f"Decision Tree Classifier · Test accuracy {metrics['accuracy']:.1%} · "
    "predicts whether a customer profile matches the 'High Value Customer' segment "
    "(above-median spend + above-median repeat purchases)."
)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 70, 32)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)
        category = st.selectbox("Category", encoders["Category"].classes_)
    with col2:
        item = st.selectbox("Item Purchased", encoders["Item Purchased"].classes_)
        amount = st.number_input("Purchase Amount (USD)", min_value=1, max_value=500, value=50)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "Age": age,
        "Gender Encoded": encoders["Gender"].transform([gender])[0],
        "Category Encoded": encoders["Category"].transform([category])[0],
        "Item Purchased Encoded": encoders["Item Purchased"].transform([item])[0],
        "Purchase Amount (USD)": amount,
    }])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]

    if pred == 1:
        st.success(f"✅ Predicted: **High Value Customer** (confidence {proba[1]:.1%})")
    else:
        st.info(f"Predicted: **Standard Customer** (confidence {proba[0]:.1%})")

st.divider()
st.subheader("Model Performance")
c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Rows evaluated", sum(metrics["confusion_matrix"][0]) + sum(metrics["confusion_matrix"][1]))

st.write("**Feature importance**")
st.bar_chart(pd.Series(metrics["feature_importance"]))
