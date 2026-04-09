import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model

from optimizer import optimize_mix
from groq_advisor import biogas_ai_advisor

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Biogas AI", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0f172a;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    font-size: 36px;
    color: #22c55e;
}

/* Inputs */
.stNumberInput input {
    background-color: #1e293b;
    color: white;
    border: 1px solid #334155;
}

/* Button */
.stButton>button {
    background-color: #22c55e;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}

/* Cards */
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

/* Titles */
.metric-title {
    font-size: 20px;
    font-weight: 700;
    color: #e5e7eb;
}

/* Values */
.metric-value {
    font-size: 34px;
    font-weight: 800;
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌱 Smart Biogas Optimization Dashboard")

# ---------------- LOAD ----------------
model = load_model("model.h5", compile=False)
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("dataset_waste.csv")

features = [
    "Pig Manure (kg)", "Kitchen Food Waste (kg)", "Chicken Litter (kg)",
    "Cassava (kg)", "Bagasse Feed (kg)", "Energy Grass (kg)",
    "Banana Shafts (kg)", "Alcohol Waste (kg)", "Municipal Residue (kg)",
    "Fish Waste (kg)", "Water (L)", "Temperature (C)", "Humidity (%)",
    "C/N Ratio", "Digester Temp (C)"
]

# ---------------- FORMAT FUNCTION ----------------
def format_advice(text):
    text = text.replace("1. Performance Rating:", "\n\n### 1. Performance Rating\n")
    text = text.replace("2. Critical Issues:", "\n\n### 2. Critical Issues\n")
    text = text.replace("3. Optimization Strategy:", "\n\n### 3. Optimization Strategy\n")
    text = text.replace("4. Real-World Recommendations:", "\n\n### 4. Real-World Recommendations\n")

    text = text.replace("- ", "\n- ")
    text = text.replace("\n\n###", "\n\n<br>\n\n###")

    return text

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Input Parameters")

user_inputs = {}
for f in features:
    user_inputs[f] = st.sidebar.number_input(f, value=float(df[f].mean()))

use_ai = st.sidebar.checkbox("Enable AI Advisor")
run = st.sidebar.button("🚀 Run Analysis")

# ---------------- MAIN ----------------
if run:

    input_df = pd.DataFrame([user_inputs])[features]
    scaled = scaler.transform(input_df)

    pred = model.predict(scaled, verbose=0)
    biogas = float(pred[0][0])

    col1, col2 = st.columns(2)

    # Predicted
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">🔥 Predicted Biogas</div>
            <div class="metric-value">{biogas:.2f} m³</div>
        </div>
        """, unsafe_allow_html=True)

    # Optimizer
    with st.spinner("Optimizing..."):
        mix, best = optimize_mix(model, scaler, user_inputs)

    # Optimized
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">🚀 Optimized Biogas</div>
            <div class="metric-value">{best:.2f} m³</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Optimized Waste Distribution")
    st.json(mix)

    # ---------------- AI ----------------
    if use_ai:
        st.subheader("🤖 AI Advisor")

        with st.spinner("Analyzing..."):
            advice = biogas_ai_advisor(user_inputs, best)

        formatted = format_advice(advice)

        st.markdown(f"""
        <div class="card">
        {formatted}
        </div>
        """, unsafe_allow_html=True)

    # ---------------- CHARTS ----------------
    st.subheader("📊 Data Insights")

    df["Total Waste"] = df[features[:10]].sum(axis=1)

    fig1 = px.scatter(df, x="Total Waste", y="biogas_production",
                      title="Total Waste vs Biogas")

    fig2 = px.imshow(df.corr(numeric_only=True),
                     text_auto=True,
                     title="Feature Correlation")

    comp = pd.DataFrame({
        "Type": ["Predicted", "Optimized"],
        "Biogas": [biogas, best]
    })

    fig3 = px.bar(comp, x="Type", y="Biogas",
                  title="Prediction vs Optimization")

    for fig in [fig1, fig2, fig3]:
        fig.update_layout(
            template="plotly_dark",
            font=dict(color="white")
        )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.plotly_chart(fig3, use_container_width=True)