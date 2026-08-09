import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title="Student Mental Health Predictor", layout="wide")

st.title("Student Mental Health Prediction System")
st.write("Predict likelihood of depression using a trained Machine Learning Decision Tree model.")

# --- Helper Functions ---
def parse_year(val):
    if pd.isna(val):
        return 1
    import re
    numbers = re.findall(r"\d+", str(val))
    return int(numbers[0]) if numbers else 1

def encode_binary(val):
    return 1 if str(val).strip().lower() == "yes" else 0

# --- Model Training (scikit-learn) ---
@st.cache_resource
def train_decision_tree_model():
    """
    Creates sample dataset (or load your real CSV here) and trains 
    a real scikit-learn Decision Tree model.
    """
    # Replace this block with pd.read_csv('your_dataset.csv') if you have a CSV file
    np.random.seed(42)
    n_samples = 200
    
    sample_data = {
        'age': np.random.randint(18, 26, size=n_samples),
        'cgpa': np.round(np.random.uniform(1.5, 4.0, size=n_samples), 2),
        'year': np.random.randint(1, 5, size=n_samples),
        'marital': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'anxiety': np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30]),
        'panic': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
        'treatment': np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20]),
    }
    
    df = pd.DataFrame(sample_data)
    
    # Target rule simulation for synthetic dataset training
    # High anxiety/panic/low CGPA increases probability of Depression label
    depression_score = (
        (df['cgpa'] <= 2.2).astype(int) * 2 +
        df['anxiety'] * 2 +
        df['panic'] * 2 +
        df['treatment'] * 1 +
        (df['year'] >= 3).astype(int) * 1
    )
    df['depression'] = (depression_score >= 3).astype(int)
    
    # Define features (X) and target (y)
    feature_cols = ['age', 'cgpa', 'year', 'marital', 'anxiety', 'panic', 'treatment']
    X = df[feature_cols]
    y = df['depression']
    
    # Train the Machine Learning Model
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    
    return clf, feature_cols

# Train model once and cache
model, feature_names = train_decision_tree_model()

# --- User Input UI ---
st.header("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=30, value=20)
    cgpa = st.number_input(
        "Enter CGPA (0.00 to 4.00)", 
        min_value=0.0, 
        max_value=4.0, 
        value=3.25, 
        step=0.01, 
        format="%.2f"
    )
    year_str = st.selectbox("Year of Study", ["Year 1", "Year 2", "Year 3", "Year 4"], index=0)
    marital = st.selectbox("Marital Status", ["No", "Yes"])

with col2:
    anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
    panic = st.selectbox("Do you have Panic attack?", ["No", "Yes"])
    treatment = st.selectbox("Did you seek Specialist Treatment?", ["No", "Yes"])

if st.button("Predict Mental Health Status", type="primary"):
    # Format inputs for model evaluation
    input_data = pd.DataFrame([{
        'age': age,
        'cgpa': cgpa,
        'year': parse_year(year_str),
        'marital': encode_binary(marital),
        'anxiety': encode_binary(anxiety),
        'panic': encode_binary(panic),
        'treatment': encode_binary(treatment)
    }])[feature_names]

    # Make ML Prediction
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Prediction Result: **DEPRESSION DETECTED** (Probability: {prediction_prob:.0%})")
    else:
        st.success(f"Prediction Result: **NO DEPRESSION DETECTED** (Probability: {prediction_prob:.0%})")

    st.subheader("Trained Decision Tree Visualization")
    st.write("This diagram represents the actual decision rules learned automatically by the model from data:")

    # Render actual trained Tree from scikit-learn
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        model, 
        feature_names=feature_names, 
        class_names=["No Depress", "Depress"], 
        filled=True, 
        rounded=True, 
        fontsize=8,
        ax=ax
    )
    st.pyplot(fig)