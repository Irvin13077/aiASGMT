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
    
    depression_score = (
        (df['cgpa'] <= 2.2).astype(int) * 2 +
        df['anxiety'] * 2 +
        df['panic'] * 2 +
        df['treatment'] * 1 +
        (df['year'] >= 3).astype(int) * 1
    )
    df['depression'] = (depression_score >= 3).astype(int)
    
    feature_cols = ['age', 'cgpa', 'year', 'marital', 'anxiety', 'panic', 'treatment']
    X = df[feature_cols]
    y = df['depression']
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    
    return clf, feature_cols

# Train model once and cache
model, feature_names = train_decision_tree_model()

# --- User Input UI ---
st.header("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=30, value=22)
    cgpa = st.number_input(
        "Enter CGPA (0.00 to 4.00)", 
        min_value=0.0, 
        max_value=4.0, 
        value=1.50, 
        step=0.01, 
        format="%.2f"
    )
    year_str = st.selectbox("Year of Study", ["Year 1", "Year 2", "Year 3", "Year 4"], index=2)
    marital = st.selectbox("Marital Status", ["No", "Yes"], index=1)

with col2:
    anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"], index=0)
    panic = st.selectbox("Do you have Panic attack?", ["No", "Yes"], index=0)
    treatment = st.selectbox("Did you seek Specialist Treatment?", ["No", "Yes"], index=0)

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

    st.subheader("Decision Flow Path Visualization")
    st.write("Left branch = **TRUE / YES**, Right branch = **FALSE / NO**.")
    st.write("Active evaluation path is highlighted with a **thick red border**:")

    # 1. Get exact active node indices from scikit-learn tree engine
    node_indicator = model.decision_path(input_data)
    active_node_ids = set(node_indicator.indices)

    # 2. Render base plot
    fig, ax = plt.subplots(figsize=(16, 8))
    annotations = plot_tree(
        model, 
        feature_names=feature_names, 
        class_names=["No Depress", "Depress"], 
        filled=True, 
        rounded=True, 
        fontsize=8,
        ax=ax
    )

    # 3. Map annotations using actual node indices derived from text content
    # Extract sample counts or node values directly to ensure accurate indexing
    for annotation in annotations:
        text = annotation.get_text()
        bbox_patch = annotation.get_bbox_patch()
        
        # Determine node identity by testing if its printed attributes match active nodes
        is_active = False
        for node_id in active_node_ids:
            n_samples = model.tree_.n_node_samples[node_id]
            val = model.tree_.value[node_id]
            
            # Match text values rendered in node box
            if f"samples = {n_samples}" in text:
                # Disambiguate if duplicate sample numbers exist
                val_str = f"value = [{int(val[0][0])}, {int(val[0][1])}]"
                if val_str in text or f"value = {val[0].tolist()}" in text:
                    is_active = True
                    break

        if is_active:
            if bbox_patch is not None:
                bbox_patch.set_edgecolor("#FF0000")  # Red border
                bbox_patch.set_linewidth(3.5)
                bbox_patch.set_alpha(1.0)
            annotation.set_weight("bold")
            annotation.set_alpha(1.0)
        else:
            if bbox_patch is not None:
                bbox_patch.set_alpha(0.15)
            annotation.set_alpha(0.15)

    st.pyplot(fig)