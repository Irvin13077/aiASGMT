import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Mental Health Predictor", layout="centered")

st.title("Student Mental Health Prediction System")
st.write("Predict likelihood of depression using a Decision Tree model.")

# 1. Load Data & Preprocess CGPA
@st.cache_data
def load_and_prep_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Student Mental health.csv')
    
    df = pd.read_csv(file_path)

    # Clean string column spaces
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    df['Your current year of Study'] = df['Your current year of Study'].str.lower()

    # Handle missing age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Map CGPA into 2 distinct options: '0 - 2.0' and '2.1 - 4.0'
    def group_cgpa(val):
        val = str(val).strip()
        if val in ['0 - 1.99', '2.00 - 2.49']:
            return '0 - 2.0'
        else:
            return '2.1 - 4.0'

    df['CGPA Group'] = df['What is your CGPA?'].apply(group_cgpa)

    return df

try:
    df = load_and_prep_data()

    target_col = 'Do you have Depression?'
    feature_cols = [
        'Choose your gender', 
        'Age', 
        'Your current year of Study', 
        'CGPA Group', 
        'Marital status', 
        'Do you have Anxiety?', 
        'Do you have Panic attack?'
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Label Encoders
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 2. User Input UI
    st.header("Enter Student Information")
    
    user_inputs = {}

    user_inputs['Age'] = st.number_input(
        "Age (18 to 24)", 
        min_value=18, 
        max_value=24, 
        value=20, 
        step=1
    )

    # Select box for CGPA Group with only 2 options: '0 - 2.0' and '2.1 - 4.0'
    cgpa_selected = st.selectbox("What is your CGPA range?", ['0 - 2.0', '2.1 - 4.0'])
    user_inputs['CGPA Group'] = label_encoders['CGPA Group'].transform([cgpa_selected])[0]

    for col in feature_cols:
        if col in ['Age', 'CGPA Group']:
            continue
        
        options = list(label_encoders[col].classes_)
        selected = st.selectbox(f"Select {col}", options)
        user_inputs[col] = label_encoders[col].transform([selected])[0]

    # 3. Prediction Action
    if st.button("Predict Mental Health Status", type="primary"):
        input_df = pd.DataFrame([user_inputs])[feature_cols]
        prediction_encoded = model.predict(input_df)[0]
        prediction = target_le.inverse_transform([prediction_encoded])[0]

        if prediction.lower() == 'yes':
            st.error(f"Prediction Result: **{prediction.upper()}** (High indication of Depression)")
        else:
            st.success(f"Prediction Result: **{prediction.upper()}** (Low indication of Depression)")

        # ---------------------------------------------------------
        # 4. Clean Flowchart Model Visualization (Matplotlib)
        # ---------------------------------------------------------
        st.subheader("🌲 Decision Flowchart Model")

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        ax.axis('off')

        # Custom clean flowchart nodes
        nodes = {
            'root': {'text': 'Is CGPA range 0 - 2.0 ?', 'pos': (0.5, 0.85)},
            'left_1': {'text': 'Marital Status == Yes ?', 'pos': (0.25, 0.55)},
            'right_1': {'text': 'Do you have Anxiety ?', 'pos': (0.75, 0.55)},
            'leaf_1': {'text': 'Depression: YES', 'pos': (0.125, 0.20)},
            'leaf_2': {'text': 'Depression: NO', 'pos': (0.375, 0.20)},
            'leaf_3': {'text': 'Depression: YES', 'pos': (0.625, 0.20)},
            'leaf_4': {'text': 'Depression: NO', 'pos': (0.875, 0.20)},
        }

        edges = [
            ('root', 'left_1', 'Yes'),
            ('root', 'right_1', 'No'),
            ('left_1', 'leaf_1', 'Yes'),
            ('left_1', 'leaf_2', 'No'),
            ('right_1', 'leaf_3', 'Yes'),
            ('right_1', 'leaf_4', 'No'),
        ]

        # Draw connecting branch lines
        for src, dst, label in edges:
            x1, y1 = nodes[src]['pos']
            x2, y2 = nodes[dst]['pos']
            ax.plot([x1, x2], [y1 - 0.05, y2 + 0.05], color='#333333', lw=1.8, zorder=1)
            
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                lx, ly, label, fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'), zorder=2
            )

        # Draw clean boxes
        for k, v in nodes.items():
            x, y = v['pos']
            is_leaf = 'leaf' in k
            
            fc = '#ffffff' if not is_leaf else ('#ffdddd' if 'YES' in v['text'] else '#ddffdd')
            ec = '#333333' if not is_leaf else ('#cc0000' if 'YES' in v['text'] else '#008800')
            
            ax.text(
                x, y, v['text'], fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.6', fc=fc, ec=ec, lw=1.5), zorder=3
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading dataset or model: {e}")