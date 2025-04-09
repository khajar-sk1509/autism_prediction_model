import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
# --- Setup page ---
st.set_page_config(page_title="ASD Predictor", layout="wide")

# --- Utility: Set background dynamically ---
def set_background(image_path):
    import base64
    with open(image_path, "rb") as f:
        img = f.read()
    b64_img = base64.b64encode(img).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                          url("data:image/png;base64,{b64_img}");
        background-size: cover;
        background-position: right top;
        background-repeat: no-repeat;
        color: white;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Main toggle for switching views ---
view = st.radio("🔁 Switch View", ["🏠 Home", "📊 Results"], horizontal=True)

# === 🏠 HOME VIEW ===
if view == "🏠 Home":
    # --- Title and Instructions ---
    st.title("Autism Spectrum Disorder")
    st.title("(ASD) 🧠 Predictor")
    st.markdown("Answer the screening questions to receive a prediction by selecting one model.")

    # --- Dropdown on the LEFT below the heading ---
    left_col, _ = st.columns([1, 2])
    with left_col:
        model_choice = st.selectbox("🔽 Select Model", ["None", "Random Forest", "XGBoost", "SVC"])

    # --- Dynamic Background ---
    if model_choice == "None":
        set_background("asd.png")
    elif model_choice == "Random Forest":
        set_background("Random_forest.png")
    elif model_choice == "XGBoost":
        set_background("xgboost.png")
    elif model_choice == "SVC":
        set_background("SVC.png")

    # --- Layout and Prediction Interface ---
    col1, col2 = st.columns([2, 3])
    with col1:
        if model_choice != "None":
            model_files = {
                "Random Forest": "random_forest_model.pkl",
                "XGBoost": "xgboost_model.pkl",
                "SVC": "svc_model.pkl"
            }
            model = pickle.load(open(model_files[model_choice], "rb"))
            scaler = pickle.load(open("scaler.pkl", "rb"))

            st.subheader("📝 Screening Questions")

            questions = {
                "A1_Score": "A1: Do you often notice small sounds when others do not?",
                "A2_Score": "A2: Do you usually concentrate more on the whole picture rather than small details?",
                "A3_Score": "A3: Do you find it easy to do more than one thing at once?",
                "A4_Score": "A4: If there's an interruption, can you switch back to what you were doing very quickly?",
                "A5_Score": "A5: Do you find it easy to read between the lines when someone is talking to you?",
                "A6_Score": "A6: Do you know how to tell if someone listening to you is getting bored?",
                "A7_Score": "A7: When reading a story, do you find it difficult to understand the character's intention?",
                "A8_Score": "A8: Do you enjoy collecting information about categories of things (e.g., cars, birds, trains)?",
                "A9_Score": "A9: Is it easy for you to figure out how someone is feeling just by looking at their face?",
                "A10_Score": "A10: Do you find it difficult to work out people’s intentions?"
            }

            A_scores = []
            for key, question in questions.items():
                response = st.radio(f"{question}", ["Yes", "No"], horizontal=True, key=key)
                score = 1 if response == "Yes" else 0
                A_scores.append(score)

            age = st.slider("Age", 1, 100, 25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ethnicity = st.selectbox("Ethnicity", ["White-European", "Black", "Asian", "Others"])
            jaundice = st.radio("Born with jaundice?", ["Yes", "No"], horizontal=True)
            austim = st.radio("Family history of autism?", ["Yes", "No"], horizontal=True)
            country = st.selectbox("Country of residence", ["India", "USA", "UK", "Others"])
            relation = st.selectbox("Relation to person", ["Self", "Parent", "Guardian", "Other"])
            score_result = st.slider("Result from app test (0-20)", 0, 20, 10)

            gender_map = {"Male": 1, "Female": 0}
            ethnicity_map = {"White-European": 2, "Black": 1, "Asian": 0, "Others": 3}
            country_map = {"India": 1, "USA": 2, "UK": 3, "Others": 0}
            relation_map = {"Self": 2, "Parent": 0, "Guardian": 3, "Other": 1}

            input_data = A_scores + [
                age,
                gender_map[gender],
                ethnicity_map[ethnicity],
                1 if jaundice == "Yes" else 0,
                1 if austim == "Yes" else 0,
                country_map[country],
                score_result,
                relation_map[relation]
            ]

            columns = [
                'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                'age', 'gender', 'ethnicity', 'jaundice', 'austim',
                'contry_of_res', 'result', 'relation'
            ]

            df = pd.DataFrame([input_data], columns=columns)
            df['sum_score'] = sum(A_scores)
            df['age'] = np.log1p(df['age'])
            df = df[scaler.feature_names_in_]
            X_scaled = scaler.transform(df)

            if st.button("🔍 Predict"):
                prediction = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1]

                st.subheader("🧠 Prediction Result")
                if prediction == 1:
                    st.error("⚠️ Likely signs of ASD")
                else:
                    st.success("✅ No significant ASD indicators")

                st.metric("Model Confidence", f"{prob * 100:.2f}%")


# === 📊 RESULTS VIEW ===
else:
    st.title("📊 Model Performance Metrics")
    st.markdown("Below are the evaluation results for all trained models:")
    
    df = pd.read_csv("asd_model_metrics.csv")
    st.dataframe(df)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    model_names = df['Model'].tolist()
    bar_width = 0.18
    index = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        # Multiply by 100 to scale up to percentage
        values = df[metric].astype(float).tolist()
        values = [v * 100 if v <= 1 else v for v in values]
        ax.bar(index + i * bar_width, values, width=bar_width, label=metric)

    ax.set_xlabel("Algorithms", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("All Algorithms Performance Graph", fontsize=14)
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_ylim(0, 100)
    ax.legend(title="Metrics")
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    st.pyplot(fig)





