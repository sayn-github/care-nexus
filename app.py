import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="CareNexus",
    layout="wide",
    page_icon="🧑‍⚕️HealthSuiteAI"
)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'CareNexus',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart'],
        default_index=0
    )

# ================= Diabetes Prediction Page =================
if selected == 'Diabetes Prediction':
    # Load model and transformer
    diab_model = joblib.load('trained_model.pkl')
    transformer = joblib.load('quantile_transformer.pkl')

    # App title
    st.title("🩺 Real-Time Diabetes Prediction")
    st.write("Enter patient details below to get a prediction.")

    # Input form
    # Input form with 3 columns
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)

        with col2:
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=171)

        with col3:
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=33)

        col4, col5, _ = st.columns(3)

        with col4:
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=33.3)

        with col5:
            age = st.number_input("Age", min_value=1, max_value=120, value=24)

        submitted = st.form_submit_button("Predict")


    if submitted:
        input_df = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'SkinThickness': [skin_thickness],
            'BMI': [bmi],
            'Age': [age],
        })

        transformed_input = transformer.transform(input_df)
        prediction = diab_model.predict(transformed_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes.")
        else:
            st.success("✅ Low risk of Diabetes.")

# ================= Heart Disease Prediction Page =================
elif selected == 'Heart Disease Prediction':
    # Load model and scaler
    heart_model = joblib.load('best_heart_attack_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Page title
    st.title('❤️ Heart Disease Prediction using ML')

    def predict_heart_attack(features):
        features_scaled = scaler.transform([features])
        return heart_model.predict(features_scaled)[0]

    # Input fields in 3-column layout
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], index=0)
        oldpeak = st.number_input("ST Depression by Exercise", min_value=0.0, value=1.1)
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], index=0)

    with col2:
        sex = st.selectbox("Sex", [0, 1], index=0, format_func=lambda x: "Female" if x == 0 else "Male")
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, value=244)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, value=162)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2], index=0)

    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=0)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], index=0)
        ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3], index=0)

    # Predict button
    if st.button("Predict"):
        features = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]
        prediction = predict_heart_attack(features)

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

# import streamlit as st
# from streamlit_option_menu import option_menu
# import numpy as np
# import streamlit as st
# import joblib

# # Set page configuration

# st.set_page_config(page_title="CareNexus",
#                    layout="wide",
#                    page_icon="🧑‍⚕️HealthSuiteAI")

    
# # sidebar for navigation
# with st.sidebar:
#     selected = option_menu('CareNexus',

#                            ['Diabetes Prediction',
#                             'Heart Disease Prediction'],
#                            menu_icon='hospital-fill',
#                            icons=['activity', 'heart'],
#                            default_index=0)

# #Diabetes Prediction Page
# if selected == 'Diabetes Prediction':

#     # Load the saved model and transformer
#     diab_model = joblib.load('trained_model.pkl')
#     transformer = joblib.load('quantile_transformer.pkl')

#     # App title
#     st.title("🩺 Real-Time Diabetes Prediction")
#     st.write("Enter patient details below to get a prediction.")

#     # Input form
#     with st.form("prediction_form"):
#         pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)
#         glucose = st.number_input("Glucose", min_value=0, max_value=300, value=171)
#         skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=33)
#         bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=33.3)
#         age = st.number_input("Age", min_value=1, max_value=120, value=24)

#         submitted = st.form_submit_button("Predict")

#     # Make prediction on form submission
#     if submitted:
#         input_df = pd.DataFrame({
#             'Pregnancies': [pregnancies],
#             'Glucose': [glucose],
#             'SkinThickness': [skin_thickness],
#             'BMI': [bmi],
#             'Age': [age],
#         })

#         # Transform the input
#         transformed_input = transformer.transform(input_df)

#         # Predict
#         prediction = diab_model.predict(transformed_input)[0]

#         # Output result
#         if prediction == 1:
#             st.error("The model predicts a high risk of diabetes.")
#         else:
#             st.success("The model predicts a low risk of diabetes.")


# # # Heart Disease Prediction Page
# if selected == 'Heart Disease Prediction':

#     # page title
#     st.title('Heart Disease Prediction using ML')

#     col1, col2, col3 = st.columns(3)
    
#     # Load the saved model and scaler
#     heart_model = joblib.load('best_heart_attack_model.pkl')
#     scaler = joblib.load('scaler.pkl')

#     # Define the prediction function
#     def predict_heart_attack(features):
#         features_scaled = scaler.transform([features])
#         prediction = heart_model.predict(features_scaled)[0]
#         return prediction

#     # ===== Input rows with 3 fields each =====
#     row1 = st.columns(3)
#     with row1[0]:
#         age = st.number_input("Age", min_value=1, max_value=100, value=50)
#     with row1[1]:
#         sex = st.selectbox("Sex", [0, 1], index=0, format_func=lambda x: "Female" if x == 0 else "Male")
#     with row1[2]:
#         cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)

#     row2 = st.columns(3)
#     with row2[0]:
#         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
#     with row2[1]:
#         chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, value=244)
#     with row2[2]:
#         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=0)

#     row3 = st.columns(3)
#     with row3[0]:
#         restecg = st.selectbox("Resting ECG Results", [0, 1, 2], index=0)
#     with row3[1]:
#         thalach = st.number_input("Max Heart Rate Achieved", min_value=0, value=162)
#     with row3[2]:
#         exang = st.selectbox("Exercise Induced Angina", [0, 1], index=0)

#     row4 = st.columns(3)
#     with row4[0]:
#         oldpeak = st.number_input("ST Depression by Exercise", min_value=0.0, value=1.1)
#     with row4[1]:
#         slope = st.selectbox("Slope of ST Segment", [0, 1, 2], index=0)
#     with row4[2]:
#         ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3], index=0)

#     row5 = st.columns(3)
#     with row5[0]:
#         thal = st.selectbox("Thalassemia", [0, 1, 2, 3], index=0)

#     # ===== Prediction button =====
#     if st.button("Predict"):
#         features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
#                     exang, oldpeak, slope, ca, thal]
#         prediction = predict_heart_attack(features)

#         if prediction == 1:
#             st.error("High Risk of Heart Attack")
#         else:
#             st.success("Low Risk of Heart Attack")

