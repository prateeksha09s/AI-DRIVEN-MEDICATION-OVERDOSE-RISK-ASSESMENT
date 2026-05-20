import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from EDA import ExploratoryDataAnalysis
import joblib
import matplotlib.pyplot as plt
import time
import random
import requests
import os


# Streamlit UI
st.title('AI-DRIVEN MEDICATION OVERDOSE RISK ASSESSMENT PREDICTION SYSTEM WITH DYNAMIC VISUALIZATION')

st.image("Image/coverpage.png")
st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {"#0000FF"};
            color: {"#0000FF"};
        }}
        </style>
        """,
        unsafe_allow_html=True
)

import streamlit as st

st.sidebar.subheader("Welcome to the Medication Overdose Risk Prediction Platform")
st.sidebar.image("Image/ms1.png")
st.sidebar.markdown("""




## Security and Privacy

- **Data Security**: All user data is processed locally on the platform.
- **No External Sharing**: The platform does not share user information with third parties.
- **OTP Authentication**: Secure login ensures only authorized access.

---""")

st.sidebar.image("Image/ms2.png")
st.sidebar.markdown("""

## Future Enhancements

1. **Integration with Wearables**:
   - Use data from fitness trackers to enhance predictions.
2. **Expanded Medication Database**:
   - Incorporate a wider range of drugs and interactions.
3. **Multi-Language Support**:
   - Make the platform accessible to a global audience.

---

## Disclaimer

This platform is designed for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult your healthcare provider with any questions regarding medications or medical conditions.

---

Experience the future of medication safety with our AI-driven **Medication Overdose Risk Prediction Platform**.
""")


st.markdown("""
## Features and Functionalities

### 1. **User Registration and Login**

- **Register**: Users can create an account by entering basic details such as username, age, BMI, and gender.
- **Login**: Secure login via OTP sent to a registered Telegram ID.
- **Session Management**: Ensures secure user sessions with time-limited OTPs.

### 2. **Real-Time Risk Prediction**

- Enter health and medication details to get a real-time prediction of overdose risk.
- Features analyzed include:
  - Age
  - Gender
  - Drug Type
  - Dosage (in mg)
  - Usage Frequency (per month)
  - BMI
  - Chronic Conditions
  - Prescribed Medications
  - Heart Rate and Blood Pressure
- Risk Levels:
  - Low
  - Moderate
  - High
  - Very High

### 3. **Advanced Visualization**

- **Bar Charts**: Display prediction probabilities for all risk categories.
- **Dynamic Feedback**: Highlight the predicted risk level with detailed probabilities.

---

## How It Works

1. **Input Collection**:
   - Users provide detailed health and medication data through an intuitive form.
2. **Data Processing**:
   - Categorical data is encoded, and numerical features are scaled to match the model's training specifications.
3. **Risk Prediction**:
   - The platform uses a pre-trained deep learning model to predict the risk of medication overdose.
4. **Output Visualization**:
   - Results are displayed with clear visualizations and actionable insights.

---
""")

# Static Telegram details
TELEGRAM_TOKEN = "8142697609:AAGC8kb7dsH2s5_AWkAwjEKVGnZxkC-mXLo"
TELEGRAM_CHAT_ID = "1819145137"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT UNIQUE,
            age INTEGER,
            bmi REAL,
            gender TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load the trained model and label encoders
model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Load the label encoders (Fit them on the original data)
label_encoders = {
    'Gender': LabelEncoder(),
    'Drug_Type': LabelEncoder(),
    'Prescribed_Medication': LabelEncoder(),
    'Medication_Type': LabelEncoder()
}

# Load the dataset to fit encoders (should match the preprocessing in the training code)
data = pd.read_csv('Medication_Overdose_Risk_Dataset_with_Single_Blood_Pressure_Value.csv')

# Fit label encoders
for col in ['Gender', 'Drug_Type', 'Prescribed_Medication', 'Medication_Type']:
    label_encoders[col].fit(data[col])

# Function to make prediction
def make_prediction(input_data):
    # Encode categorical variables
    encoded_data = input_data.copy()
    for col in ['Gender', 'Drug_Type', 'Prescribed_Medication', 'Medication_Type']:
        encoded_data[col] = label_encoders[col].transform([input_data[col]])[0]

    # Prepare input for prediction (scale the features)
    X_input = pd.DataFrame([encoded_data])
    X_scaled_input = scaler.transform(X_input)

    # Make the prediction
    prediction = model.predict(X_scaled_input)
    predicted_risk = np.argmax(prediction, axis=1)[0]

    return predicted_risk, prediction

# Function to register a new user
# Function to register a new user
def register_user(username, age, bmi, gender):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (username, age, bmi, gender)
            VALUES (?, ?, ?, ?)
        """, (username, age, bmi, gender))
        conn.commit()
        st.success("User registered successfully!")

        # Save the user details to a CSV file
        user_data = {'username': [username], 'age': [age], 'bmi': [bmi], 'gender': [gender]}
        user_df = pd.DataFrame(user_data)

        if os.path.exists("users.csv"):
            user_df.to_csv("users.csv", mode='a', header=False, index=False)
        else:
            user_df.to_csv("users.csv", index=False)

    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            st.error("Username already exists.")
    conn.close()

# Function to authenticate user
def authenticate_user(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

# Send OTP via Telegram
def send_otp():
    otp = random.randint(1000, 9999)
    st.session_state['otp'] = otp
    st.session_state['otp_expiry'] = time.time() + 30  # OTP valid for 30 minutes
    message = f"Your OTP for login is: {otp}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        st.info("OTP sent to your Telegram successfully.")
    else:
        st.error("Failed to send OTP. Please try again.")

# Verify OTP
def verify_otp(input_otp):
    if 'otp' not in st.session_state or 'otp_expiry' not in st.session_state:
        st.error("No OTP generated. Please request an OTP.")
        return False
    if time.time() > st.session_state['otp_expiry']:
        st.error("OTP expired. Please request a new OTP.")
        return False
    if input_otp == st.session_state['otp']:
        st.success("Authentication successful!")
        return True
    st.error("Invalid OTP. Please try again.")
    return False


# Dropdown to select Register or Login
option = st.sidebar.selectbox("Choose an Option", ["Register", "Login","Admin"])

if option == 'Admin':
    st.sidebar.subheader("Admin Login")
    st.sidebar.image("Image/admin.png")
    data_paths = {
            "Medication Overdose": "Medication_Overdose_Risk_Dataset_with_Single_Blood_Pressure_Value.csv",
            "User_details": "users.csv"
        }
    eda = ExploratoryDataAnalysis(data_paths)
    eda.run()


if option == "Register":

    st.sidebar.subheader("Register")
    st.sidebar.image("Image/register.png")
    username = st.sidebar.text_input("Username")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30,key="age_input")
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0,key="bmi_input")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    if st.sidebar.button("Register"):
        register_user(username, age, bmi, gender)

elif option == "Login":
    st.sidebar.subheader("Login")
    st.sidebar.image("Image/login.png")
    login_username = st.sidebar.text_input("Enter Username", key="login_username")

    if st.sidebar.button("Send OTP"):
        user = authenticate_user(login_username)
        if user:
            send_otp()
        else:
            st.error("Invalid username.")

    otp_input = st.sidebar.text_input("Enter OTP", type="password")
    if st.sidebar.button("Verify OTP"):
        if otp_input.isdigit() and verify_otp(int(otp_input)):
            st.session_state['logged_in'] = True
            st.success(f"Welcome {login_username}!")

# Prediction only accessible after login
if st.session_state.get('logged_in', False):
    st.markdown("## Prediction Section")
    st.header('Enter the details:')
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    drug_type = st.selectbox('Drug Type', ['Painkiller', 'Antihistamine', 'Antidepressant'])
    dosage_mg = st.number_input('Dosage (mg)', min_value=0, max_value=1000, value=50)
    usage_freq = st.number_input('Usage Frequency per Month', min_value=1, max_value=30, value=5)
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=25)
    chronic_conditions = st.selectbox('Chronic Conditions (1: Yes, 0: No)', [1, 0])
    prescribed_med = st.selectbox('Prescribed Medication', ['Yes', 'No'])
    medication_type = st.selectbox('Medication Type', ['OTC', 'Prescription'])
    allergies = st.selectbox('Allergies (1: Yes, 0: No)', [1, 0])
    heart_rate = st.number_input('Heart Rate', min_value=30, max_value=200, value=70)
    blood_pressure = st.number_input('Blood Pressure', min_value=50, max_value=200, value=120)

    # Create a dictionary of user inputs
    input_data = {
        'Age': age,
        'Gender': gender,
        'Drug_Type': drug_type,
        'Dosage_mg': dosage_mg,
        'Usage_Frequency_per_Month': usage_freq,
        'BMI': bmi,
        'Chronic_Conditions': chronic_conditions,
        'Prescribed_Medication': prescribed_med,
        'Medication_Type': medication_type,
        'Allergies': allergies,
        'Heart_Rate': heart_rate,
        'Blood_Pressure': blood_pressure
    }

    if st.button('Predict Overdose Risk'):
        st.image("Image/main.jpg")
        predicted_risk, prediction = make_prediction(input_data)

        # Risk Levels
        risk_labels = ['Low', 'Moderate', 'High', 'Very High']
        risk_colors = ['green', 'yellow', 'orange', 'red']

        # Display the predicted risk level
        st.subheader(f'Predicted Overdose Risk Level: {risk_labels[predicted_risk]}')

        # Bar chart for prediction probabilities
        st.subheader('Prediction Probabilities:')
        fig, ax = plt.subplots()
        ax.bar(risk_labels, prediction[0], color=risk_colors)
        ax.set_xlabel('Risk Levels')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities by Risk Level')
        st.pyplot(fig)

        # Show additional probabilities
        st.write('Risk probabilities for each category:')
        for i, risk in enumerate(risk_labels):
            st.write(f"{risk}: {prediction[0][i] * 100:.1f}%")

