import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and label encoders
model = load_model('trained_model.h5')  # Ensure the model is saved as 'trained_model.h5'

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the label encoders (fit them using the original data from the dataset)
label_encoders = {
    'Gender': LabelEncoder(),
    'Drug_Type': LabelEncoder(),
    'Prescribed_Medication': LabelEncoder(),
    'Medication_Type': LabelEncoder()
}

# Define the manual input data (replace with actual data)
input_data = {
    'Age': 90,
    'Gender': 'Male',  # This should be encoded using the LabelEncoder
    'Drug_Type': 'Painkiller',  # This should be encoded using the LabelEncoder
    'Dosage_mg': 135,
    'Usage_Frequency_per_Month': 4,
    'BMI': 28,
    'Chronic_Conditions': 1,
    'Prescribed_Medication': 'No',  # This should be encoded using the LabelEncoder
    'Medication_Type': 'OTC',  # This should be encoded using the LabelEncoder
    'Allergies': 0,
    'Heart_Rate': 200,
    'Blood_Pressure': 135
}

# Fit the label encoders again on the original data
# Here, we are assuming that the original labels are available in the dataset.
# Alternatively, you can save the original labels and load them here for transformation.
# Here is an example of fitting the encoders using the `data` variable:

# Fit the label encoders with the data (fit it on the whole dataset for transformation)
# This assumes the `data` variable has been preprocessed as in your training script

data = pd.read_csv('Medication_Overdose_Risk_Dataset_with_Single_Blood_Pressure_Value.csv')  # Load the dataset
for col in ['Gender', 'Drug_Type', 'Prescribed_Medication', 'Medication_Type']:
    label_encoders[col].fit(data[col])

# Encode categorical variables
encoded_data = input_data.copy()
for col in ['Gender', 'Drug_Type', 'Prescribed_Medication', 'Medication_Type']:
    encoded_data[col] = label_encoders[col].transform([input_data[col]])[0]

# Prepare input for prediction (scale the features)
X_input = pd.DataFrame([encoded_data])
X_scaled_input = scaler.transform(X_input)  # Use transform here instead of fit_transform

# Make the prediction
prediction = model.predict(X_scaled_input)
predicted_risk = np.argmax(prediction, axis=1)[0]

# Output the result
risk_labels = ['Low', 'Moderate', 'High', 'Very High']  # Adjust according to the unique risk values
print(f"The predicted overdose risk level is: {risk_labels[predicted_risk]}")
