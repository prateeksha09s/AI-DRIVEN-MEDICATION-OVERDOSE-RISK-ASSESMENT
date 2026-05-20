import gradio as gr
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("trained_model.h5")

def predict(age, bp, chol, glucose):
    input_data = np.array([[age, bp, chol, glucose]])
    pred = model.predict(input_data)[0][0]

    return "⚠️ High Risk" if pred > 0.5 else "✅ Low Risk"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Glucose")
    ],
    outputs="text",
    title="Medication Overdose Risk Prediction"
)

demo.launch()