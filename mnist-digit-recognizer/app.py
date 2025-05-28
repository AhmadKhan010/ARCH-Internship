import gradio as gr
import numpy as np
from PIL import Image
import joblib

# Load trained models
rf_clf = joblib.load("models/rf_model.pkl")
sgd_clf = joblib.load("models/sgd_model.pkl")

def predict_digit(img, model_choice="RandomForest"):
    try:
        if isinstance(img, dict) and 'composite' in img:
            img_array = np.array(img['composite'])
        elif isinstance(img, np.ndarray):
            img_array = img
        elif isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            return "Error: Unsupported input type."

        if img_array.ndim == 3:
            img_array = np.mean(img_array, axis=2).astype(np.uint8)

        img = Image.fromarray(img_array).resize((28, 28)).convert("L")
        img_array = np.array(img)
        img_array = 255 - img_array  # invert colors
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, -1)

        if model_choice == "RandomForest":
            pred = rf_clf.predict(img_array)[0]
            return f"Random Forest Predicted Digit: {pred}"
        elif model_choice == "SGD":
            pred = sgd_clf.predict(img_array)[0]
            return f"SGD Classifier Predicted Digit: {pred}"
        elif model_choice == "Both":
            rf_pred = rf_clf.predict(img_array)[0]
            sgd_pred = sgd_clf.predict(img_array)[0]
            return f"Random Forest: {rf_pred}, SGD: {sgd_pred}"
        else:
            return "Error: Invalid model selection."
    except Exception as e:
        return f"Error processing image: {str(e)}"

interface = gr.Interface(
    fn=predict_digit,
    inputs=[
        gr.Sketchpad(image_mode="L", label="Draw a digit"),
        gr.Dropdown(
            choices=["RandomForest", "SGD", "Both"],
            label="Select Model",
            value="RandomForest"
        )
    ],
    outputs="text",
    title="MNIST Digit Recognizer",
    description="Draw a digit and select a model to predict the digit."
)

interface.launch()
