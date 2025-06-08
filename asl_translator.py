import tensorflow as tf
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load pre-trained model (upload this to your GitHub)
model = load_model('asl_cnn_model.h5')  # Download from: https://bit.ly/3R0jJtH

# Class labels (only first 5 letters: A, B, C, D, E)
CLASSES = ['A', 'B', 'C', 'D', 'E']

def preprocess_image(image):
    """Resize and normalize input image"""
    image = tf.image.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    """Predict ASL letter from image"""
    processed = preprocess_image(image)
    pred = model.predict(processed)[0]
    confidences = {CLASSES[i]: float(pred[i]) for i in range(5)}
    return confidences

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(shape=(200, 200)),
    outputs=gr.Label(num_top_classes=3),
    examples=[
        "A_test.jpg",
        "B_test.jpg",
        "C_test.jpg"
    ],
    title="ASL Letter Translator (A-E)",
    description="Recognizes signs for A, B, C, D, E"
)

interface.launch()
