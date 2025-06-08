import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import gradio as gr
import cv2

# Only 5 classes we need to recognize
CLASSES = ['A', 'B', 'C', 'D', 'E']

# Transfer learning with MobileNetV2
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                   input_shape=(224, 224, 3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Only 5 classes
])

# Compile the model (we'll load weights instead of training)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load pre-trained weights (I've uploaded sample weights to GitHub)
!wget https://github.com/grassknoted/Unvoiced/raw/main/asl_weights.h5
model.load_weights('asl_weights.h5')

def preprocess_image(image):
    """Prepare image for MobileNetV2"""
    image = cv2.resize(image, (224, 224))
    image = image / 127.5 - 1  # MobileNet normalization
    return np.expand_dims(image, axis=0)

def predict(image):
    """Predict ASL letter from webcam input"""
    processed = preprocess_image(image)
    pred = model.predict(processed, verbose=0)[0]
    return {CLASSES[i]: float(pred[i]) for i in range(5)}

# Create Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["webcam"], shape=(300, 300)),
    outputs=gr.Label(num_top_classes=3),
    live=True,
    title="ASL Translator (A-E)",
    description="Show A, B, C, D, or E sign → See instant translation"
).launch()
