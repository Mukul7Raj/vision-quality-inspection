import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def get_model():
    # Define a simple CNN for binary classification (Pass vs Defect)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # In a real scenario, you would load pre-trained weights here
    # model.load_weights('path_to_weights.h5')
    
    return model

def preprocess_image(image_bytes):
    # Decode image from bytes
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Resize to match model input
    img = cv2.resize(img, (128, 128))
    
    # Optional blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict(image_bytes, model):
    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)
    
    # Assuming 0 is PASS and 1 is DEFECT
    # Adjust based on how the model is trained
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        verdict = "DEFECT DETECTED"
        score = confidence
    else:
        verdict = "PASS"
        score = 1.0 - confidence
        
    return verdict, score
