# ============================================================
# app.py - Facial Emotion Recognition Streamlit Web App
# ============================================================

import os
import sys


import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import streamlit as st
import plotly.express as px
from PIL import Image
import h5py
import json
# Suppress TensorFlow logs for cleaner UI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- CONFIGURATION ---
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# Paths to models (Must match what you saved in training)
MODEL_PATHS = {
    "Custom CNN (48x48 - Fastest)": "models/best_emotion_model.h5",
}

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Face Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff4b4b, #fcb045, #4bc8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_resource(show_spinner="Loading AI Model... This might take a minute.")

def load_model(model_name):
    """Loads the selected model and returns it along with its expected input shape."""
    path = MODEL_PATHS[model_name]
    if not os.path.exists(path):
        st.error(f"Model file not found at `{path}`. Please ensure you have trained the model.")
        st.stop()
    
    model = tf.keras.models.load_model(path)
    
    if "Custom CNN" in model_name:
        return model, 48, "grayscale"
    else:
        return model, 128, "rgb"

def detect_and_crop_face(image_np):
    """Uses Haar Cascade to detect a face and crop it with padding."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Failed to load Haar cascade file. Check that models/haarcascade_frontalface_default.xml exists.")
        st.stop()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Take the largest face found
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Add padding (30% around the face)
        pad_x, pad_y = int(w * 0.3), int(h * 0.3)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(image_np.shape[1], x + w + pad_x), min(image_np.shape[0], y + h + pad_y)
        
        cropped_face = image_np[y1:y2, x1:x2]
        return cropped_face, True
    return image_np, False

def preprocess_image(image_pil, target_size, color_mode):
    """Converts PIL to numpy, detects face, resizes, and normalizes."""
    image_np = np.array(image_pil)
    
    # Step 1: Detect and crop face
    cropped_img, face_found = detect_and_crop_face(image_np)
    
    # Step 2: Convert color mode
    if color_mode == 'grayscale':
        img_processed = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    else:
        img_processed = cropped_img # Already RGB
    
    # Step 3: Resize
    img_resized = cv2.resize(img_processed, (target_size, target_size))
    
    # Step 4: Normalize
    img_normalized = img_resized / 255.0
    
    # Step 5: Expand dimensions for batch size
    if color_mode == 'grayscale':
        img_input = np.expand_dims(img_normalized, axis=(0, -1)) # Shape: (1, 48, 48, 1)
    else:
        img_input = np.expand_dims(img_normalized, axis=0)       # Shape: (1, 128, 128, 3)
        
    return img_input, face_found, cropped_img

def predict_emotion(model, processed_image):
    """Runs inference and returns probabilities."""
    predictions = model.predict(processed_image, verbose=0)[0]
    return predictions

# --- MAIN APPLICATION UI ---

st.markdown('<div class="main-header">🎭 Facial Emotion Recognition AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image or use your webcam to detect emotions in real-time.</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Selection
    selected_model_name = st.selectbox(
        "Select AI Model",
        options=list(MODEL_PATHS.keys()),
        index=0,
        help="Custom CNN is fastest. Pre-trained models are slower but potentially more accurate."
    )
    
    # Load Model
    model, img_size, color_mode = load_model(selected_model_name)
    st.success(f"✅ {selected_model_name} Loaded!")
    
    st.divider()
    st.info("""
    **How to use:**
    1. Choose a model in settings.
    2. Upload a clear picture of a face, OR use the camera.
    3. Click 'Detect Emotion'.
    """)

# --- MAIN AREA: INPUT METHODS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'webp'],
        key="uploader"
    )

with col2:
    st.subheader("📸 Webcam")
    camera_photo = st.camera_input("Take a picture", key="camera")

# Determine which input to use
input_image = None
source_type = None

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    source_type = "Uploaded"
elif camera_photo is not None:
    input_image = Image.open(camera_photo).convert('RGB')
    source_type = "Webcam"

# --- PROCESSING & RESULTS ---
if input_image is not None:
    st.markdown("---")
    
    # Layout for Image and Results
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("🖼️ Input Image")
        st.image(input_image, use_container_width=True)
        
        if st.button("🚀 Detect Emotion", use_container_width=True):
            with st.spinner("Analyzing facial features..."):
                # Preprocess
                processed_img, face_found, cropped_display = preprocess_image(
                    input_image, img_size, color_mode
                )
                
                # Predict
                probabilities = predict_emotion(model, processed_img)
                predicted_idx = np.argmax(probabilities)
                predicted_emotion = EMOTIONS[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                # Store in session state to persist across button clicks
                st.session_state['results'] = {
                    'emotion': predicted_emotion,
                    'confidence': confidence,
                    'probs': probabilities,
                    'face_found': face_found
                }
                
    # Display Results if they exist in session state
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        with res_col2:
            st.subheader("🎯 Prediction Result")
            
            if not results['face_found']:
                st.warning("⚠️ No face clearly detected. Analyzing the full image (results may be inaccurate).")
            
            # Big Emoji and Text
            emoji = EMOTION_EMOJIS.get(results['emotion'], '❓')
            st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>{emoji}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #ff4b4b; font-weight: bold; text-transform: uppercase;'>{results['emotion']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 20px; color: #666;'>Confidence: {results['confidence']:.2%}</p>", unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                "Emotion": EMOTIONS,
                "Probability": results['probs'],
                "Emoji": [EMOTION_EMOJIS[e] for e in EMOTIONS]
            })
            
            fig = px.bar(
                prob_df, 
                x='Probability', 
                y='Emotion', 
                orientation='h',
                color='Probability',
                color_continuous_scale='Reds',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(
                height=400, 
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
else:
    st.markdown("---")
    st.info("👆 Please upload an image or take a photo using the webcam to get started.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #aaa; font-size: 12px;'>Built with TensorFlow, OpenCV, and Streamlit | Facial Emotion Recognition Project</p>", unsafe_allow_html=True)
