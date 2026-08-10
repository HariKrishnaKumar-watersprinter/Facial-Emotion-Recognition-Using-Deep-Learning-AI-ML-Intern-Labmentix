🎭 Facial Emotion Recognition Using Deep Learning
PythonTensorFlowStreamlitLicense

An end-to-end, production-ready Deep Learning project that classifies human facial expressions into 7 distinct emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise). This project goes beyond basic tutorials by solving critical real-world ML challenges: extreme class imbalance, training timeouts on large datasets, and low-latency edge deployment.

📊 Business Problem & Why This Matters
The Subjectivity Bottleneck: Manual emotion tracking in customer support or UX testing is unscalable, slow, and highly biased.
The Hidden Data Problem: Over 70% of communication is non-verbal. Businesses have oceans of visual data (CCTV, video calls) but lack tools to extract emotional sentiment automatically.
Failure to Intercept Churn: Without real-time visual cues, support systems cannot proactively detect rising customer frustration before it results in churn.
Inaccurate Product Feedback: Users often say one thing but their faces express another. Automated FER captures genuine micro-expressions during product testing, eliminating flawed self-reported data.
✨ Key Features & Advanced Techniques
Unlike standard FER implementations, this project implements advanced MLOps and Data Engineering techniques:

Solving Class Imbalance (The "Disgust" Problem):
Balanced Data Generator: A custom Keras Sequence that dynamically forces perfectly equal class distributions in every single batch. No more lazy model bias toward "Happy" faces.
Focal Loss: Replaces standard Cross-Entropy to mathematically force the network to focus on hard-to-predict minority classes.
Crash-Proof "Chunked" Training Loop:
Replaces standard model.fit() to prevent Google Colab timeouts and session crashes.
Processes data in chunks, provides live progress bars, and saves temporary weights every few minutes. If the kernel dies, you lose zero progress.
High-Speed Transfer Learning Optimizations:
Reduces pre-trained model input from 224x224 to 128x128 (75% pixel reduction).
Enables Mixed Precision (FP16) training for ~2x GPU speedup.
Multi-processing data loading to prevent GPU starvation.
Multi-Model Architecture Comparison:
Trains and compares 5 different architectures: Custom CNN, VGG16, ResNet50, MobileNetV2, and EfficientNetB0.
Production Deployment:
Interactive Streamlit Web App with OpenCV Haar Cascade face detection and interactive Plotly charts.
🗂️ Project Architecture
project_root/
├── dataset/
│ ├── train/ (48x48 Grayscale images)
│ │ ├── angry/
│ │ ├── disgust/
│ │ ├── fear/
│ │ ├── happy/
│ │ ├── neutral/
│ │ ├── sad/
│ │ └── surprise/
│ └── test/ (Same structure as train)
├── models/ (Generated after training)
│ ├── best_emotion_model.h5
│ ├── best_mobilenetv2_emotion.h5
│ └── emotion_model.tflite
├── outputs/
│ ├── plots/ (ROC, Confusion Matrix, Feature Maps, etc.)
│ ├── model_summary.json
│ └── training_history.csv
├── app.py (Streamlit Web Application)
├── requirements.txt
└── emotion_recognition.ipynb (Main Training Notebook)

text


---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.8+
*   NVIDIA GPU (highly recommended for Transfer Learning models)

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition
pip install -r requirements.txt
3. Prepare Dataset
Place your dataset in the dataset/train/ and dataset/test/ directories following the folder structure shown above. (Compatible with FER2013, AffectNet, or custom datasets).

4. Run Training
Open Jupyter Notebook and run the cells sequentially:

bash

jupyter notebook emotion_recognition.ipynb
Note: The chunked training loop will automatically save the best weights to the models/ directory.

5. Run the Streamlit App
Once models are trained, deploy the interactive web app:

bash

streamlit run app.py
🧠 Model Selection Guide
The project outputs multiple models for different deployment scenarios:

Model
Input Size
Speed
Best Use Case
Custom CNN	48x48 Grayscale	⚡ Lightning Fast	Real-time webcam, CPU deployment, mobile apps.
MobileNetV2	128x128 RGB	🏃 Fast	Edge devices (Raspberry Pi), mobile apps needing high accuracy.
EfficientNetB0	128x128 RGB	🚶 Moderate	Server-side API where highest accuracy is required.
VGG16 / ResNet50	128x128 RGB	🐢 Heavy	Baseline benchmarking, academic comparison.

📈 Evaluation Metrics
The model is evaluated beyond simple accuracy to ensure fairness across minority classes:

Macro F1-Score: Ensures "Disgust" and "Fear" perform just as well as "Happy".
Normalized Confusion Matrix: Identifies exactly which emotions the model confuses (e.g., Fear vs. Surprise).
ROC-AUC Curves (One-vs-Rest): Validates the model's confidence thresholds across all 7 classes.
Feature Map Visualization: Peers inside the CNN to see exactly which facial features (eyes, mouth) the model is activating on.
🛠️ Tech Stack
Deep Learning: TensorFlow, Keras
Computer Vision: OpenCV (Haar Cascades, Image Processing)
Data Manipulation: NumPy, Pandas, Scikit-Learn
Visualization: Matplotlib, Seaborn, Plotly
Deployment: Streamlit
🚧 Future Work
Attention Mechanisms: Implement CBAM or Self-Attention to force the model to look at specific facial regions (e.g., eyes for "Sad", mouth for "Happy").
Video/Temporal Analysis: Extend from static images to sequential video frames using LSTMs or Transformers to capture temporal emotion dynamics.
Edge Optimization: Use TensorFlow Lite Micro to deploy the Custom CNN directly on microcontrollers (ESP32) for IoT smart-home emotion sensors.
