# Vision-Based Quality Inspection System

## Project Overview
Automated component defect detection using classical image processing (OpenCV) and Deep Learning (CNN). This system allows users to upload images of industrial components and instantly receive a PASS or DEFECT DETECTED verdict.

## Tech Stack
- **Python**
- **OpenCV** (Image preprocessing: resize, blur, normalize)
- **TensorFlow / Keras** (Convolutional Neural Network)
- **Flask** (REST API & Server)
- **HTML/JS/CSS** (Frontend dashboard)

## Key Features
- Real-time image preprocessing (blurring, resizing, normalization)
- CNN-based inference for defect classification
- Flask REST API endpoint `/predict` for inference
- Clean, responsive web UI for uploading images and viewing results

## Directory Structure
```
vision-quality-inspection/
│
├── app.py                 # Flask server and routing
├── model.py               # CNN architecture and OpenCV preprocessing
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
├── .gitignore             # Ignored files for Git
└── templates/
    └── index.html         # Frontend HTML/JS/CSS
```

## Setup & Running Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/Mukul7Raj/vision-quality-inspection.git
   cd vision-quality-inspection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

4. **Access the Web UI**:
   Open your browser and navigate to: `http://localhost:5000`

## Results & Evaluation
*(Note: Since this is a template, specific metrics will depend on the dataset and trained weights used.)*
- **Accuracy**: ~95% (Example metric)
- **Loss**: ~0.15 (Example metric)
- **Inference Time**: < 100ms per image
