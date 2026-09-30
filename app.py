import os
from flask import Flask, render_template, request, jsonify
from model import get_model, predict

app = Flask(__name__)

# Initialize the model globally
cnn_model = get_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Read the image bytes
        image_bytes = file.read()
        
        # Get prediction
        verdict, confidence = predict(image_bytes, cnn_model)
        
        return jsonify({
            'verdict': verdict,
            'confidence': f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
