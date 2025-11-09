"""
MNIST Digit Recognition API
Flask API with web interface for digit recognition
"""

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image
import io
import base64
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift

app = Flask(__name__)

# Configuration
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')
MODEL_PATH = f'models/mnist_model_{MODEL_VERSION}.h5'
METADATA_PATH = f'models/mnist_model_{MODEL_VERSION}_metadata.json'

# Load model
print(f"🔄 Loading model version: {MODEL_VERSION}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    
    # Load metadata
    if os.path.exists(METADATA_PATH):
        # with open(METADATA_PATH, 'r') as f:
        #     metadata = json.load(f)
        # print(f"📋 Metadata loaded: Accuracy {metadata['test_accuracy']*100:.2f}%")
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        # Handle both possible key names
        accuracy_value = metadata.get('test_accuracy') or metadata.get('accuracy')
        if accuracy_value is not None:
            print(f"📋 Metadata loaded: Accuracy {accuracy_value * 100:.2f}%")
        else:
            print("📋 Metadata loaded (no accuracy info found)")

    else:
        metadata = {'version': MODEL_VERSION, 'test_accuracy': 'N/A'}
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    metadata = {}

# HTML Template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Digit Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 800px;
            width: 100%;
        }
        
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .version-badge {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            display: inline-block;
            font-size: 0.9em;
            margin-bottom: 30px;
        }
        
        .info-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }
        
        .canvas-container {
            text-align: center;
            margin: 30px 0;
        }
        
        canvas {
            border: 3px solid #667eea;
            border-radius: 10px;
            cursor: crosshair;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .predict-btn {
            background: #667eea;
            color: white;
        }
        
        .predict-btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .clear-btn {
            background: #f1f3f5;
            color: #495057;
        }
        
        .clear-btn:hover {
            background: #e9ecef;
        }
        
        .result {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            min-height: 100px;
        }
        
        .prediction {
            font-size: 4em;
            color: #667eea;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .confidence {
            font-size: 1.2em;
            color: #495057;
        }
        
        .probabilities {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        
        .prob-item {
            background: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .prob-digit {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .prob-bar {
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin: 8px 0;
            overflow: hidden;
        }
        
        .prob-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s;
        }
        
        .prob-value {
            font-size: 0.9em;
            color: #868e96;
        }
        
        .loading {
            display: none;
            color: #667eea;
            font-size: 1.2em;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 MNIST Digit Recognition</h1>
        <div style="text-align: center;">
            <span class="version-badge">Model Version: {{ version }}</span>
            {% if accuracy != 'N/A' %}
            <span class="version-badge">Accuracy: {{ "%.2f"|format(accuracy * 100) }}%</span>
            {% endif %}
        </div>
        
        <div class="info-box">
            <strong>📝 Instructions:</strong> Draw a digit (0-9) in the canvas below. The model will predict what digit you drew!
        </div>
        
        <div class="canvas-container">
            <canvas id="canvas" width="280" height="280"></canvas>
        </div>
        
        <div class="buttons">
            <button class="predict-btn" onclick="predict()">🔮 Predict Digit</button>
            <button class="clear-btn" onclick="clearCanvas()">🗑️ Clear Canvas</button>
        </div>
        
        <div class="result" id="result" style="display: none;">
            <div class="loading" id="loading">
                <span class="spinner"></span>Analyzing...
            </div>
            <div id="prediction-content"></div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        // Set up canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouch);
        canvas.addEventListener('touchend', stopDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
        
        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }
        
        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').style.display = 'none';
        }
        
        async function predict() {
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            const predictionContent = document.getElementById('prediction-content');
            
            // Show loading
            resultDiv.style.display = 'block';
            loadingDiv.style.display = 'block';
            predictionContent.innerHTML = '';
            
            // Get image data
            const imageData = canvas.toDataURL('image/png');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                
                // Hide loading
                loadingDiv.style.display = 'none';
                
                if (data.error) {
                    predictionContent.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                    return;
                }
                
                // Display prediction
                let html = `
                    <div class="prediction">${data.prediction}</div>
                    <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
                    <div class="probabilities">
                `;
                
                for (let i = 0; i < 10; i++) {
                    const prob = data.probabilities[i];
                    html += `
                        <div class="prob-item">
                            <div class="prob-digit">${i}</div>
                            <div class="prob-bar">
                                <div class="prob-fill" style="width: ${prob * 100}%"></div>
                            </div>
                            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                        </div>
                    `;
                }
                
                html += `</div>`;
                predictionContent.innerHTML = html;
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                predictionContent.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render web interface"""
    # return render_template_string(
    #     HTML_TEMPLATE,
    #     version=MODEL_VERSION,
    #     accuracy=metadata.get('test_accuracy', 'N/A')
    # )
    return render_template_string(
        HTML_TEMPLATE,
        version=MODEL_VERSION,
        accuracy=metadata.get('test_accuracy') or metadata.get('accuracy', 'N/A')
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from drawn image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        
        # # Invert colors (drawing is black on white, MNIST is white on black)
        # image_array = 255 - image_array
        
        # # Normalize
        # image_array = image_array.astype('float32') / 255.0
        # image_array = image_array.reshape(1, 28, 28, 1)

        # Invert colors (drawing is black on white, MNIST is white on black)
        image_array = 255 - image_array
        image_array = image_array.astype('float32') / 255.0

        # Center the digit in the 28x28 image
        cy, cx = center_of_mass(image_array)
        if not (np.isnan(cy) or np.isnan(cx)):  # avoid NaN when image is blank
            shift_y, shift_x = 14 - cy, 14 - cx
            image_array = shift(image_array, shift=(shift_y, shift_x))

        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Predict
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': predictions[0].tolist(),
            'model_version': MODEL_VERSION
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/version', methods=['GET'])
def version():
    """Get model version and metadata"""
    return jsonify({
        'version': MODEL_VERSION,
        'metadata': metadata,
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'version': MODEL_VERSION
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)