from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class indices
model_path = "saved_models/citrus_mobilenetv2_model.keras"
class_indices_path = "class_indices.json"

try:
    model = tf.keras.models.load_model(model_path)
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    idx_to_class = {}

# Disease information database
DISEASE_INFO = {
    "Black spot": {
        "description": "Black spot is a fungal disease causing dark lesions on citrus fruits and leaves, leading to premature fruit drop and reduced quality.",
        "symptoms": "Circular black or brown spots on leaves and fruits, yellowing of surrounding tissue, premature leaf drop, fruit blemishes and decay.",
        "treatment": "Apply copper-based fungicides, remove infected plant material, improve air circulation, avoid overhead watering, and prune affected branches."
    },
    "canker": {
        "description": "Citrus canker is a highly contagious bacterial disease that causes lesions on leaves, stems, and fruit, severely affecting tree health and fruit marketability.",
        "symptoms": "Raised brown lesions with oily appearance, yellow halos around spots, leaf drop, twig dieback, and corky scabs on fruit surfaces.",
        "treatment": "Remove and destroy infected tissue, apply copper bactericides, maintain tree vigor through proper nutrition, and implement strict sanitation practices."
    },
    "greening": {
        "description": "Huanglongbing (citrus greening) is a devastating bacterial disease transmitted by psyllids, causing decline and eventual death of citrus trees.",
        "symptoms": "Yellow shoots, blotchy mottled leaves, lopsided bitter fruits, premature fruit drop, and overall tree decline with stunted growth.",
        "treatment": "Remove infected trees immediately, control psyllid vectors with insecticides, plant disease-free certified nursery stock, and monitor regularly for early detection."
    },
    "healthy": {
        "description": "The citrus plant shows no signs of disease and exhibits normal, vigorous growth with proper coloration and structure.",
        "symptoms": "Vibrant green leaves, uniform growth, no spots or lesions, healthy fruit development, and strong overall plant vigor.",
        "treatment": "Maintain preventive care: regular watering, balanced fertilization, pest monitoring, proper pruning, and good orchard sanitation practices."
    },
    "other": {
        "description": "It is not citrus leaf.please upload the image of citrus leaf."
        
    }
}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.65

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_disease_info(disease_name):
    """Get disease information from database"""
    disease_key = disease_name.lower()
    return DISEASE_INFO.get(disease_key, {
        "description": "Information not available for this condition.",
        "symptoms": "Please consult with a plant pathologist for detailed information.",
        "treatment": "Seek professional advice for proper treatment recommendations."
    })

def predict_image(img_path):
    """Predict disease from image"""
    try:
        if model is None:
            return {
                "success": False,
                "error": "Model not loaded. Please check server configuration."
            }

        # Load & preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = model.predict(img_array, verbose=0)
        pred_class_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_class_idx] * 100)
        predicted_class = idx_to_class.get(pred_class_idx, "Unknown")

        # Get all predictions
        all_predictions = {}
        for idx, prob in enumerate(preds[0]):
            class_name = idx_to_class.get(idx, f"Class_{idx}")
            all_predictions[class_name] = float(prob * 100)

        # Check if confidence is below threshold
        if preds[0][pred_class_idx] < CONFIDENCE_THRESHOLD:
            return {
                "success": False,
                "error": f"Low confidence detection ({confidence:.1f}%). Please upload a clear citrus leaf image.",
                "confidence": confidence
            }

        # Get disease information
        disease_info = get_disease_info(predicted_class)

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "disease_info": disease_info,
            "all_predictions": all_predictions
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing image: {str(e)}"
        }

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Please upload JPG, JPEG, or PNG images."
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction
        result = predict_image(filepath)

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": list(idx_to_class.values()) if idx_to_class else []
    })

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "Citrus Disease Detection API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Upload image for disease detection",
            "/health": "GET - Check API health status"
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Citrus Disease Detection System")
    print("=" * 50)
    print(f"Model loaded: {model is not None}")
    print(f"Available classes: {list(idx_to_class.values())}")
    print("Starting server on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)