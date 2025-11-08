from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model and class indices
model_path = "citrus_disease_model.h5"
class_indices_path = "class_indices.json"

model = tf.keras.models.load_model(model_path)
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.65

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    try:
        # Load & preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]

        # Check if confidence is below threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "error": True,
                "message": "Unable to identify the image. Please upload a clear citrus leaf image from the trained categories.",
                "confidence": float(confidence * 100)
            }

        return {
            "error": False,
            "name": idx_to_class[pred_class],
            "confidence": float(confidence * 100)
        }

    except Exception as e:
        return {
            "error": True,
            "message": f"Error processing image: {str(e)}",
            "confidence": 0
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload folder if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Get prediction
            result = predict_image(filepath)
            
            return render_template('result.html', result=result, filename=filename)
    
    return render_template('upload.html')

@app.route('/about/<disease>')
def about_disease(disease):
    return render_template('about.html', disease=disease)

@app.route('/tips/<disease>')
def tips(disease):
    return render_template('tips.html', disease=disease)

if __name__ == '__main__':
    app.run(debug=True)