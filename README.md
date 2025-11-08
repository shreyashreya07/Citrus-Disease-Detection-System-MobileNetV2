# 🍊 Citrus Disease Detection System

A comprehensive deep learning application for classifying citrus fruit diseases using transfer learning with MobileNetV2. This project includes a complete training pipeline, Flask web API, modern web interface, and thorough evaluation tools.

## 📖 Overview

This system uses state-of-the-art computer vision techniques to identify citrus diseases from leaf images. It combines transfer learning, extensive data augmentation, and a modern web interface to provide accurate, real-time disease detection and treatment recommendations.

## ✨ Features

### Training Pipeline
- **Transfer Learning**: Leverages MobileNetV2 pre-trained on ImageNet
- **Two-Phase Training**: Initial transfer learning followed by fine-tuning
- **Advanced Data Augmentation**: Rotation, shifting, zooming, and brightness adjustment
- **Smart Callbacks**: Early stopping, model checkpointing, and adaptive learning rate
- **Comprehensive Metrics**: Confusion matrix, classification report, and training history

### Web Application
- **Modern UI**: Dark-themed, responsive interface with animations
- **Drag & Drop Upload**: Easy image upload functionality
- **Real-time Analysis**: Instant disease prediction with confidence scores
- **Disease Information**: Detailed descriptions, symptoms, and treatment recommendations
- **Probability Visualization**: Bar charts showing all class probabilities
- **Mobile Responsive**: Works seamlessly on all devices

### Evaluation Tools
- **Test Set Evaluation**: Comprehensive model testing
- **Confusion Matrix**: Visual representation of classification performance
- **ROC Curves**: Per-class receiver operating characteristic analysis
- **Classification Metrics**: Precision, recall, F1-score with visualizations

## 📁 Project Structure

```
DL PRO NEW/
├── dataset/
│   ├── train/
│   │   ├── black_spot/
│   │   ├── canker/
│   │   ├── greening/
│   │   ├── healthy/
│   │   └── other/
│   ├── val/
│   │   └── [same structure]
│   └── test/
│       └── [same structure]
├── saved_models/
│   ├── citrus_mobilenetv2_model.keras
│   ├── best_model.keras
│   ├── class_names.txt
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── train_model.py                    # Model training script
├── app.py                      # Flask web application
├── evaluate.py                 # Model evaluation script
├── class_indices.json          # Class mapping file
├── requirements.txt            # Python dependencies
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/shreyashreya07/Citrus-Disease-Detection-System-MobileNetV2
cd DL PRO NEW
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
tensorflow>=2.10.0
flask>=2.3.0
flask-cors>=4.0.0
numpy>=1.23.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
pillow>=10.0.0
werkzeug>=2.3.0
```

## 📂 Dataset Setup

Organize your citrus disease images in the following structure:

```
dataset/
├── train/          # Training images (70-80%)
│   ├── black_spot/
│   ├── canker/
│   ├── greening/
│   ├── healthy/
│   └── other/
├── val/            # Validation images (10-15%)
│   └── [same structure]
└── test/           # Test images (10-15%)
    └── [same structure]
```

**Supported Formats:** JPG, JPEG, PNG

**Recommended Image Size:** 224x224 pixels (automatically resized)

**Dataset Guidelines:**
- Balance classes with similar number of images per category
- Use high-quality, clear images of citrus leaves
- Include various lighting conditions and angles
- Minimum 100 images per class recommended

## 🚀 Usage

### 1. Train the Model

Run the training script to create your disease classification model:

```bash
python train.py
```

**What happens during training:**
1. Loads and augments training data
2. Builds MobileNetV2-based architecture
3. Phase 1: Transfer learning (20 epochs)
4. Phase 2: Fine-tuning (30 additional epochs)
5. Saves best model and generates evaluation plots

**Output Files:**
- `saved_models/citrus_mobilenetv2_model.keras` - Final trained model
- `saved_models/best_model.keras` - Best checkpoint
- `saved_models/class_names.txt` - Class labels
- `saved_models/training_history.png` - Training curves
- `saved_models/confusion_matrix.png` - Validation confusion matrix
- `saved_models/classification_report.txt` - Detailed metrics

**Training Time:** 1-3 hours (depending on hardware and dataset size)

### 2. Create Class Indices File

After training, create `class_indices.json`:

```python
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
```

### 3. Launch Web Application

Start the Flask server:

```bash
python app.py
```

Access the application:
- **Local:** http://localhost:5000
- **Network:** http://0.0.0.0:5000

### 4. Evaluate Model Performance

Run comprehensive evaluation on test set:

```bash
python evaluate.py
```

**Evaluation Outputs:**
- Test accuracy
- Confusion matrix visualization
- Classification report with bar charts
- ROC curves for each class

## 🏗️ Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 Base
(Pre-trained on ImageNet)
    ↓
Global Average Pooling 2D
    ↓
Batch Normalization
    ↓
Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(num_classes, Softmax)
    ↓
Output (Class Probabilities)
```

### Training Strategy

**Phase 1: Transfer Learning (20 epochs)**
- Base model: Frozen
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Categorical crossentropy

**Phase 2: Fine-Tuning (30 epochs)**
- Last 20 layers: Unfrozen
- Learning rate: 0.00001
- Allows model to adapt to citrus-specific features

### Data Augmentation

Training images undergo:
- Random rotation: ±40°
- Width/height shifts: 20%
- Shear transformations: 20%
- Zoom range: 20%
- Horizontal flips: 50% probability
- Brightness adjustment: 0.8-1.2x
- Normalization: Rescale to [0,1]

## 🌐 Web Application

### Features

**Upload Interface:**
- Drag & drop functionality
- File browser selection
- Supported formats: JPG, PNG, JPEG
- Maximum file size: 16MB

**Results Display:**
- Disease name with confidence score
- Animated probability bars
- Disease description
- Symptom identification
- Treatment recommendations
- All class predictions with percentages

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Image prediction API |
| `/health` | GET | System health check |
| `/api` | GET | API documentation |

### Disease Information Database

The system provides detailed information for each disease:

- **Black Spot**: Fungal infection treatment
- **Canker**: Bacterial disease management
- **Greening**: Huanglongbing control measures
- **Healthy**: Preventive care guidelines
- **Other**: Non-citrus leaf detection

### Confidence Threshold

- **Threshold:** 65% (0.65)
- Predictions below threshold return error message
- Ensures reliable disease identification
- Prevents misclassification of unclear images

## 📊 Model Evaluation

### Evaluation Script Features

```python
# Load test data
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
```

### Visualizations Generated

1. **Confusion Matrix**
   - Heatmap showing true vs predicted labels
   - Identifies commonly confused classes
   - Helps diagnose model weaknesses

2. **Classification Report Bar Chart**
   - Precision, recall, F1-score per class
   - Visual comparison of metrics
   - Easy identification of problem classes

3. **ROC Curves**
   - Per-class receiver operating characteristic
   - Area Under Curve (AUC) scores
   - Model discrimination capability

### Metrics Explained

- **Accuracy:** Overall correct predictions
- **Precision:** Correct positive predictions / Total positive predictions
- **Recall:** Correct positive predictions / Actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **AUC:** Area under ROC curve (0.5 = random, 1.0 = perfect)

## 🔌 API Documentation

### POST /predict

Upload image for disease prediction.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/citrus_leaf.jpg"
```

**Success Response:**
```json
{
  "success": true,
  "predicted_class": "black_spot",
  "confidence": 94.5,
  "disease_info": {
    "description": "Black spot is a fungal disease...",
    "symptoms": "Circular black or brown spots...",
    "treatment": "Apply copper-based fungicides..."
  },
  "all_predictions": {
    "black_spot": 94.5,
    "canker": 3.2,
    "healthy": 1.8,
    "greening": 0.4,
    "other": 0.1
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Low confidence detection (62.3%). Please upload a clear citrus leaf image.",
  "confidence": 62.3
}
```

### GET /health

Check API status and available classes.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["black_spot", "canker", "greening", "healthy", "other"]
}
```

## ⚙️ Configuration

### Training Parameters

Edit `train.py` to adjust:

```python
# Image and batch configuration
IMG_SIZE = 224          # Input image dimensions
BATCH_SIZE = 32         # Samples per batch
EPOCHS = 50            # Maximum training epochs

# Dataset paths
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'

# Callbacks
EarlyStopping(patience=10)           # Stop if no improvement
ModelCheckpoint(save_best_only=True) # Save best model
ReduceLROnPlateau(patience=5)        # Reduce learning rate
```

### Web Application Settings

Edit `app.py` to configure:

```python
# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model paths
model_path = "saved_models/citrus_mobilenetv2_model.keras"
class_indices_path = "class_indices.json"

# Prediction threshold
CONFIDENCE_THRESHOLD = 0.65  # 65% minimum confidence

# Server settings
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📈 Results & Metrics

### Expected Performance

With proper training data:
- **Validation Accuracy:** 90-95%
- **Test Accuracy:** 88-95%
- **Precision:** 85-95% per class
- **Recall:** 85-95% per class
- **Inference Time:** <100ms per image

### Performance Tips

**To improve accuracy:**
1. Increase training data per class (500+ images recommended)
2. Balance dataset across all classes
3. Use high-quality, well-lit images
4. Increase training epochs (monitor for overfitting)
5. Experiment with different augmentation parameters
6. Try other base models (ResNet50, EfficientNet)

**To reduce overfitting:**
1. Increase dropout rates
2. Add more data augmentation
3. Use early stopping
4. Reduce model complexity
5. Implement L2 regularization

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Error**
```
Error: Unable to load model
```
**Solution:** Ensure `saved_models/citrus_mobilenetv2_model.keras` exists and run training first.

**2. Class Indices Missing**
```
FileNotFoundError: class_indices.json
```
**Solution:** Create class indices file after training (see Usage section 2).

**3. Low Accuracy**
```
Validation accuracy stuck at 60%
```
**Solution:** 
- Check dataset quality and balance
- Increase training data
- Verify correct data augmentation
- Ensure proper preprocessing

**4. Out of Memory**
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:** Reduce `BATCH_SIZE` in train.py (try 16 or 8).

**5. CORS Error in Web App**
```
Access blocked by CORS policy
```
**Solution:** Flask-CORS is already configured. Check if frontend URL matches allowed origins.

### GPU Configuration

For CUDA-enabled GPUs:

```python
import tensorflow as tf

# Check GPU availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 🎨 Customization

### Add New Disease Classes

1. Create new folder in `dataset/train/`, `val/`, `test/`
2. Add images to new class folder
3. Retrain model with `python train.py`
4. Update disease information in `app.py`:

```python
DISEASE_INFO = {
    "new_disease": {
        "description": "Description here...",
        "symptoms": "Symptoms here...",
        "treatment": "Treatment here..."
    }
}
```

### Change Base Model

Replace MobileNetV2 with other architectures:

```python
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3

def create_model(num_classes):
    base_model = ResNet50(  # or EfficientNetB0, InceptionV3
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Rest of the code...
```

### Modify UI Theme

Edit `templates/index.html` CSS variables:

```css
:root {
    --bg-dark: #0a0e27;
    --accent-primary: #00ff87;
    --accent-secondary: #00d9ff;
    /* Customize colors */
}
```

## 🙏 Acknowledgments

- **MobileNetV2 Architecture** by Google Research
- **TensorFlow/Keras** for deep learning framework
- **ImageNet Dataset** for pre-trained weights
- **Flask Framework** for web application
- **Scientific Community** for citrus disease research

## 🚀 Future Enhancements

- [ ] Mobile application (Android/iOS)
- [ ] Real-time camera detection
- [ ] Multi-language support
- [ ] Treatment recommendation system
- [ ] Historical tracking of infected areas
- [ ] Integration with farming management systems
- [ ] Advanced visualization dashboard
- [ ] Model interpretability (Grad-CAM)
- [ ] Batch image processing
- [ ] Cloud deployment guide

---

**🍊 Ready to detect citrus diseases with 90%+ accuracy! Happy coding! 🎯**

**Made with ❤️ for sustainable agriculture**