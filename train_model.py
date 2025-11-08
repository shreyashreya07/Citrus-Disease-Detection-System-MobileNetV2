import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"\n✅ Detected {NUM_CLASSES} classes: {list(train_generator.class_indices.keys())}")

# -----------------------------
# Build Model: MobileNetV2
# -----------------------------
def create_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model, base_model

model, base_model = create_model(NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📘 Model Summary")
model.summary()

# -----------------------------
# Callbacks
# -----------------------------
os.makedirs('saved_models', exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('saved_models/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
]

# -----------------------------
# Phase 1: Transfer Learning
# -----------------------------
print("\n🚀 Starting Phase 1: Transfer Learning")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Phase 2: Fine-Tuning
# -----------------------------
print("\n🔧 Starting Phase 2: Fine-Tuning")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False  # unfreeze last 20 layers only

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks,
    verbose=1
)

# Combine histories
for k in history.history.keys():
    history.history[k].extend(history_fine.history.get(k, []))

# -----------------------------
# Save Final Model
# -----------------------------
model.save('saved_models/citrus_mobilenetv2_model.keras')
print("\n✅ Model saved as saved_models/citrus_mobilenetv2_model.keras")

# Save class names
with open('saved_models/class_names.txt', 'w') as f:
    for name in train_generator.class_indices.keys():
        f.write(f"{name}\n")
print("✅ Class names saved!")

# -----------------------------
# Evaluation
# -----------------------------
print("\n📊 Evaluating final model...")
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n✅ Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"📉 Final Validation Loss: {val_loss:.4f}")

# -----------------------------
# Plot Accuracy & Loss
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('saved_models/training_history.png', dpi=300)
print("✅ Training history plot saved at saved_models/training_history.png")

# -----------------------------
# Confusion Matrix + Report
# -----------------------------
print("\n📈 Generating confusion matrix...")

# Predict on validation data
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(val_generator.classes, y_pred)
class_labels = list(train_generator.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("saved_models/confusion_matrix.png", dpi=300)
print("✅ Confusion matrix saved at saved_models/confusion_matrix.png")

# Classification report
report = classification_report(val_generator.classes, y_pred, target_names=class_labels)
print("\n📋 Classification Report:")
print(report)

with open("saved_models/classification_report.txt", "w") as f:
    f.write(report)
print("✅ Classification report saved!")

print("\n🎯 Training Complete — Ready for 90%+ accuracy!")
