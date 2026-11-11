import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "PlantVillage"
MODEL_DIR = BASE_DIR / "ml"

MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
CLASSES_PATH = MODEL_DIR / "plant_disease_classes.json"

# Training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

print("TensorFlow version:", tf.__version__)
print("Dataset path:", DATA_DIR)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder nahi mila: {DATA_DIR}"
    )

# Load dataset
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Total disease classes:", num_classes)
print("Classes:", class_names)

# Save class names for prediction later
with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# Improve data-loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Load pretrained MobileNetV2
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initially freeze pretrained layers
base_model.trainable = False

# Build our disease classification model
inputs = keras.Input(shape=(224, 224, 3))

x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model
print("\nTraining started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Evaluate model
loss, accuracy = model.evaluate(val_ds, verbose=1)
print(f"\nValidation accuracy: {accuracy * 100:.2f}%")

# Save trained model
model.save(MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
print("Class names saved at:", CLASSES_PATH)
print("Plant disease model training completed!")