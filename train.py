import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


IMG_SIZE = 224
DATA_DIR = "data/train/images"
MODEL_NAME = "defect_detector_cnn.h5"

print("Loading data...")

# LOAD DATA

classes = sorted(os.listdir(DATA_DIR))

data = []
labels = []

for label, cls in enumerate(classes):
    path = os.path.join(DATA_DIR, cls)

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)

        except Exception as e:
            print("Error:", e)

X = np.array(data, dtype="float32") / 255.0
y = np.array(labels)

print("Dataset shape:", X.shape)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# BUILD CNN MODEL

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# TRAIN

print("Training started...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# SAVE MODEL

model.save(MODEL_NAME)
print(f"✅ Model saved as {MODEL_NAME}")

