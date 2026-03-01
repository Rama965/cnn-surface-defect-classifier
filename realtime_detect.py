import cv2
import numpy as np
import tensorflow as tf
import os
import time

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
MODEL_PATH = "defect_detector_cnn.h5"
DATA_DIR = "data/train/images"
CONF_THRESHOLD = 0.6

# ==============================
# LOAD MODEL
# ==============================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = sorted(os.listdir(DATA_DIR))

print("Classes:", classes)

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("✅ Press 'q' to quit")

prev_time = 0

# ==============================
# REAL-TIME LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    # Label logic
    if confidence < CONF_THRESHOLD:
        label = "Unknown"
        color = (0, 0, 255)
    else:
        label = f"{classes[class_idx]} ({confidence:.2f})"
        color = (0, 255, 0)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Draw text
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show window
    cv2.imshow("Real-Time Defect Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()