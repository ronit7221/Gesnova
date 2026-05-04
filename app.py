from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time

app = Flask(__name__)

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("model/gesnova_model_landmarks (1).keras")

with open("model/labels (2).json") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# ===== CAMERA (SINGLE SOURCE) =====
cap = cv2.VideoCapture(0)

latest_frame = None
latest_prediction = {"label": "-", "confidence": 0}
last_prediction_time = 0

# ===== FEATURE EXTRACTION =====
def extract_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    while len(landmarks) < 126:
        landmarks.extend([0, 0, 0])

    return np.array(landmarks, dtype=np.float32)

# ===== VIDEO STREAM =====
def generate_frames():
    global latest_frame, latest_prediction, last_prediction_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        latest_frame = frame.copy()

        # ===== LIMIT PREDICTION RATE =====
        current_time = time.time()

        if current_time - last_prediction_time > 0.2:
            features = extract_features(frame)

            if np.any(features):
                pred = model.predict(features.reshape(1, -1), verbose=0)
                conf = float(np.max(pred))
                label = labels[np.argmax(pred)]

                if conf > 0.6:
                    latest_prediction = {"label": label, "confidence": conf}
                else:
                    latest_prediction = {"label": "Uncertain", "confidence": conf}
            else:
                latest_prediction = {"label": "No Hand", "confidence": 0}

            last_prediction_time = current_time

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    return jsonify(latest_prediction)

# ===== RUN =====
app.run(debug=True)