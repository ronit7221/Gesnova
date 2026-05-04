import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

# ==============================
# MEDIAPIPE SETUP
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)

# ==============================
# MODEL
# ==============================
IMG_SIZE = 128

model = tf.keras.models.load_model(
    "model/gesnova_hybrid_model.keras"
)

labels = json.load(open(
    "model/labels_hybrid.json"
))
labels = {int(k): v for k, v in labels.items()}

print("✅ Model loaded")

# ==============================
# LANDMARK EXTRACTION
# ==============================
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    right_hand = [0.0] * 63
    left_hand  = [0.0] * 63

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name

            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])

            if handedness == "Right":
                right_hand = coords
            else:
                left_hand = coords

    return np.array(right_hand + left_hand, dtype=np.float32)

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

SMOOTH_FRAMES = 5
pred_history = []

print("✅ Webcam started — Press Q to quit")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # ===== PREPROCESSING (FIXED) =====
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = img_resized / 255.0

    landmarks = extract_landmarks(frame)

    text = "No Hand Detected"
    color = (0, 0, 255)

    # ===== PREDICTION =====
    if np.any(landmarks):
        pred = model.predict([
            np.expand_dims(img_resized, axis=0),
            np.expand_dims(landmarks, axis=0)
        ], verbose=0)

        confidence = float(np.max(pred))
        class_idx = int(np.argmax(pred))

        if confidence > 0.75:
            pred_history.append((class_idx, confidence))

            if len(pred_history) > SMOOTH_FRAMES:
                pred_history.pop(0)

            # ===== WEIGHTED SMOOTHING =====
            scores = {}
            for idx, conf in pred_history:
                scores[idx] = scores.get(idx, 0) + conf

            smooth_idx = max(scores, key=scores.get)

            text = f"{labels[smooth_idx]} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            pred_history.clear()
            text = f"Uncertain ({confidence:.2f})"
            color = (0, 165, 255)
    else:
        pred_history.clear()

    # ===== DISPLAY =====
    cv2.rectangle(display_frame, (0, 0), (500, 60), (0, 0, 0), -1)

    cv2.putText(display_frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    cv2.putText(display_frame, "Press Q to quit",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Gesture Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
print("✅ Done")