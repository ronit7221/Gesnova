import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import tkinter as tk
from PIL import Image, ImageTk

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("model/gesnova_model_landmarks (1).keras")

with open("model/labels (2).json", "r") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6
)

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

    return np.array(landmarks, dtype=np.float32), results


# ===== MAIN APP =====
class GesNovaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GesNova")
        self.root.geometry("800x600")
        self.root.configure(bg="#0f172a")

        self.cap = None
        self.running = False
        self.frame_count = 0

        self.create_home()

        # Proper close handling
        self.root.protocol("WM_DELETE_WINDOW", self.stop_camera)

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ===== HOME =====
    def create_home(self):
        self.clear()

        title = tk.Label(
            self.root,
            text="GesNova",
            font=("Helvetica", 40, "bold"),
            fg="white",
            bg="#0f172a"
        )
        title.pack(pady=100)

        start_btn = tk.Button(
            self.root,
            text="Start ISL Conversion",
            font=("Helvetica", 16),
            bg="#22c55e",
            fg="white",
            padx=20,
            pady=10,
            command=self.start_camera
        )
        start_btn.pack()

    # ===== START CAMERA =====
    def start_camera(self):
        self.clear()

        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Mac fix
        self.running = True

        self.video_label = tk.Label(self.root, bg="#0f172a")
        self.video_label.pack()

        self.pred_label = tk.Label(
            self.root,
            text="Starting...",
            font=("Helvetica", 20),
            fg="white",
            bg="#0f172a"
        )
        self.pred_label.pack(pady=10)

        self.update_frame()

    # ===== UPDATE LOOP =====
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # Frame skipping for stability
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            self.root.after(30, self.update_frame)
            return

        features, _ = extract_features(frame)

        text = "No Hand"

        if np.any(features):
            pred = model.predict(features.reshape(1, -1), verbose=0)
            label = labels[np.argmax(pred)]
            conf = float(np.max(pred))

            if conf > 0.6:
                text = f"{label} ({conf:.2f})"
            else:
                text = "Uncertain"

        self.pred_label.config(text=text)

        # Convert frame safely
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk  # prevent memory leak

        self.root.after(30, self.update_frame)

    # ===== STOP CAMERA =====
    def stop_camera(self):
        self.running = False

        if self.cap:
            self.cap.release()

        self.root.destroy()


# ===== RUN =====
root = tk.Tk()
app = GesNovaApp(root)
root.mainloop()