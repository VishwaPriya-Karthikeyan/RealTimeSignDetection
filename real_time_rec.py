import cv2
import os
import numpy as np
from tkinter import *
from tkinter import simpledialog, messagebox
import mediapipe as mp

# Create folders directory
DATASET_PATH = "gestures_dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Image dimensions
IMG_SIZE = 224

# Capture hand ROI from frame
def get_hand_roi(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_list) * w)
            x_max = int(max(x_list) * w)
            y_min = int(min(y_list) * h)
            y_max = int(max(y_list) * h)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ensure safe crop
            margin = 20
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)

            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                return None, frame
            cropped_resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            return cropped_resized, frame
    return None, frame

# Collect images for a given gesture
def collect_images(gesture_name):
    gesture_path = os.path.join(DATASET_PATH, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    total_imgs = 100

    while count < total_imgs:
        ret, frame = cap.read()
        if not ret:
            break

        hand_img, drawn_frame = get_hand_roi(frame)

        if hand_img is not None:
            filename = os.path.join(gesture_path, f"{gesture_name}_{count}.jpg")
            cv2.imwrite(filename, hand_img)
            count += 1
            cv2.putText(drawn_frame, f"Saved: {count}/{total_imgs}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(drawn_frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Collecting Images", drawn_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Dataset Collection", f"Images for '{gesture_name}' collected successfully!")

# Main GUI
def launch_portal():
    root = Tk()
    root.geometry("400x200")
    root.title("Sign Language Portal")

    def add_gesture():
        gesture = simpledialog.askstring("New Gesture", "Enter the gesture name (e.g., Hello):")
        if gesture:
            collect_images(gesture.strip())

    def ask_to_add():
        answer = messagebox.askyesno("Add Gesture", "Do you want to add a new gesture?")
        if answer:
            add_gesture()

    Label(root, text="Sign Language Recognition", font=("Helvetica", 16)).pack(pady=20)
    Button(root, text="Add New Gesture", command=ask_to_add, width=20, height=2).pack(pady=10)
    Button(root, text="Exit", command=root.destroy, width=20, height=2).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    launch_portal()