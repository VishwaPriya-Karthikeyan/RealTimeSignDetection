import cv2
import os
import time

# === Configuration ===
DATA_DIR = './data'
NUM_CLASSES = 26       # A-Z
NUM_IMAGES = 100       # Images per letter
IMG_SIZE = 64          # Resize images to 64x64

# === Create data directory if not exists ===
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# === Open Webcam ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows

# === Check if camera is working ===
if not cap.isOpened():
    print("❌ Cannot open webcam. Please check camera settings or permissions.")
    exit()

for i in range(NUM_CLASSES):
    label = chr(ord('A') + i)  # Converts 0 to 'A', 1 to 'B', ...
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    print(f"\nShow hand gesture for letter '{label}' in front of the webcam.")
    print("Starting capture in 3 seconds...")
    time.sleep(3)

    count = 0
    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame. Exiting...")
            break

        # Resize the image for model training later
        roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        filepath = os.path.join(label_dir, f"{count}.jpg")
        cv2.imwrite(filepath, roi)

        # Display capture status
        cv2.putText(frame, f"Letter: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Image {count+1}/{NUM_IMAGES}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Dataset Collection", frame)

        # Wait 100 ms or exit if 'q' is pressed
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            print("🚪 Exiting early as requested...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        count += 1

print("\n✅ Dataset collection complete! All letters captured.")
cap.release()
cv2.destroyAllWindows()
