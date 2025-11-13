import cv2
import numpy as np
from PIL import Image
import os

# Folder where capture_face.py saved images
DATASET_PATH = "dataset"
MODEL_PATH = "trainer.yml"

# Face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def getImagesAndLabels(path):
    imagePaths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    faceSamples = []
    ids = []

    print(f"[INFO] Found {len(imagePaths)} image files in '{path}'")

    for imagePath in imagePaths:
        # convert image to grayscale
        PIL_img = Image.open(imagePath).convert("L")
        img_numpy = np.array(PIL_img, "uint8")

        # Expect filename like: User.1.1.jpg → id = 1
        try:
            filename = os.path.split(imagePath)[-1]
            id_ = int(filename.split(".")[1])
        except Exception:
            print(f"[WARN] Skipping '{imagePath}' (filename format not User.<id>.<n>.jpg)")
            continue

        faces = detector.detectMultiScale(img_numpy)

        if len(faces) == 0:
            print(f"[WARN] No face detected in '{imagePath}', skipping")
            continue

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id_)

    return faceSamples, ids


if __name__ == "__main__":
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Dataset folder '{DATASET_PATH}' not found.")
        raise SystemExit(1)

    print("[INFO] Starting training...")
    faces, ids = getImagesAndLabels(DATASET_PATH)

    if len(faces) == 0:
        print("[ERROR] No faces found in dataset. Did you run capture_face.py?")
        raise SystemExit(1)

    print(f"[INFO] Training on {len(ids)} samples, {len(set(ids))} unique IDs")

    # IMPORTANT: this requires opencv-contrib-python, not just opencv-python
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(MODEL_PATH)

    print(f"[INFO] Training completed. Model saved as '{MODEL_PATH}'")
