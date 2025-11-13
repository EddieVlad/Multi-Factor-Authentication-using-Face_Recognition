import cv2
import numpy as np
from PIL import Image
import os
import sqlite3
import hashlib
import tkinter as tk
from tkinter import messagebox

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = "users.db"
DATASET_PATH = "dataset"
MODEL_PATH = "trainer.yml"

# -----------------------------
# DB & PASSWORD HELPERS
# -----------------------------
def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def create_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password)),
        )
        conn.commit()
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        user_id = None
    conn.close()
    return user_id

def get_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row  # (id, username, password_hash) or None

# -----------------------------
# FACE CAPTURE & TRAINING
# -----------------------------
def capture_faces_for_user(user_id: int, max_images: int = 150):
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        messagebox.showerror("Camera Error", "Could not open camera.")
        return False

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    messagebox.showinfo(
        "Face Capture",
        "Face capture will start now.\n\nLook at the camera and slowly move your head.\nPress Q to stop early."
    )

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(
                f"{DATASET_PATH}/User.{user_id}.{count}.jpg",
                gray[y:y + h, x:x + w]
            )

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"Images: {count}/{max_images}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Capturing Faces", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or count >= max_images:
            break

    cam.release()
    cv2.destroyAllWindows()

    if count == 0:
        messagebox.showwarning("Capture Failed", "No faces captured.")
        return False

    return True

def get_images_and_labels(path: str):
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    face_samples = []
    ids = []

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert("L")  # grayscale
        img_numpy = np.array(PIL_img, "uint8")

        try:
            filename = os.path.split(image_path)[-1]
            id_ = int(filename.split(".")[1])  # User.<id>.<n>.jpg
        except Exception:
            continue

        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id_)

    return face_samples, ids

def train_model():
    if not os.path.isdir(DATASET_PATH):
        messagebox.showerror("Training Error", "Dataset folder not found.")
        return False

    faces, ids = get_images_and_labels(DATASET_PATH)
    if len(faces) == 0:
        messagebox.showerror("Training Error", "No faces found in dataset.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(MODEL_PATH)
    return True

# -----------------------------
# FACE VERIFICATION
# -----------------------------
def verify_face_for_user(user_id: int, timeout_frames: int = 200) -> bool:
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Model Error", "Model not trained yet.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        messagebox.showerror("Camera Error", "Could not open camera.")
        return False

    messagebox.showinfo(
        "Face Verification",
        "Face verification started.\nLook at the camera.\nPress Q to cancel."
    )

    frames_checked = 0
    success_count = 0
    REQUIRED_SUCCESSES = 5
    CONF_THRESHOLD = 60.0  # lower = better

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frames_checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            predicted_id, conf = recognizer.predict(roi)

            if predicted_id == user_id and conf < CONF_THRESHOLD:
                success_count += 1
                color = (0, 255, 0)
                text = f"User {predicted_id} OK ({conf:.1f})"
            else:
                color = (0, 0, 255)
                text = "Not recognized"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Face Verification", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam.release()
            cv2.destroyAllWindows()
            return False

        if success_count >= REQUIRED_SUCCESSES:
            cam.release()
            cv2.destroyAllWindows()
            return True

        if frames_checked >= timeout_frames:
            break

    cam.release()
    cv2.destroyAllWindows()
    return False

# -----------------------------
# TKINTER UI
# -----------------------------
def handle_signup(username_entry, password_entry):
    username = username_entry.get().strip()
    password = password_entry.get().strip()

    if not username or not password:
        messagebox.showwarning("Input Error", "Username and password are required.")
        return

    user_id = create_user(username, password)
    if user_id is None:
        messagebox.showerror("Signup Error", "Username already exists.")
        return

    # Capture face and train model
    if not capture_faces_for_user(user_id):
        messagebox.showerror("Signup Error", "Face capture failed.")
        return

    if train_model():
        messagebox.showinfo("Success", "Signup complete with face data!")
    else:
        messagebox.showerror("Signup Error", "Training failed.")

def handle_login(username_entry, password_entry):
    username = username_entry.get().strip()
    password = password_entry.get().strip()

    if not username or not password:
        messagebox.showwarning("Input Error", "Username and password are required.")
        return

    user = get_user(username)
    if user is None:
        messagebox.showerror("Login Error", "User not found.")
        return

    user_id, _, stored_hash = user
    if stored_hash != hash_password(password):
        messagebox.showerror("Login Error", "Incorrect password.")
        return

    # Password correct -> now face verification
    ok = verify_face_for_user(user_id)
    if ok:
        messagebox.showinfo("Login Success", "MFA successful! Access granted.")
    else:
        messagebox.showerror("Login Failed", "Face verification failed.")

def build_ui():
    root = tk.Tk()
    root.title("MFA Demo - Password + Face")

    # SIGNUP FRAME
    signup_frame = tk.LabelFrame(root, text="Sign Up", padx=10, pady=10)
    signup_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    tk.Label(signup_frame, text="Username:").grid(row=0, column=0, sticky="e")
    signup_username = tk.Entry(signup_frame, width=25)
    signup_username.grid(row=0, column=1)

    tk.Label(signup_frame, text="Password:").grid(row=1, column=0, sticky="e")
    signup_password = tk.Entry(signup_frame, show="*", width=25)
    signup_password.grid(row=1, column=1)

    tk.Button(
        signup_frame,
        text="Sign Up (Password + Face)",
        command=lambda: handle_signup(signup_username, signup_password),
        width=25
    ).grid(row=2, column=0, columnspan=2, pady=5)

    # LOGIN FRAME
    login_frame = tk.LabelFrame(root, text="Login (MFA)", padx=10, pady=10)
    login_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    tk.Label(login_frame, text="Username:").grid(row=0, column=0, sticky="e")
    login_username = tk.Entry(login_frame, width=25)
    login_username.grid(row=0, column=1)

    tk.Label(login_frame, text="Password:").grid(row=1, column=0, sticky="e")
    login_password = tk.Entry(login_frame, show="*", width=25)
    login_password.grid(row=1, column=1)

    tk.Button(
        login_frame,
        text="Login with MFA",
        command=lambda: handle_login(login_username, login_password),
        width=25
    ).grid(row=2, column=0, columnspan=2, pady=5)

    root.resizable(False, False)
    return root

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    create_db()
    ui = build_ui()
    ui.mainloop()
