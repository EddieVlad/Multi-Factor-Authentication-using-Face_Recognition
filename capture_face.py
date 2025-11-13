import cv2
import os

# Use CAP_DSHOW for Windows camera stability
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

user_id = input("Enter your user ID: ")

print("\n[INFO] Initializing face capture...")
print("[INFO] Please look at the camera and move your head slowly.\n")

count = 0
MAX_IMAGES = 150     # ← increase or decrease this number as needed

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        # Save the image in dataset folder
        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y + h, x:x + w])

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(img, f"Images: {count}/{MAX_IMAGES}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Capturing Faces", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):        # Quit early
        break

    if count >= MAX_IMAGES:   # Stop when enough images captured
        break

print("\n[INFO] Done capturing images")
cam.release()
cv2.destroyAllWindows()
