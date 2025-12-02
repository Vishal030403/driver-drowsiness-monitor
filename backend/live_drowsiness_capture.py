import cv2
import numpy as np
import tensorflow as tf

from model_loader import DrowsinessDetector


# ---------- Helper functions (local to this script) ----------

def detect_face_local(frame, face_cascade):
    """Detect the largest face in the frame and return ROI + coordinates."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, None

    # Pick the largest face (in case of multiple)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_roi = frame[y:y + h, x:x + w]
    return face_roi, (x, y, w, h)


def preprocess_face_roi(face_roi, target_size=(224, 224)):
    """Resize, convert, normalize and add batch dimension."""
    face = cv2.resize(face_roi, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face


# ---------- Main script ----------

def main():
    # Path relative to the backend folder where you're running the script
    model_path = "model/driver_drowsiness_final_model.keras"

    # Use your existing DrowsinessDetector (DO NOT TOUCH model_loader.py)
    detector = DrowsinessDetector(model_path=model_path)

    # Get the loaded Keras model
    model = detector.model

    # Try to reuse face cascade from DrowsinessDetector, otherwise load our own
    if hasattr(detector, "face_cascade") and detector.face_cascade is not None:
        face_cascade = detector.face_cascade
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("=" * 60)
    print("📷 Live Driver Drowsiness Detection")
    print("=" * 60)
    print("Instructions:")
    print("- A window will open showing your camera feed")
    print("- A GREEN rectangle will appear around the detected face")
    print("- Text will show: ALERT / DROWSY + confidence")
    print("- Press 'q' to quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera.")
                break

            face_roi, coords = detect_face_local(frame, face_cascade)

            label_text = "NO FACE"
            color = (0, 0, 255)  # default red

            if face_roi is not None and coords is not None:
                x, y, w, h = coords

                # Preprocess and predict
                input_tensor = preprocess_face_roi(face_roi)
                preds = model.predict(input_tensor, verbose=0)
                prob = float(preds[0][0])

                if prob >= 0.5:
                    status = "DROWSY"
                    color = (0, 0, 255)  # red
                else:
                    status = "ALERT"
                    color = (0, 255, 0)  # green

                label_text = f"{status}: {prob * 100:.1f}%"

                # Draw green/red rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Put label above the rectangle
                cv2.putText(
                    frame,
                    label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                # No face detected
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Driver Drowsiness Detection", frame)

            


            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Camera released, window closed.")


if __name__ == "__main__":
    main()
