
import cv2
import numpy as np
from mediapipe import solutions as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load trained face recognizer and names
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainer.yml")
names = np.load("names.npy", allow_pickle=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not received.")
        break

    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_ = int(bbox.width * w)
            h_ = int(bbox.height * h)

            # ✅ Clamp coordinates to valid image range
            x, y = max(0, x), max(0, y)
            w_ = min(w_ , w - x)
            h_ = min(h_ , h - y)

            # Skip invalid or zero-sized crops
            if w_ <= 0 or h_ <= 0:
                continue

            face_crop = frame[y:y+h_, x:x+w_]
            if face_crop.size == 0:
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            try:
                id_, conf = face_recognizer.predict(gray)
                name = names[id_] if conf < 80 else "Unknown"
            except Exception as e:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w_, y+h_), color, 2)
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

