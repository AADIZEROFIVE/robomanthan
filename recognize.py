# import cv2, numpy as np
# from mediapipe import solutions as mp

# mp_face_detection = mp.face_detection
# detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read("trainer.yml")
# names = np.load("names.npy")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = detector.process(rgb)

#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x, y, w_, h_ = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
#             gray = cv2.cvtColor(frame[y:y+h_, x:x+w_], cv2.COLOR_BGR2GRAY)
#             id_, conf = face_recognizer.predict(gray)
#             name = names[id_] if conf < 80 else "Unknown"
#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x+w_, y+h_), color, 2)
#             cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# from mediapipe import solutions as mp

# # Initialize MediaPipe Face Detection
# mp_face_detection = mp.face_detection
# detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# # Load trained face recognizer and names
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read("trainer.yml")
# names = np.load("names.npy", allow_pickle=True)

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Camera frame not received.")
#         break

#     # Convert frame to RGB for MediaPipe
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = detector.process(rgb)

#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x = int(bbox.xmin * w)
#             y = int(bbox.ymin * h)
#             w_ = int(bbox.width * w)
#             h_ = int(bbox.height * h)

#             # ✅ Clamp coordinates to valid image range
#             x, y = max(0, x), max(0, y)
#             w_ = min(w_ , w - x)
#             h_ = min(h_ , h - y)

#             # Skip invalid or zero-sized crops
#             if w_ <= 0 or h_ <= 0:
#                 continue

#             face_crop = frame[y:y+h_, x:x+w_]
#             if face_crop.size == 0:
#                 continue

#             gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

#             try:
#                 id_, conf = face_recognizer.predict(gray)
#                 name = names[id_] if conf < 80 else "Unknown"
#             except Exception as e:
#                 name = "Unknown"

#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x+w_, y+h_), color, 2)
#             cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






# recognize_with_register.py
import os
import cv2
import time
import numpy as np
import mediapipe as mp

# CONFIG
DATASET_DIR = "dataset"
MODEL_FILE = "trainer.yml"
NAMES_FILE = "names.npy"
SAMPLES_PER_PERSON = 30
CAPTURE_DELAY = 0.18  # seconds between saved frames

# initialize mediapipe detector
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Ensure dataset dir exists
os.makedirs(DATASET_DIR, exist_ok=True)

# create recognizer (if opencv-contrib installed)
def create_recognizer():
    try:
        return cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        raise RuntimeError("cv2.face not available. Install opencv-contrib-python.") from e

# Train recognizer from images in dataset/
def train_recognizer():
    print("[TRAIN] Scanning dataset and preparing training data...")
    faces = []
    labels = []
    names = []
    label = 0

    for person in sorted(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue
        person_images = sorted(os.listdir(person_path))
        print(f"[TRAIN] {person}: {len(person_images)} images")
        count_added = 0
        for fname in person_images:
            fpath = os.path.join(person_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if not results.detections:
                continue
            # take first detection only for training images
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            h_img, w_img, _ = img.shape
            x = int(bbox.xmin * w_img)
            y = int(bbox.ymin * h_img)
            w_ = int(bbox.width * w_img)
            h_ = int(bbox.height * h_img)
            # clamp
            x, y = max(0, x), max(0, y)
            w_ = min(w_, w_img - x)
            h_ = min(h_, h_img - y)
            if w_ <= 0 or h_ <= 0:
                continue
            face = img[y:y+h_, x:x+w_]
            if face.size == 0:
                continue
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(gray)
            labels.append(label)
            count_added += 1
        if count_added == 0:
            print(f"[TRAIN] WARNING: No valid faces found for {person}. Skipping.")
            continue
        names.append(person)
        label += 1

    if len(faces) == 0:
        print("[TRAIN] No faces found in dataset. Training aborted.")
        return None, None

    recognizer = create_recognizer()
    print("[TRAIN] Training recognizer on", len(faces), "faces ...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    np.save(NAMES_FILE, np.array(names))
    print("[TRAIN] Done. Model saved to", MODEL_FILE)
    return recognizer, names

# Load existing model if available
def load_trainer_if_exists():
    if os.path.exists(MODEL_FILE) and os.path.exists(NAMES_FILE):
        recognizer = create_recognizer()
        recognizer.read(MODEL_FILE)
        names = np.load(NAMES_FILE, allow_pickle=True)
        print("[INFO] Loaded existing model:", MODEL_FILE)
        return recognizer, names
    else:
        print("[INFO] No existing model found. Train after adding people.")
        return None, np.array([])

# Utility: capture samples for a name
def capture_samples_for_name(name, samples=SAMPLES_PER_PERSON):
    pdir = os.path.join(DATASET_DIR, name)
    os.makedirs(pdir, exist_ok=True)
    # find next available index
    existing = [int(os.path.splitext(f)[0]) for f in os.listdir(pdir) if f.split(".")[0].isdigit()]
    start_idx = max(existing) + 1 if existing else 1

    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        print("⚠️ Camera not available for capturing samples.")
        return
    print(f"[CAPTURE] Starting capture for '{name}' - {samples} samples.")
    print("Look at the camera. Capture begins in 3 seconds...")
    time.sleep(3)

    count = 0
    idx = start_idx
    while count < samples:
        ret, frame = cap_local.read()
        if not ret:
            print("[CAPTURE] Frame not received, retrying...")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        display = frame.copy()

        if results.detections:
            # choose largest detection (closest)
            best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bbox = best.location_data.relative_bounding_box
            h_f, w_f, _ = frame.shape
            x = int(bbox.xmin * w_f)
            y = int(bbox.ymin * h_f)
            w_ = int(bbox.width * w_f)
            h_ = int(bbox.height * h_f)
            # clamp
            x, y = max(0, x), max(0, y)
            w_ = min(w_, w_f - x)
            h_ = min(h_, h_f - y)
            if w_ > 0 and h_ > 0:
                face = frame[y:y+h_, x:x+w_]
                if face.size != 0:
                    # save face crop (resize to reasonable size)
                    face_resized = cv2.resize(face, (200, 200))
                    out_path = os.path.join(pdir, f"{idx}.jpg")
                    cv2.imwrite(out_path, face_resized)
                    idx += 1
                    count += 1
                    cv2.rectangle(display, (x, y), (x+w_, y+h_), (0, 255, 0), 2)
                    cv2.putText(display, f"Saved {count}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow(f"Capture - {name}", display)
        if cv2.waitKey(int(CAPTURE_DELAY * 1000)) & 0xFF == ord('q'):
            print("[CAPTURE] User aborted capture.")
            break

    cap_local.release()
    cv2.destroyWindow(f"Capture - {name}")
    print(f"[CAPTURE] Finished capturing {count} samples for '{name}'.")

# Main loop
if __name__ == "__main__":
    recognizer, names = load_trainer_if_exists()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not detected!")

    print("Controls:  [n] Register new person   [q] Quit")
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received from camera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            display = frame.copy()

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    w_ = int(bbox.width * w)
                    h_ = int(bbox.height * h)
                    x, y = max(0, x), max(0, y)
                    w_ = min(w_, w - x)
                    h_ = min(h_, h - y)
                    if w_ <= 0 or h_ <= 0:
                        continue
                    face_crop = frame[y:y+h_, x:x+w_]
                    if face_crop.size == 0:
                        continue

                    if recognizer is not None and len(names) > 0:
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        try:
                            id_, conf = recognizer.predict(gray)
                            # lower conf = better match for LBPH; tune threshold as needed
                            if conf < 80 and 0 <= id_ < len(names):
                                name = str(names[id_])
                            else:
                                name = "Unknown"
                        except Exception:
                            name = "Unknown"
                    else:
                        name = "No model"

                    color = (0, 255, 0) if name not in ("Unknown", "No model") else (0, 0, 255)
                    cv2.rectangle(display, (x, y), (x+w_, y+h_), color, 2)
                    cv2.putText(display, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition (press 'n' to register)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):
                # register new user
                print("\n[REGISTER] Enter name for new person (no spaces): ", end="", flush=True)
                new_name = input().strip()
                if new_name == "":
                    print("[REGISTER] Invalid name, skipping.")
                    continue
                # capture samples
                capture_samples_for_name(new_name, samples=SAMPLES_PER_PERSON)
                # retrain model
                recognizer, names = train_recognizer()
                if recognizer is None:
                    print("[REGISTER] Training failed or no data. Model not updated.")
                else:
                    print("[REGISTER] Model updated. New names:", names)
            elif key == ord('q'):
                print("Quitting...")
                break

    cap.release()
    cv2.destroyAllWindows()
