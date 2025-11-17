# import cv2, os, numpy as np
# from mediapipe import solutions as mp

# mp_face_detection = mp.face_detection
# detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# faces, labels, names = [], [], []
# label = 0

# for person in os.listdir("dataset"):
#     person_path = os.path.join("dataset", person)
#     if not os.path.isdir(person_path):
#         continue
#     print(f"Training {person}...")
#     for file in os.listdir(person_path):
#         img_path = os.path.join(person_path, file)
#         image = cv2.imread(img_path)
#         if image is None:
#             continue
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         result = detector.process(rgb)
#         if result.detections:
#             for detection in result.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 h, w, _ = image.shape
#                 x, y, w_, h_ = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
#                 face = image[y:y+h_, x:x+w_]
#                 gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                 faces.append(gray)
#                 labels.append(label)
#     names.append(person)
#     label += 1

# print("Training complete, saving model...")
# face_recognizer.train(faces, np.array(labels))
# face_recognizer.save("trainer.yml")
# np.save("names.npy", np.array(names))
# print("✅ Model saved as trainer.yml")



import cv2
import os
import numpy as np
from mediapipe import solutions as mp

# Initialize Mediapipe face detector
mp_face_detection = mp.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces, labels, names = [], [], []
label = 0

print("📂 Starting dataset training...\n")

for person in os.listdir("dataset"):
    person_path = os.path.join("dataset", person)
    if not os.path.isdir(person_path):
        continue

    print(f"🧠 Training {person}...")
    valid_count = 0

    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        image = cv2.imread(img_path)

        # Skip unreadable files
        if image is None:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue

        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        # Skip if no face found
        if not result.detections:
            print(f"⚠️ No face detected in: {img_path}")
            continue

        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape

            # Convert to absolute coordinates and clamp to image boundaries
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            w_ = int(bbox.width * w)
            h_ = int(bbox.height * h)
            x2 = min(w, x + w_)
            y2 = min(h, y + h_)

            # Crop face region safely
            face = image[y:y2, x:x2]
            if face.size == 0:
                print(f"⚠️ Invalid crop in: {img_path}")
                continue

            # Convert to grayscale for training
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(gray)
            labels.append(label)
            valid_count += 1

    if valid_count > 0:
        names.append(person)
        label += 1
        print(f"✅ {person} trained with {valid_count} valid faces.\n")
    else:
        print(f"❌ No valid faces found for {person}, skipped.\n")

# Train and save model
if len(faces) == 0:
    print("🚫 No faces found in the entire dataset. Please check your images.")
else:
    print("🧩 Training LBPH recognizer...")
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save("trainer.yml")
    np.save("names.npy", np.array(names))
    print("\n✅ Training complete!")
    print("📦 Model saved as trainer.yml")
    print("📦 Names saved as names.npy")
