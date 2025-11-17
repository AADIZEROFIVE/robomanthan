import os, cv2, time, numpy as np
from flask import Flask, request, jsonify, Response
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1

# CONFIG
DATASET_DIR = "dataset"
THRESHOLD = 0.55  # lower -> stricter matching

# Setup
app = Flask(__name__)
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------- FACE UTILS -------------------- #
def get_embedding(face_img):
    face = cv2.resize(face_img, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(face, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    tensor = (tensor - 127.5) / 128.0
    with torch.no_grad():
        emb = model(tensor.to(device)).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

def capture_face(name):
    pdir = os.path.join(DATASET_DIR, name)
    os.makedirs(pdir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"[CAPTURE] Capturing 30 faces for {name}...")
    count = 0
    while count < 30:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_, h_ = int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            w_, h_ = min(w_, w - x), min(h_, h - y)
            face = frame[y:y+h_, x:x+w_]
            if face.size > 0:
                out_path = os.path.join(pdir, f"{count+1}.jpg")
                cv2.imwrite(out_path, face)
                count += 1
                cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[CAPTURE] Done capturing for {name}")

def build_database():
    print("[DB] Building face embeddings...")
    db = {}
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue
        embeddings = []
        for f in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, f))
            if img is None:
                continue
            emb = get_embedding(img)
            embeddings.append(emb)
        if embeddings:
            db[person] = np.mean(embeddings, axis=0)
    np.save("embeddings.npy", db)
    print("[DB] Saved embeddings for", len(db), "people.")
    return db

def load_db():
    if os.path.exists("embeddings.npy"):
        db = np.load("embeddings.npy", allow_pickle=True).item()
        print("[DB] Loaded database:", list(db.keys()))
        return db
    return {}

def recognize_frame(frame, db):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if not results.detections:
        return frame

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        w_, h_ = int(bbox.width * w), int(bbox.height * h)
        x, y = max(0, x), max(0, y)
        w_, h_ = min(w_, w - x), min(h_, h - y)
        face = frame[y:y+h_, x:x+w_]
        if face.size == 0:
            continue
        emb = get_embedding(face)

        name, best_sim = "Unknown", 0
        for person, ref_emb in db.items():
            sim = np.dot(emb, ref_emb)
            if sim > best_sim:
                best_sim, name = sim, person
        if best_sim < THRESHOLD:
            name = "Unknown"
        color = (0,255,0) if name!="Unknown" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w_, y+h_), color, 2)
        cv2.putText(frame, f"{name} ({best_sim:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# -------------------- FLASK API -------------------- #
db = load_db()

@app.route("/add_person", methods=["POST"])
def add_person():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"status": "error", "msg": "Name required"}), 400
    capture_face(name)
    global db
    db = build_database()
    return jsonify({"status": "ok", "msg": f"Added {name} and updated database."})

@app.route("/video_feed")
def video_feed():
    cap = cv2.VideoCapture(0)
    def gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = recognize_frame(frame, db)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def home():
    return '''
    <h2>Face Recognition Server</h2>
    <form action="/add_person" method="post">
        <input name="name" placeholder="Enter name" required>
        <button type="submit">Add Person</button>
    </form>
    <br>
    <a href="/video_feed">Start Video</a>
    '''

if __name__ == "__main__":
    print("[INFO] Starting Face Recognition Server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
