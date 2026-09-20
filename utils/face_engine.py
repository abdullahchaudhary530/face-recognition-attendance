"""
Face Detection & Recognition Engine
Uses OpenCV LBPH (Local Binary Pattern Histogram) recognizer
No external face_recognition library needed
"""
import cv2
import numpy as np
import os
import pickle
import json
from datetime import datetime

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "face_model.yml")
LABELS_FILE = os.path.join(MODELS_DIR, "labels.pkl")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def get_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


def get_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80
    )
    return recognizer


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "rb") as f:
            return pickle.load(f)
    return {}, {}  # id->name, name->id


def save_labels(id_to_name, name_to_id):
    with open(LABELS_FILE, "wb") as f:
        pickle.dump((id_to_name, name_to_id), f)


def preprocess_face(face_img, size=(200, 200)):
    """Normalize face image for better recognition."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, size)
    # Histogram equalization for lighting invariance
    equalized = cv2.equalizeHist(resized)
    return equalized


def detect_faces(frame, detector, scale=1.1, neighbors=5, min_size=(50, 50)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray


def capture_training_images(employee_id, employee_name, num_samples=50, progress_callback=None):
    """
    Capture face images from webcam for training.
    Returns list of face images.
    """
    detector = get_face_detector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    face_samples = []
    count = 0
    
    # Save dir for this employee
    emp_dir = os.path.join(DATA_DIR, f"emp_{employee_id}")
    os.makedirs(emp_dir, exist_ok=True)
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces, gray = detect_faces(frame, detector)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            processed = preprocess_face(face_roi)
            face_samples.append(processed)
            
            # Save sample image
            img_path = os.path.join(emp_dir, f"sample_{count}.jpg")
            cv2.imwrite(img_path, processed)
            count += 1
            
            if progress_callback:
                progress_callback(count, num_samples, frame, (x, y, w, h))
            
            if count >= num_samples:
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return face_samples


def train_model_from_data():
    """Train LBPH model from all stored employee face data."""
    id_to_name, name_to_id = load_labels()
    users = load_users()
    
    faces = []
    labels = []
    
    for emp_id, info in users.items():
        emp_dir = os.path.join(DATA_DIR, f"emp_{emp_id}")
        if not os.path.exists(emp_dir):
            continue
        
        label = info.get("label_id")
        if label is None:
            continue
            
        for img_file in os.listdir(emp_dir):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(emp_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(label)
    
    if len(faces) < 2:
        return False, "Not enough training data. Add at least 2 employees."
    
    recognizer = get_recognizer()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    
    return True, f"Model trained with {len(faces)} samples from {len(users)} employees."


def recognize_face(face_gray):
    """Returns (employee_id_str, confidence) or (None, 0) if not recognized."""
    if not os.path.exists(MODEL_FILE):
        return None, 0
    
    id_to_name, name_to_id = load_labels()
    users = load_users()
    
    recognizer = get_recognizer()
    recognizer.read(MODEL_FILE)
    
    processed = preprocess_face(face_gray)
    label, confidence = recognizer.predict(processed)
    
    # Lower confidence = better match in LBPH
    if confidence < 80:
        # Find employee by label_id
        for emp_id, info in users.items():
            if info.get("label_id") == label:
                return emp_id, confidence
    
    return None, confidence


def get_model_status():
    users = load_users()
    trained = os.path.exists(MODEL_FILE)
    return {
        "trained": trained,
        "employee_count": len(users),
        "model_path": MODEL_FILE if trained else None
    }
