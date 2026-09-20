"""
Face Attendance System - Flask Web Server
All endpoints for the GUI dashboard
"""
import os
import sys
import json
import base64
import threading
import time
from datetime import datetime, date
from flask import Flask, jsonify, request, send_from_directory, render_template_string

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.face_engine import (
    get_face_detector, get_recognizer, detect_faces,
    preprocess_face, recognize_face, train_model_from_data,
    load_users, save_users, load_labels, save_labels,
    get_model_status, DATA_DIR, MODEL_FILE
)
from utils.attendance import (
    check_in, check_out, get_today_records, get_records_for_date,
    get_all_records, get_available_dates, get_analytics,
    get_employee_history, load_settings, save_settings,
    load_active_sessions
)

import cv2
import numpy as np
import pickle

app = Flask(__name__)

# ─── Global Camera State ─────────────────────────────────────────────────────
camera_state = {
    "active": False,
    "mode": None,          # "recognize" | "train"
    "train_emp_id": None,
    "train_emp_name": None,
    "train_count": 0,
    "train_total": 50,
    "last_frame_b64": None,
    "last_detection": None,
    "error": None,
    "thread": None
}
camera_lock = threading.Lock()

# ─── Static HTML Route ────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

# ─── Employees API ───────────────────────────────────────────────────────────

@app.route("/api/employees", methods=["GET"])
def get_employees():
    users = load_users()
    return jsonify({"employees": users})


@app.route("/api/employees", methods=["POST"])
def add_employee():
    data = request.json
    emp_id = data.get("id", "").strip()
    name = data.get("name", "").strip()
    dept = data.get("department", "General").strip()
    
    if not emp_id or not name:
        return jsonify({"success": False, "message": "ID and Name are required"}), 400
    
    users = load_users()
    if emp_id in users:
        return jsonify({"success": False, "message": f"Employee ID {emp_id} already exists"}), 400
    
    # Assign a numeric label for LBPH
    existing_labels = [u.get("label_id", 0) for u in users.values()]
    new_label = max(existing_labels, default=0) + 1
    
    users[emp_id] = {
        "name": name,
        "department": dept,
        "label_id": new_label,
        "registered_at": datetime.now().isoformat(),
        "trained": False
    }
    save_users(users)
    
    id_to_name, name_to_id = load_labels()
    id_to_name[new_label] = name
    name_to_id[name] = new_label
    save_labels(id_to_name, name_to_id)
    
    return jsonify({"success": True, "message": f"Employee {name} added", "employee": users[emp_id]})


@app.route("/api/employees/<emp_id>", methods=["DELETE"])
def delete_employee(emp_id):
    users = load_users()
    if emp_id not in users:
        return jsonify({"success": False, "message": "Employee not found"}), 404
    
    name = users[emp_id]["name"]
    del users[emp_id]
    save_users(users)
    
    # Remove training data dir
    import shutil
    emp_dir = os.path.join(DATA_DIR, f"emp_{emp_id}")
    if os.path.exists(emp_dir):
        shutil.rmtree(emp_dir)
    
    return jsonify({"success": True, "message": f"Employee {name} deleted"})


# ─── Training API ─────────────────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def start_training():
    data = request.json
    emp_id = data.get("employee_id")
    
    users = load_users()
    if emp_id not in users:
        return jsonify({"success": False, "message": "Employee not found"}), 404
    
    with camera_lock:
        if camera_state["active"]:
            return jsonify({"success": False, "message": "Camera already in use"}), 409
        
        camera_state["active"] = True
        camera_state["mode"] = "train"
        camera_state["train_emp_id"] = emp_id
        camera_state["train_emp_name"] = users[emp_id]["name"]
        camera_state["train_count"] = 0
        camera_state["train_total"] = data.get("samples", 50)
        camera_state["error"] = None
    
    thread = threading.Thread(target=_run_training_capture, daemon=True)
    thread.start()
    camera_state["thread"] = thread
    
    return jsonify({"success": True, "message": f"Training started for {users[emp_id]['name']}"})


def _run_training_capture():
    detector = get_face_detector()
    emp_id = camera_state["train_emp_id"]
    emp_name = camera_state["train_emp_name"]
    total = camera_state["train_total"]
    
    emp_dir = os.path.join(DATA_DIR, f"emp_{emp_id}")
    os.makedirs(emp_dir, exist_ok=True)
    
    # Clear old samples
    for f in os.listdir(emp_dir):
        os.remove(os.path.join(emp_dir, f))
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    try:
        while count < total and camera_state["active"] and camera_state["mode"] == "train":
            ret, frame = cap.read()
            if not ret:
                break
            
            faces, gray = detect_faces(frame, detector)
            display = frame.copy()
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                processed = preprocess_face(face_roi)
                
                img_path = os.path.join(emp_dir, f"sample_{count}.jpg")
                cv2.imwrite(img_path, processed)
                count += 1
                camera_state["train_count"] = count
                
                # Draw on display
                color = (0, 255, 0)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, f"Captured: {count}/{total}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                break
            
            # Draw progress bar
            prog = int((count / total) * frame.shape[1])
            cv2.rectangle(display, (0, frame.shape[0]-20), (prog, frame.shape[0]), (0, 255, 100), -1)
            cv2.putText(display, f"Training: {emp_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Encode frame
            _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            camera_state["last_frame_b64"] = base64.b64encode(buf).decode()
            
            time.sleep(0.05)
    
    except Exception as e:
        camera_state["error"] = str(e)
    finally:
        cap.release()
        
        if count >= total:
            # Mark employee as trained
            users = load_users()
            if emp_id in users:
                users[emp_id]["trained"] = True
                users[emp_id]["sample_count"] = count
                save_users(users)
        
        camera_state["active"] = False
        camera_state["mode"] = None


@app.route("/api/train/status", methods=["GET"])
def training_status():
    return jsonify({
        "active": camera_state["active"],
        "mode": camera_state["mode"],
        "count": camera_state["train_count"],
        "total": camera_state["train_total"],
        "frame": camera_state.get("last_frame_b64"),
        "error": camera_state.get("error")
    })


@app.route("/api/train/model", methods=["POST"])
def train_model():
    success, message = train_model_from_data()
    return jsonify({"success": success, "message": message})


@app.route("/api/train/stop", methods=["POST"])
def stop_training():
    camera_state["active"] = False
    return jsonify({"success": True, "message": "Training stopped"})


# ─── Recognition / Attendance API ────────────────────────────────────────────

@app.route("/api/recognize/start", methods=["POST"])
def start_recognition():
    with camera_lock:
        if camera_state["active"]:
            return jsonify({"success": False, "message": "Camera already in use"}), 409
        camera_state["active"] = True
        camera_state["mode"] = "recognize"
        camera_state["last_detection"] = None
        camera_state["error"] = None
    
    thread = threading.Thread(target=_run_recognition, daemon=True)
    thread.start()
    camera_state["thread"] = thread
    return jsonify({"success": True, "message": "Recognition started"})


def _run_recognition():
    detector = get_face_detector()
    cap = cv2.VideoCapture(0)
    
    # Cooldown per employee to prevent rapid duplicate scans
    last_scan = {}  # emp_id -> timestamp
    COOLDOWN = 5  # seconds
    
    try:
        while camera_state["active"] and camera_state["mode"] == "recognize":
            ret, frame = cap.read()
            if not ret:
                break
            
            faces, gray = detect_faces(frame, detector)
            display = frame.copy()
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emp_id, confidence = recognize_face(face_roi)
                
                users = load_users()
                now = time.time()
                
                if emp_id and emp_id in users:
                    name = users[emp_id]["name"]
                    dept = users[emp_id]["department"]
                    color = (0, 255, 0)
                    label_text = f"{name} ({confidence:.1f})"
                    
                    # Check cooldown
                    if now - last_scan.get(emp_id, 0) > COOLDOWN:
                        last_scan[emp_id] = now
                        
                        # Determine check-in or check-out
                        sessions = load_active_sessions()
                        action = "CHECK_OUT" if str(emp_id) in sessions else "CHECK_IN"
                        
                        if action == "CHECK_IN":
                            ok, msg, rec = check_in(emp_id, name, dept)
                        else:
                            ok, msg, rec = check_out(emp_id, name)
                        
                        camera_state["last_detection"] = {
                            "emp_id": emp_id,
                            "name": name,
                            "department": dept,
                            "action": action,
                            "success": ok,
                            "message": msg,
                            "record": rec,
                            "timestamp": datetime.now().isoformat()
                        }
                else:
                    color = (0, 0, 255)
                    label_text = f"Unknown ({confidence:.1f})"
                
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, label_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(display, "RECOGNITION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, datetime.now().strftime("%H:%M:%S"), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            camera_state["last_frame_b64"] = base64.b64encode(buf).decode()
            
            time.sleep(0.03)
    
    except Exception as e:
        camera_state["error"] = str(e)
    finally:
        cap.release()
        camera_state["active"] = False
        camera_state["mode"] = None


@app.route("/api/recognize/status", methods=["GET"])
def recognition_status():
    return jsonify({
        "active": camera_state["active"],
        "mode": camera_state["mode"],
        "frame": camera_state.get("last_frame_b64"),
        "detection": camera_state.get("last_detection"),
        "error": camera_state.get("error")
    })


@app.route("/api/recognize/stop", methods=["POST"])
def stop_recognition():
    camera_state["active"] = False
    return jsonify({"success": True, "message": "Recognition stopped"})


# ─── Manual Attendance ────────────────────────────────────────────────────────

@app.route("/api/attendance/checkin", methods=["POST"])
def manual_checkin():
    data = request.json
    emp_id = data.get("employee_id")
    users = load_users()
    if emp_id not in users:
        return jsonify({"success": False, "message": "Employee not found"}), 404
    
    emp = users[emp_id]
    ok, msg, rec = check_in(emp_id, emp["name"], emp["department"], notes="Manual")
    return jsonify({"success": ok, "message": msg, "record": rec})


@app.route("/api/attendance/checkout", methods=["POST"])
def manual_checkout():
    data = request.json
    emp_id = data.get("employee_id")
    users = load_users()
    if emp_id not in users:
        return jsonify({"success": False, "message": "Employee not found"}), 404
    
    emp = users[emp_id]
    ok, msg, rec = check_out(emp_id, emp["name"], notes="Manual")
    return jsonify({"success": ok, "message": msg, "record": rec})


# ─── Records & Analytics API ─────────────────────────────────────────────────

@app.route("/api/attendance/today", methods=["GET"])
def today_attendance():
    records = get_today_records()
    sessions = load_active_sessions()
    return jsonify({
        "records": records,
        "active_sessions": sessions,
        "date": date.today().isoformat()
    })


@app.route("/api/attendance/date/<date_str>", methods=["GET"])
def date_attendance(date_str):
    records = get_records_for_date(date_str)
    return jsonify({"records": records, "date": date_str})


@app.route("/api/attendance/dates", methods=["GET"])
def available_dates():
    return jsonify({"dates": get_available_dates()})


@app.route("/api/attendance/employee/<emp_id>", methods=["GET"])
def employee_history(emp_id):
    days = int(request.args.get("days", 30))
    records = get_employee_history(emp_id, days)
    return jsonify({"records": records, "employee_id": emp_id})


@app.route("/api/analytics", methods=["GET"])
def analytics():
    days = int(request.args.get("days", 30))
    data = get_analytics(days)
    return jsonify(data)


@app.route("/api/model/status", methods=["GET"])
def model_status():
    status = get_model_status()
    users = load_users()
    trained_count = sum(1 for u in users.values() if u.get("trained"))
    status["trained_employees"] = trained_count
    return jsonify(status)


# ─── Settings API ─────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
def get_settings_api():
    return jsonify(load_settings())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    settings = load_settings()
    settings.update(data)
    save_settings(settings)
    return jsonify({"success": True, "message": "Settings saved", "settings": settings})


if __name__ == "__main__":
    os.makedirs(STATIC_DIR, exist_ok=True)
    print("\n" + "="*50)
    print("  Face Attendance System")
    print("  Open: http://localhost:5050")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
