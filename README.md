# 👁 Face Attendance System

A complete Python-based face recognition attendance system with a web dashboard.

## Features

- 🎥 **Live Face Detection** — Real-time webcam feed with OpenCV
- 🧠 **LBPH Face Recognition** — Train on employee face samples
- ✅ **Auto Check-in / Check-out** — Detects who is in or out
- ⏰ **Late / Early Detection** — Configurable schedule thresholds
- 📋 **CSV Records** — Daily attendance files saved automatically
- 📊 **Dashboard** — Live stats, active sessions, trends
- 📈 **Analytics** — 30-day employee performance summaries
- ⚙️ **Settings** — Configurable work hours and organization name

---

## Project Structure

```
face_attendance/
├── app.py                  ← Flask web server (start this)
├── setup.py                ← Install dependencies
├── requirements.txt
├── static/
│   └── index.html          ← Dashboard GUI
├── utils/
│   ├── face_engine.py      ← OpenCV face detection & recognition
│   └── attendance.py       ← Attendance logic, CSV, analytics
├── data/
│   ├── users.json          ← Employee registry
│   ├── settings.json       ← Work schedule config
│   ├── active_sessions.json ← Who is currently checked in
│   └── emp_EMP001/         ← Face sample images per employee
│       ├── sample_0.jpg
│       └── ...
├── models/
│   ├── face_model.yml      ← Trained LBPH model
│   └── labels.pkl          ← ID ↔ Name mapping
└── records/
    ├── attendance_2024-01-15.csv
    └── attendance_2024-01-16.csv
```

---

## Quick Start

### 1. Install dependencies
```bash
python setup.py
# or manually:
pip install flask opencv-contrib-python numpy --break-system-packages
```

### 2. Start the server
```bash
python app.py
```

### 3. Open the dashboard
```
http://localhost:5050
```

---

## How to Use

### Step 1 — Add Employees
- Go to **Employees** page
- Click **+ Add Employee**
- Enter ID (e.g. `EMP001`), Name, Department

### Step 2 — Train Faces
- Go to **Train Model** page
- Select an employee → click **Start Capture**
- Look at the camera — it captures 50 face samples
- Repeat for each employee
- Click **Train Model** after all captures

### Step 3 — Mark Attendance
- Go to **Live Recognition** page
- Click **▶ Start**
- Employees walk in front of camera
  - First detection → **CHECK IN** recorded
  - Second detection (later) → **CHECK OUT** recorded
- All data saved to CSV automatically

### Step 4 — View Records
- **Dashboard** — Today's summary
- **Attendance Log** — Browse by date, export CSV
- **Analytics** — 30-day trends per employee

---

## CSV Record Format

Each daily file: `records/attendance_YYYY-MM-DD.csv`

| Column | Description |
|--------|-------------|
| Date | YYYY-MM-DD |
| Employee_ID | Unique ID |
| Name | Full name |
| Department | Department name |
| Check_In | HH:MM:SS |
| Check_Out | HH:MM:SS |
| Status | PRESENT / LATE / EARLY_LEAVE / ABSENT |
| Working_Hours | Decimal hours |
| Check_In_Status | ON_TIME / LATE |
| Check_Out_Status | ON_TIME / EARLY_LEAVE / NOT_OUT |
| Notes | Face Detected / Manual |

---

## Status Logic

| Status | Condition |
|--------|-----------|
| `PRESENT` | Checked in on time |
| `LATE` | Checked in > threshold after work_start |
| `EARLY_LEAVE` | Checked out > threshold before work_end |
| `LATE+EARLY` | Both late and early leave |
| `ABSENT` | No check-in recorded |

Default thresholds (configurable in Settings):
- **Late**: 15 minutes after 09:00
- **Early Leave**: 30 minutes before 17:00

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/employees` | List all employees |
| POST | `/api/employees` | Add employee |
| DELETE | `/api/employees/<id>` | Remove employee |
| POST | `/api/train/start` | Start face capture |
| GET | `/api/train/status` | Capture progress |
| POST | `/api/train/model` | Train LBPH model |
| POST | `/api/recognize/start` | Start live recognition |
| GET | `/api/recognize/status` | Recognition feed + detection |
| POST | `/api/recognize/stop` | Stop camera |
| POST | `/api/attendance/checkin` | Manual check-in |
| POST | `/api/attendance/checkout` | Manual check-out |
| GET | `/api/attendance/today` | Today's records |
| GET | `/api/attendance/date/<date>` | Records by date |
| GET | `/api/attendance/dates` | Available dates |
| GET | `/api/analytics?days=30` | Analytics summary |
| GET | `/api/model/status` | Model info |
| GET/POST | `/api/settings` | Get/save settings |

---

## Tech Stack

- **Python 3.8+**
- **OpenCV** — Face detection (Haar Cascade) + Recognition (LBPH)
- **Flask** — Lightweight web server
- **HTML/CSS/JS** — Single-file dashboard, no frameworks
- **CSV** — Simple, portable attendance records

---

## Troubleshooting

**Camera not opening?**
- Ensure no other app is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `face_engine.py`

**Low recognition accuracy?**
- Capture more samples (increase to 80–100)
- Ensure good lighting during capture
- Capture at different angles and expressions
- Retrain the model after adding new employees

**`cv2.face` not found?**
```bash
pip install opencv-contrib-python --break-system-packages
```

**Port already in use?**
Change port in `app.py`: `app.run(port=5051)`
