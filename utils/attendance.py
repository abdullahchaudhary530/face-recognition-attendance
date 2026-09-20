"""
Attendance Manager
- Check-in / Check-out tracking
- Late / Early detection
- CSV record keeping
- Analytics
"""
import csv
import json
import os
from datetime import datetime, timedelta, date
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RECORDS_DIR = os.path.join(BASE_DIR, "records")
DATA_DIR = os.path.join(BASE_DIR, "data")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "active_sessions.json")

os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─── Default Work Schedule ───────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "work_start": "09:00",       # Expected check-in time
    "work_end": "17:00",         # Expected check-out time
    "late_threshold_min": 15,    # Minutes after work_start = LATE
    "early_threshold_min": 30,   # Minutes before work_end = EARLY LEAVE
    "organization": "My Organization",
    "timezone": "PKT"            # Pakistan Standard Time
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
        # Merge with defaults for any missing keys
        for k, v in DEFAULT_SETTINGS.items():
            s.setdefault(k, v)
        return s
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_csv_path(date_str=None):
    if date_str is None:
        date_str = date.today().isoformat()
    return os.path.join(RECORDS_DIR, f"attendance_{date_str}.csv")


CSV_HEADERS = [
    "Date", "Employee_ID", "Name", "Department",
    "Check_In", "Check_Out", "Status",
    "Working_Hours", "Check_In_Status", "Check_Out_Status", "Notes"
]


def _read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _get_status_flags(check_in_str, check_out_str, settings):
    """Returns (check_in_status, check_out_status, overall_status)"""
    work_start = datetime.strptime(settings["work_start"], "%H:%M").time()
    work_end = datetime.strptime(settings["work_end"], "%H:%M").time()
    late_delta = timedelta(minutes=settings["late_threshold_min"])
    early_delta = timedelta(minutes=settings["early_threshold_min"])

    ci_status = "ON_TIME"
    co_status = "ON_TIME"
    overall = "PRESENT"

    if check_in_str:
        ci_time = datetime.strptime(check_in_str, "%H:%M:%S").time()
        late_cutoff = (datetime.combine(date.today(), work_start) + late_delta).time()
        if ci_time > late_cutoff:
            ci_status = "LATE"
            overall = "LATE"

    if check_out_str:
        co_time = datetime.strptime(check_out_str, "%H:%M:%S").time()
        early_cutoff = (datetime.combine(date.today(), work_end) - early_delta).time()
        if co_time < early_cutoff:
            co_status = "EARLY_LEAVE"
            if overall == "LATE":
                overall = "LATE+EARLY"
            else:
                overall = "EARLY_LEAVE"
    elif check_in_str:
        co_status = "NOT_OUT"
        overall = overall if overall != "PRESENT" else "PRESENT"

    return ci_status, co_status, overall


def _calc_working_hours(check_in_str, check_out_str):
    if not check_in_str or not check_out_str:
        return ""
    try:
        ci = datetime.strptime(check_in_str, "%H:%M:%S")
        co = datetime.strptime(check_out_str, "%H:%M:%S")
        delta = co - ci
        if delta.total_seconds() < 0:
            return ""
        hours = delta.total_seconds() / 3600
        return f"{hours:.2f}"
    except Exception:
        return ""


def load_active_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE) as f:
            return json.load(f)
    return {}


def save_active_sessions(sessions):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def check_in(employee_id, employee_name, department="General", notes="Face Detected"):
    """Record check-in. Returns (success, message, record)."""
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")
    csv_path = get_csv_path(today)
    settings = load_settings()

    rows = _read_csv(csv_path)

    # Check already checked in today
    for row in rows:
        if row["Employee_ID"] == str(employee_id) and row["Date"] == today:
            if row["Check_In"]:
                return False, f"{employee_name} already checked in at {row['Check_In']}", row

    ci_status, _, overall = _get_status_flags(now_time, None, settings)

    new_row = {
        "Date": today,
        "Employee_ID": str(employee_id),
        "Name": employee_name,
        "Department": department,
        "Check_In": now_time,
        "Check_Out": "",
        "Status": overall,
        "Working_Hours": "",
        "Check_In_Status": ci_status,
        "Check_Out_Status": "",
        "Notes": notes
    }
    rows.append(new_row)
    _write_csv(csv_path, rows)

    # Track active session
    sessions = load_active_sessions()
    sessions[str(employee_id)] = {"check_in": now_time, "date": today, "name": employee_name}
    save_active_sessions(sessions)

    return True, f"✅ Check-in recorded: {employee_name} at {now_time} [{ci_status}]", new_row


def check_out(employee_id, employee_name, notes="Face Detected"):
    """Record check-out. Returns (success, message, record)."""
    today = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M:%S")
    csv_path = get_csv_path(today)
    settings = load_settings()

    rows = _read_csv(csv_path)
    updated = False
    result_row = None

    for row in rows:
        if row["Employee_ID"] == str(employee_id) and row["Date"] == today:
            if row["Check_Out"]:
                return False, f"{employee_name} already checked out at {row['Check_Out']}", row
            if not row["Check_In"]:
                return False, f"{employee_name} has no check-in record today", row

            _, co_status, overall = _get_status_flags(row["Check_In"], now_time, settings)
            hours = _calc_working_hours(row["Check_In"], now_time)

            row["Check_Out"] = now_time
            row["Check_Out_Status"] = co_status
            row["Working_Hours"] = hours
            row["Status"] = overall
            updated = True
            result_row = row
            break

    if not updated:
        return False, f"No check-in found for {employee_name} today", None

    _write_csv(csv_path, rows)

    # Remove from active sessions
    sessions = load_active_sessions()
    sessions.pop(str(employee_id), None)
    save_active_sessions(sessions)

    return True, f"✅ Check-out: {employee_name} at {now_time} | Hours: {hours} [{co_status}]", result_row


def get_today_records():
    today = date.today().isoformat()
    return _read_csv(get_csv_path(today))


def get_records_for_date(date_str):
    return _read_csv(get_csv_path(date_str))


def get_all_records(days=30):
    """Get records for the last N days."""
    all_rows = []
    for i in range(days):
        d = (date.today() - timedelta(days=i)).isoformat()
        all_rows.extend(_read_csv(get_csv_path(d)))
    return all_rows


def get_available_dates():
    dates = []
    for fname in sorted(os.listdir(RECORDS_DIR), reverse=True):
        if fname.startswith("attendance_") and fname.endswith(".csv"):
            d = fname.replace("attendance_", "").replace(".csv", "")
            dates.append(d)
    return dates


def get_analytics(days=30):
    """Compute attendance analytics."""
    records = get_all_records(days)
    
    total = len(records)
    present = sum(1 for r in records if r.get("Check_In"))
    late = sum(1 for r in records if r.get("Check_In_Status") == "LATE")
    early_leave = sum(1 for r in records if r.get("Check_Out_Status") == "EARLY_LEAVE")
    absent = sum(1 for r in records if not r.get("Check_In"))

    # Per-employee summary
    emp_summary = defaultdict(lambda: {"present": 0, "late": 0, "early": 0, "hours": 0.0, "name": ""})
    for r in records:
        eid = r.get("Employee_ID", "")
        emp_summary[eid]["name"] = r.get("Name", eid)
        if r.get("Check_In"):
            emp_summary[eid]["present"] += 1
        if r.get("Check_In_Status") == "LATE":
            emp_summary[eid]["late"] += 1
        if r.get("Check_Out_Status") == "EARLY_LEAVE":
            emp_summary[eid]["early"] += 1
        try:
            emp_summary[eid]["hours"] += float(r.get("Working_Hours") or 0)
        except Exception:
            pass

    # Daily trend (last 7 days)
    daily_trend = {}
    for i in range(7):
        d = (date.today() - timedelta(days=i)).isoformat()
        day_rows = _read_csv(get_csv_path(d))
        daily_trend[d] = {
            "present": sum(1 for r in day_rows if r.get("Check_In")),
            "late": sum(1 for r in day_rows if r.get("Check_In_Status") == "LATE"),
            "total": len(day_rows)
        }

    return {
        "total_records": total,
        "present": present,
        "late": late,
        "early_leave": early_leave,
        "absent": absent,
        "employee_summary": dict(emp_summary),
        "daily_trend": daily_trend
    }


def get_employee_history(employee_id, days=30):
    """Get attendance history for a specific employee."""
    records = get_all_records(days)
    return [r for r in records if r.get("Employee_ID") == str(employee_id)]


def mark_absent_for_date(date_str, employee_ids_names):
    """Mark employees as absent for a specific date."""
    csv_path = get_csv_path(date_str)
    existing = _read_csv(csv_path)
    existing_ids = {r["Employee_ID"] for r in existing}
    
    for emp_id, name, dept in employee_ids_names:
        if str(emp_id) not in existing_ids:
            existing.append({
                "Date": date_str,
                "Employee_ID": str(emp_id),
                "Name": name,
                "Department": dept,
                "Check_In": "",
                "Check_Out": "",
                "Status": "ABSENT",
                "Working_Hours": "",
                "Check_In_Status": "ABSENT",
                "Check_Out_Status": "ABSENT",
                "Notes": "Auto-marked absent"
            })
    _write_csv(csv_path, existing)
