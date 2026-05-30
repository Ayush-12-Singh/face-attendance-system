# ================= IMPORTS =================
from flask import Flask, render_template, Response, request, redirect, send_file, session
import cv2
import os
import re
import json
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime
import time
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib import error as urlerror

# ================= APP =================
app = Flask(__name__)
app.secret_key = 'secret123'

# ================= GLOBALS =================
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = None
label_map = {}

last_marked_time = {}  # 12-hour control
camera_active = False  # 🔥 camera lock

# Active "class session" (teacher selects subject+section and then AI/manual marks attendance into it)
active_session_id = None
active_session_subject = None
active_session_section = None

# LBPH: lower confidence => better match. Tune this to reduce false positives.
# Lower = stricter (fewer false positives, may miss some). 55-65 is often better.
CONFIDENCE_THRESHOLD = 58
# Require the same identity to be detected consistently for a few frames
# before we mark attendance (improves accuracy a lot in noisy conditions).
STABLE_FRAMES = 4

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Attendance table (v3): added `subject` + `session_id` for per-class-session totals
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, mode TEXT, section TEXT,
                  subject TEXT, session_id INTEGER)''')

    # If the DB already has an older attendance table, add missing columns.
    c.execute("PRAGMA table_info(attendance)")
    cols = [row[1] for row in c.fetchall()]
    if 'section' not in cols:
        c.execute("ALTER TABLE attendance ADD COLUMN section TEXT")
    if 'subject' not in cols:
        c.execute("ALTER TABLE attendance ADD COLUMN subject TEXT")
    if 'session_id' not in cols:
        c.execute("ALTER TABLE attendance ADD COLUMN session_id INTEGER")

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS sections
                 (section_name TEXT PRIMARY KEY)''')

    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (name TEXT PRIMARY KEY, section_name TEXT NOT NULL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS subjects
                 (subject_name TEXT PRIMARY KEY)''')

    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  subject_name TEXT NOT NULL,
                  section TEXT NOT NULL,
                  date TEXT NOT NULL,
                  created_at TEXT NOT NULL)''')

    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", ('admin', '1234'))
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", ('ayush', '1234'))
    conn.commit()
    conn.close()

init_db()

# ================= SECTION / STUDENT HELPERS =================
def normalize_name(name: str | None) -> str:
    return (name or "").strip().upper()


def normalize_section(section: str | None) -> str:
    s = (section or "").strip().upper()
    return s if s else "GENERAL"


def normalize_subject(subject: str | None) -> str:
    s = (subject or "").strip().upper()
    return s if s else "GENERAL"


def upsert_student(name: str, section: str | None):
    name_n = normalize_name(name)
    section_n = normalize_section(section)
    if not name_n:
        return

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO sections(section_name) VALUES (?)", (section_n,))
    c.execute(
        "INSERT OR REPLACE INTO students(name, section_name) VALUES (?, ?)",
        (name_n, section_n),
    )
    conn.commit()
    conn.close()


def get_student_section(name: str) -> str:
    name_n = normalize_name(name)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT section_name FROM students WHERE name=?", (name_n,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else "GENERAL"

# ================= TRAIN MODEL =================
def person_name_from_filename(filename: str) -> str:
    """
    Expected format: <person_name>_<index>.jpg
    Person name itself may contain underscores, so we parse from the last '_' before a numeric index.
    """
    stem = os.path.splitext(filename)[0]
    m = re.match(r"^(.*)_(\d+)$", stem)
    return m.group(1) if m else stem


def train_model():
    global recognizer, label_map

    faces = []
    labels = []
    label_map = {}

    name_to_label_id = {}
    label_id = 0

    for file in os.listdir('images'):
        path = os.path.join('images', file)

        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        name = person_name_from_filename(file)

        if name not in name_to_label_id:
            name_to_label_id[name] = label_id
            label_map[label_id] = name
            label_id += 1

        current_id = name_to_label_id[name]

        # Most images in ./images are already cropped faces (capture() crops before saving),
        # but sometimes you may upload a non-cropped photo. We try cascade detection and
        # fall back to the full image if detection fails.
        detected = face_cascade.detectMultiScale(gray, 1.2, 6)
        if len(detected) > 0:
            # Use the largest detected face region.
            x, y, w, h = max(detected, key=lambda b: b[2] * b[3])
            face = gray[y:y + h, x:x + w]
        else:
            face = gray

        # Preprocess for more stable LBPH training/prediction.
        face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_AREA)
        face = cv2.equalizeHist(face)
        faces.append(face)
        labels.append(current_id)

    if len(faces) > 0:
        # radius=2, neighbors=8: slightly better accuracy for small datasets
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8)
        recognizer.train(faces, np.array(labels))
        print("✅ Model trained")

train_model()

# ================= MARK ATTENDANCE =================
def markAttendance(name, mode="AI", section: str | None = None):
    # Backward compatible wrapper: if no subject/session provided, mark into legacy style.
    return markAttendanceV3(name=name, mode=mode, session_id=None, subject=None, section=section)


def create_session(subject: str, section: str) -> int:
    """
    Creates a new class session for (subject + section) and sets it as the active session.
    Total classes = count of sessions for the selected (subject + section).
    """
    global active_session_id, active_session_subject, active_session_section

    subject_n = normalize_subject(subject)
    section_n = normalize_section(section)

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("INSERT OR IGNORE INTO subjects(subject_name) VALUES (?)", (subject_n,))
    now = datetime.now()
    date = now.strftime('%d-%m-%Y')
    created_at = now.strftime('%H:%M:%S')

    c.execute(
        "INSERT INTO sessions(subject_name, section, date, created_at) VALUES (?, ?, ?, ?)",
        (subject_n, section_n, date, created_at),
    )
    session_id = c.lastrowid
    conn.commit()
    conn.close()

    active_session_id = session_id
    active_session_subject = subject_n
    active_session_section = section_n
    return session_id


def markAttendanceV3(name: str, mode: str = "AI", session_id: int | None = None, subject: str | None = None, section: str | None = None) -> bool:
    """
    Marks attendance into the given session (deduped per (session_id, name)).
    Returns True if a new row was inserted, False if it was a duplicate (or invalid session).
    """
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    now = datetime.now()
    date = now.strftime('%d-%m-%Y')
    time_now = now.strftime('%H:%M:%S')

    name_n = normalize_name(name)

    if session_id is not None:
        c.execute("SELECT subject_name, section FROM sessions WHERE session_id=?", (session_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return False

        subject_n = (row[0] or "").strip().upper()
        section_n = (row[1] or "").strip().upper()

        # Deduplicate: one attendance record per student per class session.
        c.execute(
            "SELECT 1 FROM attendance WHERE name=? AND session_id=?",
            (name_n, session_id),
        )
        if c.fetchone():
            conn.close()
            return False

        c.execute(
            "INSERT INTO attendance (name, date, time, mode, section, subject, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name_n, date, time_now, mode, section_n, subject_n, session_id),
        )
        conn.commit()
        conn.close()
        return True

    # Legacy fallback (no session): keep old behavior, but also populate subject column.
    subject_n = normalize_subject(subject)
    section_n = get_student_section(name_n) if section is None else normalize_section(section)

    # Prevent spam for legacy mode.
    if name_n in last_marked_time and (now - last_marked_time[name_n]).total_seconds() <= 43200:
        conn.close()
        return False

    c.execute(
        "INSERT INTO attendance (name, date, time, mode, section, subject, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name_n, date, time_now, mode, section_n, subject_n, None),
    )
    conn.commit()
    conn.close()

    last_marked_time[name_n] = now
    return True

# ================= VIDEO STREAM =================
# ================= VIDEO STREAM =================
def generate_frames():
    global active_session_id, last_marked_time
    stable_name = None
    stable_count = 0

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)
        # Process biggest face first.
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        if len(faces) == 0:
            stable_name = None
            stable_count = 0

        # 1. Process and draw ONLY if a face is found
        for (face_index, (x, y, w, h)) in enumerate(faces):
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_AREA)
            face = cv2.equalizeHist(face)
            
            display_text = "UNKNOWN"

            if recognizer is not None:
                # Wrap in try-except in case model isn't fully trained yet
                try:
                    label, confidence = recognizer.predict(face)

                    if confidence < CONFIDENCE_THRESHOLD:
                        name = label_map.get(label, "UNKNOWN").upper()
                        now = datetime.now()

                        # Only do "stability" + attendance marking on the biggest face.
                        if face_index == 0:
                            if name == stable_name:
                                stable_count += 1
                            else:
                                stable_name = name
                                stable_count = 1

                            can_mark = name != "UNKNOWN" and stable_count >= STABLE_FRAMES

                            if can_mark:
                                inserted = markAttendanceV3(
                                    name=name,
                                    mode="AI",
                                    session_id=active_session_id,
                                )
                                display_text = "MARKED" if inserted else name
                            else:
                                display_text = name
                        else:
                            display_text = name
                    else:
                        # Candidate isn't confident enough; reset stability for biggest face.
                        if face_index == 0:
                            stable_name = None
                            stable_count = 0
                except:
                    if face_index == 0:
                        stable_name = None
                        stable_count = 0
                    pass
            else:
                if face_index == 0:
                    stable_name = None
                    stable_count = 0

            color = (0, 255, 0) if display_text == "MARKED" else (255, 255, 255)

            # Moved inside the loop! 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 2. Encode and yield the frame REGARDLESS of faces being detected
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/status', methods=['GET'])
def status():
    return {
        "trained": recognizer is not None,
        "people_count": len(label_map),
        "threshold": CONFIDENCE_THRESHOLD,
        "stable_frames": STABLE_FRAMES,
        "active_session": {
            "session_id": active_session_id,
            "subject": active_session_subject,
            "section": active_session_section,
        } if active_session_id is not None else None,
    }

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global CONFIDENCE_THRESHOLD
    try:
        value = int(request.form.get('value', '').strip())
    except Exception:
        return {"status": "fail", "reason": "Invalid value"}

    # Keep within a reasonable range for LBPH confidence.
    if value < 10:
        value = 10
    if value > 200:
        value = 200

    CONFIDENCE_THRESHOLD = value
    return {"status": "success", "threshold": CONFIDENCE_THRESHOLD}

# ================= ROUTES =================

@app.route('/sections', methods=['GET'])
def sections_get():
    include_attendance = request.args.get('include_attendance', '').lower() == 'true'
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT section_name FROM sections ORDER BY section_name ASC")
    rows = c.fetchall()
    sections = {r[0] for r in rows}
    if include_attendance:
        c.execute("SELECT DISTINCT section FROM attendance WHERE section IS NOT NULL AND section != ''")
        for r in c.fetchall():
            sections.add((r[0] or "").strip().upper())
        sections.discard("")
    conn.close()
    return {"data": sorted(sections)}


@app.route('/sections', methods=['POST'])
def sections_post():
    section = request.form.get('section', None)
    section_n = normalize_section(section)
    if not section_n:
        return {"status": "fail", "reason": "Missing section"}

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO sections(section_name) VALUES (?)", (section_n,))
    conn.commit()
    conn.close()
    return {"status": "success", "section": section_n}


@app.route('/students', methods=['GET'])
def students_get():
    section = request.args.get('section', None)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    if section:
        section_n = normalize_section(section)
        c.execute(
            "SELECT name, section_name FROM students WHERE section_name=? ORDER BY name ASC",
            (section_n,),
        )
    else:
        c.execute("SELECT name, section_name FROM students ORDER BY section_name ASC, name ASC")
    rows = c.fetchall()
    conn.close()
    return {"data": rows}


@app.route('/students', methods=['POST'])
def students_post():
    name = request.form.get('name', None)
    section = request.form.get('section', None)
    if not name:
        return {"status": "fail", "reason": "Missing name"}
    upsert_student(name, section)
    return {"status": "success"}


@app.route('/students/delete', methods=['POST'])
def students_delete():
    name = request.form.get('name', None)
    if not name:
        return {"status": "fail", "reason": "Missing name"}
    name_n = normalize_name(name)
    if not name_n:
        return {"status": "fail", "reason": "Invalid name"}

    # Delete from students table
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE name=?", (name_n,))
    conn.commit()
    conn.close()

    # Delete face images for this person (match by person_name_from_filename)
    images_dir = 'images'
    if os.path.isdir(images_dir):
        for f in os.listdir(images_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                parsed = normalize_name(person_name_from_filename(f))
                if parsed == name_n:
                    try:
                        os.remove(os.path.join(images_dir, f))
                    except OSError:
                        pass

    train_model()
    return {"status": "success"}


@app.route('/sections/delete', methods=['POST'])
def sections_delete():
    section = request.form.get('section', None)
    if not section:
        return {"status": "fail", "reason": "Missing section"}
    section_n = normalize_section(section)
    if section_n == "GENERAL":
        return {"status": "fail", "reason": "Cannot delete GENERAL section"}

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Move students in this section to GENERAL
    c.execute(
        "UPDATE students SET section_name='GENERAL' WHERE section_name=?",
        (section_n,),
    )
    c.execute("DELETE FROM sections WHERE section_name=?", (section_n,))
    conn.commit()
    conn.close()
    return {"status": "success"}


def verify_google_id_token(id_token: str, expected_audience: str | None = None) -> dict:
    """
    Verify Google `id_token` using Google's public tokeninfo endpoint.
    No extra dependencies required (uses stdlib urllib + json).
    """
    if not id_token:
        raise ValueError("Missing id_token")

    url = "https://oauth2.googleapis.com/tokeninfo?id_token=" + urlparse.quote(id_token)
    try:
        with urlrequest.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            data = json.loads(body)
        except Exception:
            data = {"error": str(e)}
        raise ValueError(data.get("error_description") or data.get("error") or "Invalid Google token")

    if expected_audience:
        aud = data.get("aud")
        if aud and aud != expected_audience:
            raise ValueError("Google token audience mismatch")

    return data


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            # Generate a custom initial avatar for the admin
            session['avatar'] = f"https://ui-avatars.com/api/?name={username}&background=00d1ff&color=fff&rounded=true"
            return redirect('/')
        else:
            return render_template(
                'login.html',
                error="Invalid username or password",
                google_client_id=os.environ.get('GOOGLE_CLIENT_ID', '')
            )

    return render_template('login.html', google_client_id=os.environ.get('GOOGLE_CLIENT_ID', ''))


@app.route('/google_login', methods=['POST'])
def google_login():
    id_token = request.form.get('id_token', '')
    expected_aud = os.environ.get('GOOGLE_CLIENT_ID', None)

    try:
        token_data = verify_google_id_token(id_token, expected_aud)
        email = (token_data.get("email") or token_data.get("sub") or "").strip()
        if not email:
            return {"status": "fail", "reason": "Google token missing email"}

        session['user'] = email
        return {"status": "success"}
    except Exception as e:
        return {"status": "fail", "reason": str(e)}

@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Return rowid so the frontend can delete the exact row reliably.
    c.execute("SELECT rowid, name, date, time, mode, section, subject, session_id FROM attendance ORDER BY rowid DESC")
    data = c.fetchall()
    conn.close()
    return {"data": data}

@app.route('/manual_mark', methods=['POST'])
def manual_mark():
    name = request.form['name']
    password = request.form['password']
    section = request.form.get('section', None)
    subject = request.form.get('subject', None)

    if password == '1234':
        if section is not None:
            upsert_student(name, section)

        subject_n = normalize_subject(subject)
        section_n = normalize_section(section)

        # Use active session only if it matches subject+section; otherwise create a new session.
        sid = active_session_id
        if sid is not None:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("SELECT subject_name, section FROM sessions WHERE session_id=?", (sid,))
            row = c.fetchone()
            conn.close()
            if row:
                active_subject = (row[0] or "").strip().upper()
                active_section = (row[1] or "").strip().upper()
                if active_subject != subject_n or active_section != section_n:
                    sid = create_session(subject_n, section_n)
        else:
            sid = create_session(subject_n, section_n)

        inserted = markAttendanceV3(name, mode="MANUAL", session_id=sid)
        return {"status": "success", "marked": inserted}
    return {"status": "fail"}


# ================= SUBJECTS / SESSIONS =================

@app.route('/subjects', methods=['GET'])
def subjects_get():
    include_attendance = request.args.get('include_attendance', '').lower() == 'true'
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("SELECT subject_name FROM subjects ORDER BY subject_name ASC")
    subjects = {r[0] for r in c.fetchall()}

    if include_attendance:
        c.execute("SELECT DISTINCT subject FROM attendance WHERE subject IS NOT NULL AND subject != ''")
        for r in c.fetchall():
            subjects.add((r[0] or "").strip().upper())

    conn.close()
    subjects = [s for s in subjects if s]
    if not subjects:
        subjects = ["GENERAL"]
    return {"data": sorted(set(s.upper() for s in subjects))}


@app.route('/subjects', methods=['POST'])
def subjects_post():
    subject = request.form.get('subject', None)
    subject_n = normalize_subject(subject)
    if not subject_n:
        return {"status": "fail", "reason": "Missing subject"}

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO subjects(subject_name) VALUES (?)", (subject_n,))
    conn.commit()
    conn.close()
    return {"status": "success", "subject": subject_n}


@app.route('/subjects/delete', methods=['POST'])
def subjects_delete():
    subject = request.form.get('subject', None)
    subject_n = normalize_subject(subject)
    if not subject_n:
        return {"status": "fail", "reason": "Missing subject"}
    if subject_n == "GENERAL":
        return {"status": "fail", "reason": "Cannot delete GENERAL"}

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Delete sessions + attendance tied to this subject.
    c.execute("SELECT session_id FROM sessions WHERE subject_name=?", (subject_n,))
    session_ids = [r[0] for r in c.fetchall()]

    if session_ids:
        placeholders = ",".join(["?"] * len(session_ids))
        c.execute(f"DELETE FROM attendance WHERE session_id IN ({placeholders})", tuple(session_ids))

    c.execute("DELETE FROM sessions WHERE subject_name=?", (subject_n,))
    c.execute("DELETE FROM subjects WHERE subject_name=?", (subject_n,))
    conn.commit()
    conn.close()

    global active_session_id, active_session_subject, active_session_section
    if active_session_subject == subject_n:
        active_session_id = None
        active_session_subject = None
        active_session_section = None

    return {"status": "success"}


@app.route('/session/start', methods=['POST'])
def session_start():
    subject = request.form.get('subject', None)
    section = request.form.get('section', None)

    subject_n = normalize_subject(subject)
    section_n = normalize_section(section)

    sid = create_session(subject_n, section_n)
    return {"status": "success", "session_id": sid, "subject": subject_n, "section": section_n}


@app.route('/low-attendance', methods=['POST'])
def low_attendance():
    subject = request.form.get('subject', None)
    section = request.form.get('section', None)
    mode = request.form.get('mode', 'percentage')  # percentage | attended
    value_raw = request.form.get('value', '0')

    try:
        value = float(value_raw)
    except Exception:
        value = 0.0

    subject_n = normalize_subject(subject)
    section_n = normalize_section(section)

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute(
        "SELECT session_id FROM sessions WHERE subject_name=? AND section=? ORDER BY session_id ASC",
        (subject_n, section_n),
    )
    session_rows = c.fetchall()
    session_ids = [r[0] for r in session_rows]
    total_classes = len(session_ids)

    if total_classes == 0:
        conn.close()
        return {"status": "success", "subject": subject_n, "section": section_n, "total_classes": 0, "students": []}

    placeholders = ",".join(["?"] * len(session_ids))
    c.execute(
        f"SELECT name, COUNT(DISTINCT session_id) AS attended_classes FROM attendance WHERE session_id IN ({placeholders}) GROUP BY name",
        tuple(session_ids),
    )
    attended_map = {r[0]: int(r[1]) for r in c.fetchall()}

    c.execute(
        "SELECT name FROM students WHERE section_name=? ORDER BY name ASC",
        (section_n,),
    )
    student_names = [r[0] for r in c.fetchall()]
    conn.close()

    results = []
    for student in student_names:
        attended = attended_map.get(student, 0)
        percentage = (attended / total_classes) * 100.0 if total_classes > 0 else 0.0

        show = False
        if mode == "attended":
            show = attended < int(value)
        else:
            show = percentage < float(value)

        if show:
            results.append({
                "name": student,
                "total_classes": total_classes,
                "attended_classes": attended,
                "percentage": round(percentage, 1),
            })

    # Sort by lowest attendance first
    results.sort(key=lambda r: (r["percentage"], r["attended_classes"]))
    return {"status": "success", "subject": subject_n, "section": section_n, "total_classes": total_classes, "students": results}

@app.route('/delete', methods=['POST'])
def delete():
    rowid = request.form.get('rowid', None)
    if rowid is not None and str(rowid).strip() != "":
        rid = int(rowid)
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("DELETE FROM attendance WHERE rowid=?", (rid,))
        conn.commit()
        conn.close()
        return "OK"

    index = int(request.form.get('index', '0'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("SELECT rowid FROM attendance ORDER BY rowid DESC")
    rows = c.fetchall()

    if index < len(rows):
        c.execute("DELETE FROM attendance WHERE rowid=?", (rows[index][0],))

    conn.commit()
    conn.close()
    return "OK"

@app.route('/download')
def download():
    df = pd.read_sql_query("SELECT * FROM attendance", sqlite3.connect('database.db'))
    file_path = 'attendance.xlsx'
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    section = request.form.get('section', None)
    file = request.files['image']

    filename = f"{name}_{len(os.listdir('images'))}.jpg"
    path = os.path.join('images', filename)

    file.save(path)

    if section is not None:
        upsert_student(name, section)

    train_model()
    return redirect('/')

# ================= CAPTURE =================
@app.route('/capture', methods=['POST'])
def capture():
    import time

    print("📸 Capture started")

    name = request.form.get('name')
    section = request.form.get('section', None)
    if not name:
        return {"status": "fail"}

    cap = cv2.VideoCapture(0)

    existing = [f for f in os.listdir('images') if f.startswith(name)]
    start_index = len(existing)

    count = 0

    while count < 10:
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200,200))

            filename = f"{name}_{start_index + count + 1}.jpg"
            cv2.imwrite(os.path.join('images', filename), face)

            print("Saved:", filename)
            count += 1
            break

        time.sleep(0.3)

    cap.release()

    if section is not None:
        upsert_student(name, section)

    train_model()

    print("🎯 Capture done")

    return {"status":"success"}

@app.route('/logout')
def logout():
    session.pop('user',None)
    return redirect('/login')

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)