from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    send_file,
    Response,
)
import os
import cv2
import json
import qrcode
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet
import face_recognition


import numpy as np
from datetime import datetime, timedelta
from pyzbar.pyzbar import decode
import mediapipe as mp
import threading
import queue
import time
import pickle
import random
from email.message import EmailMessage
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_db_connection
from face_verifier import face_recognition_pipeline
import base64
import io

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad



app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Set config for consistency
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

PROFILE_PIC_URL_PREFIX = 'profile_pics'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'docx'}
QR_FOLDER = os.path.join(BASE_DIR, "static", "qr_codes")
os.makedirs(QR_FOLDER, exist_ok=True)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(USER_DATA_DIR, exist_ok=True)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "cyloixy2610@gmail.com"
SENDER_PASSWORD = "gprb gqku lqdf iemg"   # Gmail App Password

def generate_otp():
    return str(random.randint(100000, 999999))


def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, name, email,
               COALESCE(profile_pic_path, 'profile_pics/default.png')
        FROM Users
        WHERE user_id = ?
    """, user_id)

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "user_id": row[0],
        "name": row[1],
        "email": row[2],
        "profile_pic_path": row[3],
    }


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap


def allowed_profile_picture(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def allowed_document(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_DOCUMENT_EXTENSIONS
    )


def save_profile_picture(file, email):
    if not file or file.filename == '':
        return None
    if not allowed_profile_picture(file.filename):
        return None

    filename = secure_filename(
        f"{email}_{int(time.time())}_{file.filename.rsplit('.', 1)[0]}"
    )
    extension = file.filename.rsplit('.', 1)[1].lower()
    stored_name = f"{filename}.{extension}"
    return stored_name


@app.route('/')
def home():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, name, email, COALESCE(profile_pic_path, 'default.png')
        FROM Users
        WHERE user_id = ?
    """, user_id)

    row = cursor.fetchone()
    conn.close()

    if not row:
        return redirect(url_for('logout'))

    user = {
        'user_id': row[0],
        'name': row[1],
        'email': row[2],
        'profile_pic_path': row[3] if row[3] else 'default.png'
    }

    return render_template('index.html', user=user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Normalize email (lowercase and strip) to match registration
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')

    if not email or not password:
        return render_template('login.html', error='Email and password are required')

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT user_id, password_hash FROM Users WHERE email = ?",
            email
        )
        row = cursor.fetchone()
        
        if not row:
            print(f"Login attempt: User with email '{email}' not found in database")
            return render_template('login.html', error='Invalid email or password')
        
        user_id, password_hash = row[0], row[1]
        
        # Check if password hash is valid
        if not password_hash:
            print(f"Login attempt: User {user_id} has no password hash")
            return render_template('login.html', error='Account error. Please contact support.')
        
        # Verify password
        if check_password_hash(password_hash, password):
            session['user_id'] = user_id
            session['user_email'] = email
            print(f"Login successful: User {user_id} ({email})")
            return redirect(url_for('dashboard'))
        else:
            print(f"Login attempt: Invalid password for user {user_id} ({email})")
            return render_template('login.html', error='Invalid email or password')
            
    except Exception as e:
        print(f"Login error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('login.html', error='Login error. Please try again.')
    finally:
        conn.close()

@app.route('/profile_picture/<int:user_id>')
@login_required
def get_profile_picture(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT profile_pic_path
        FROM Users WHERE user_id = ?
    """, user_id)

    row = cursor.fetchone()
    conn.close()

    if not row or not row[0]:
        return '', 404

    return Response(row[0], mimetype=row[1])

@app.route('/register', methods=['GET', 'POST'])
def register():

    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    error = None

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not name or not email or not password:
            error = 'All fields are required'
        elif password != confirm_password:
            error = 'Passwords do not match'
        else:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if email exists
            cursor.execute("SELECT user_id FROM Users WHERE email = ?", email)
            if cursor.fetchone():
                error = "An account with this email already exists"
            else:
                # Hash password
                hashed = generate_password_hash(password)

                # Handle profile picture (store PATH only)
                pic = request.files.get("profile_picture")
                pic_path = None

                if pic and pic.filename:
                    filename = secure_filename(f"{email}_{int(time.time())}.jpg")
                    save_path = os.path.join("static/profile_pics", filename)
                    pic.save(save_path)
                    pic_path = f"profile_pics/{filename}"
                else:
                    pic_path = "profile_pics/default.png"

                # Insert user
                cursor.execute("""
                    INSERT INTO Users (name, email, password_hash, profile_pic_path)
                    VALUES (?, ?, ?, ?)
                """, name, email, hashed, pic_path)

                conn.commit()

                cursor.execute("SELECT user_id FROM Users WHERE email=?", email)
                user_id = cursor.fetchone()[0]

                session["user_id"] = user_id
                session["user_email"] = email

                conn.close()

                return redirect(url_for("dashboard"))

            conn.close()

    return render_template("register.html", error=error)


@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))
@app.route('/profile')
@login_required
def profile():
    user = get_current_user()

    profile_picture_url = None
    if user:
        profile_picture_url = url_for(
            'get_profile_picture',
            user_id=user['user_id']
        )

    return render_template(
        'profile.html',
        user=user,
        profile_picture_url=profile_picture_url
    )


@app.route('/update_profile_picture', methods=['POST'])
@login_required
def update_profile_picture():
    """Update user's profile picture in database"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User session expired'}), 401

        if 'profile_picture' not in request.files:
            return jsonify({'error': 'No profile picture provided'}), 400

        profile_picture_file = request.files['profile_picture']
        if profile_picture_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get user email for filename
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT email FROM Users WHERE user_id = ?", user_id)
            user_row = cursor.fetchone()
            if not user_row:
                conn.close()
                return jsonify({'error': 'User not found'}), 404
            
            user_email = user_row[0]

            # Save the new profile picture
            saved_pic = save_profile_picture(profile_picture_file, user_email)
            if not saved_pic:
                conn.close()
                return jsonify({'error': 'Failed to save profile picture'}), 500

            # Update database with new profile picture path
            cursor.execute("""
                UPDATE Users 
                SET profile_pic_path = ? 
                WHERE user_id = ?
            """, saved_pic, user_id)

            conn.commit()
            conn.close()

            return jsonify({
                'success': True,
                'message': 'Profile picture updated successfully',
                'profile_picture': saved_pic
            })
        except Exception as db_error:
            conn.rollback()
            conn.close()
            print(f"Database error updating profile picture: {str(db_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Database error: {str(db_error)}'}), 500
    except Exception as e:
        print(f"Error updating profile picture: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to update profile picture: {str(e)}'}), 500


@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT file_blob, file_name, file_mime
        FROM Files
        WHERE file_id = ?
    """, file_id)
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "File not found", 404

    return Response(
        row[0],
        mimetype=row[2],
        headers={
            "Content-Disposition": f"attachment; filename={row[1]}"
        }
    )


# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "cyloixy2610@gmail.com"  # Your Gmail address
SENDER_PASSWORD = "gprb gqku lqdf iemg"  # Your Gmail App Password

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Thread class for video capture
class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0, width=640, height=450, queue_size=2):
        super().__init__()
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.capture.read()
                if not ret:
                    self.stop()
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.015)  # Prevent busy waiting

    def read(self):
        return self.queue.get()

    def more(self):
        return not self.queue.empty()

    def stop(self):
        self.stopped = True
        self.capture.release()


def send_otp_email(to_email, otp):

    msg = MIMEMultipart()   
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] =  "OTP Verification Code"
    body = f"""
Your One-Time Password (OTP) is:

{otp}

This code is valid for 5 minutes.
"""
    msg.attach(MIMEText(body.strip(), 'plain'))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    user = get_current_user()
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # -----------------------------
        # 1. Receive file or file link
        # -----------------------------
        file = request.files.get('document')
        fileLink = request.form.get('fileLink', '').strip()

        if file and file.filename:
            original_name = secure_filename(file.filename)
            file_bytes = file.read()
            mime_type = file.mimetype
        elif fileLink:
            # Download into memory? (optional)
            return jsonify({"error": "URL uploads not implemented in encrypted mode"}), 400
        else:
            return jsonify({"error": "No file provided"}), 400

        # -----------------------------
        # 2. Encrypt file with AES-GCM
        # -----------------------------
        aes_key = get_random_bytes(32)
        iv = get_random_bytes(12)

        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=iv)
        encrypted_file, tag = cipher.encrypt_and_digest(file_bytes)

        # -----------------------------
        # 3. Save encrypted file to disk
        # -----------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stored_filename = f"{timestamp}_{original_name}"
        stored_path = os.path.join("static", "uploads", stored_filename)
        abs_path = os.path.join(BASE_DIR, stored_path)

        with open(abs_path, "wb") as f:
            f.write(encrypted_file)

        # -----------------------------
        # 4. Insert into Files table
        # -----------------------------
        expiration_hours = float(request.form.get("expiration_hours", 24))
        expiration_timestamp = datetime.now() + timedelta(hours=expiration_hours)

        cursor.execute("""
            INSERT INTO Files (uploaded_by, file_path, file_name, file_mime, expiration_timestamp)
            OUTPUT INSERTED.file_id
            VALUES (?, ?, ?, ?, ?)
        """, (
            user['user_id'],
            stored_path,
            original_name,
            mime_type,
            expiration_timestamp
        ))

        file_id = cursor.fetchone()[0]

        # -----------------------------
        # 5. Insert AES key into FileKey
        # -----------------------------
        cursor.execute("""
            INSERT INTO FileKey (file_id, aes_key, iv, tag)
            VALUES (?, ?, ?, ?)
        """, (file_id, aes_key, iv, tag))

    # -----------------------------
    # 6. Handle receiver photos + face encodings
    # -----------------------------
        receiver_index = 0
        receiver_results = []

        while f"photo_{receiver_index}" in request.files:
            photo_file = request.files[f"photo_{receiver_index}"]
            receiver_email = request.form.get(f"email_{receiver_index}").strip().lower()

            # -----------------------------
            # Save receiver photo
            # -----------------------------
            photo_filename = f"{receiver_email}_{timestamp}_{receiver_index}.jpg"
            photo_path_rel = f"static/receiver_faces/{photo_filename}"
            photo_path_abs = os.path.join(BASE_DIR, photo_path_rel)
            photo_file.save(photo_path_abs)

            # -----------------------------
            # Generate face encoding
            # -----------------------------
            img = face_recognition.load_image_file(photo_path_abs)
            enc = face_recognition.face_encodings(img)

            if not enc:
                conn.rollback()
                return jsonify({"error": f"No face detected for {receiver_email}"}), 400

            encoding_blob = pickle.dumps(enc[0])

            # Store face encoding in DB
            cursor.execute("""
                INSERT INTO ReceiverFace (file_id, receiver_email, photo_path, face_encoding)
                VALUES (?, ?, ?, ?)
            """, (file_id, receiver_email, photo_path_rel, encoding_blob))

            # -----------------------------
            # Generate QR payload
            # -----------------------------
            qr_payload = {
                "file_id": file_id,
                "receiver_email": receiver_email,
                "expiry": expiration_timestamp.isoformat()
            }

            qr_json = json.dumps(qr_payload)

            # -----------------------------
            # Generate QR PNG
            # -----------------------------
            qr_img = qrcode.make(qr_json)

            qr_filename = f"qr_{receiver_email}_{timestamp}_{receiver_index}.png"
            qr_disk_rel = f"static/qr_codes/{qr_filename}"
            qr_disk_abs = os.path.join(BASE_DIR, qr_disk_rel)

            qr_img.save(qr_disk_abs)

            # -----------------------------
            # Save QR metadata to DB
            # -----------------------------
            cursor.execute("""
                INSERT INTO QRCode (file_id, receiver_email, qr_image_path, qr_metadata)
                OUTPUT INSERTED.qr_id
                VALUES (?, ?, ?, ?)
            """, (
                file_id,
                receiver_email,
                qr_disk_rel,     # save RELATIVE path
                qr_json
            ))

            qr_id = cursor.fetchone()[0]

            receiver_results.append({
                "email": receiver_email,
                "qr_id": qr_id,
                "qr_path": "/" + qr_disk_rel.replace("\\", "/")   # for download
            })

            receiver_index += 1


        conn.commit()

        # Convert receiver_results → qr_codes format for frontend
        qr_codes = []
        for r in receiver_results:
            qr_codes.append({
                "email": r["email"],
                "qr_id": r["qr_id"],
                "qr_path": f"/qr/{r['qr_id']}"
            })

        return jsonify({
            "success": True,
            "qr_codes": qr_codes
        })

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()

    
@app.route('/qr/<int:qr_id>')
@login_required
def get_qr(qr_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT qr_image_path
        FROM QRCode
        WHERE qr_id = ?
    """, qr_id)

    row = cursor.fetchone()
    conn.close()

    if not row:
        return "QR not found", 404

    qr_path = row[0]
    abs_path = os.path.join(app.root_path, qr_path)

    return send_file(abs_path, mimetype="image/png")



@app.route('/scan', methods=['POST'])
def scan_qr():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        decoded = decode(img)

        if not decoded:
            return jsonify({"error": "Invalid QR code"}), 400

        qr_data = json.loads(decoded[0].data.decode("utf-8"))

        # Check expiry
        if datetime.now() > datetime.fromisoformat(qr_data["expiry"]):
            return jsonify({"expired": True, "message": "QR code expired"}), 200

        session["qr_data"] = qr_data
        session["otp_sent"] = False
        session["log_id"] = None

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# FACE → OTP
# =========================
@app.route('/verify_face_stream', methods=['POST'])
def verify_face_stream():
    try:
        qr = session.get("qr_data")
        if not qr:
            return jsonify({"error": "No active QR session"}), 400

        file_id = qr["file_id"]
        receiver_email = qr["receiver_email"]

        # Load known encoding
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT face_encoding FROM ReceiverFace
            WHERE file_id=? AND receiver_email=?
        """, (file_id, receiver_email))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Receiver face encoding not found"}), 400

        known_encoding = pickle.loads(row[0])

        # Process incoming frame
        frame_bytes = request.files['frame'].read()
        img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)

        if not enc:
            return jsonify({"error": "No face detected"}), 200

        distance = np.linalg.norm(known_encoding - enc[0])
        confidence = (1 - distance) * 100

        if distance > 0.6:
            return jsonify({"match": False})

        # -------------- MATCH SUCCESS ----------------
        if not session.get("otp_sent"):
            otp = str(random.randint(100000, 999999))

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO AccessLog(
                    file_id,
                    receiver_email,
                    face_match_result,
                    confidence_score,
                    otp_code,
                    otp_status
                )
                OUTPUT INSERTED.log_id
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                receiver_email,
                1,
                confidence,
                otp,
                "Sent"
            ))

            log_id = cursor.fetchone()[0]
            conn.commit()
            conn.close()

            session["log_id"] = log_id
            session["otp_sent"] = True

            # send OTP email
            send_otp_email(receiver_email, otp)

        return jsonify({"success": True, "match": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/send_otp', methods=['POST'])
@login_required
def send_otp_route():
    try:
        qr = session.get("qr_data")
        if not qr:
            return jsonify({"error": "QR session not found"}), 400

        otp = str(random.randint(100000, 999999))
        session['otp'] = otp
        session['otp_time'] = time.time()

        send_otp_email(qr["receiver_email"], otp)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# VERIFY OTP → DECRYPT
# =========================
@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    try:
        data = request.json
        otp = data.get("otp", "")
        log_id = session.get("log_id")

        if not log_id:
            return jsonify({"error": "No OTP session"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT otp_code, file_id
            FROM AccessLog
            WHERE log_id=?
        """, log_id)

        row = cursor.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": "Invalid log ID"}), 400

        correct_otp, file_id = row

        if otp != correct_otp:
            conn.close()
            return jsonify({"error": "Incorrect OTP"}), 200

        # Update log
        cursor.execute("""
            UPDATE AccessLog
            SET otp_status='Verified', access_result='Success'
            WHERE log_id=?
        """, log_id)
        conn.commit()
        conn.close()

        session["verified_file_id"] = file_id

        return jsonify({
            "success": True,
            "download_url": "/download_file"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_secure")
def download_secure_file():
    try:
        qr = session.get("qr_data")
        if not qr:
            return jsonify({"error": "No QR session"}), 400

        file_id = qr["file_id"]

        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Get file path
        cursor.execute("SELECT file_path FROM Files WHERE file_id=?", file_id)
        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "File metadata not found"}), 404

        relative_path = row[0]

        # 2. Get AES key material
        cursor.execute("""
            SELECT aes_key, iv, tag
            FROM FileKey
            WHERE file_id=?
        """, file_id)

        key_row = cursor.fetchone()
        conn.close()

        if not key_row:
            return jsonify({"error": "Missing AES key for this file"}), 500

        aes_key, iv, tag = key_row

        # 3. Load encrypted file
        abs_path = os.path.join(BASE_DIR, relative_path.lstrip("/"))

        if not os.path.exists(abs_path):
            return jsonify({"error": "Encrypted file missing"}), 404

        with open(abs_path, "rb") as f:
            encrypted_bytes = f.read()

        # 4. Decrypt AES-CBC (default)
        try:
            cipher = AES.new(aes_key, AES.MODE_CBC, iv)
            decrypted = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)

        except ValueError:
            # Fallback for AES-GCM
            cipher = AES.new(aes_key, AES.MODE_GCM, iv)
            decrypted = cipher.decrypt_and_verify(encrypted_bytes, tag)

        # 5. Send file
        return Response(
            decrypted,
            headers={
                "Content-Disposition": f"attachment; filename=secured_file"
            },
            mimetype="application/octet-stream"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem')) 