from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    send_from_directory,
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
from datetime import datetime
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


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# app.config['KNOWN_FACES'] = os.path.join(BASE_DIR, 'known_faces')
app.config['PROFILE_PIC_FOLDER'] = os.path.join(BASE_DIR, 'static', 'profile_pics')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# USER_DATA_DIR = os.path.join(BASE_DIR, 'data')
# USER_DATA_FILE = os.path.join(USER_DATA_DIR, 'users.json')
PROFILE_PIC_URL_PREFIX = 'profile_pics'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'docx'}

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['KNOWN_FACES'], exist_ok=True)
os.makedirs(app.config['PROFILE_PIC_FOLDER'], exist_ok=True)
# os.makedirs(USER_DATA_DIR, exist_ok=True)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "cyloixy2610@gmail.com"
SENDER_PASSWORD = "gprb gqku lqdf iemg"   # Gmail App Password

def generate_otp():
    return str(random.randint(100000, 999999))

# # Initialize OTP storage file if it doesn't exist
# OTP_STORAGE_FILE = os.path.join(BASE_DIR, 'otp_storage.txt')
# if not os.path.exists(OTP_STORAGE_FILE):
#     try:
#         with open(OTP_STORAGE_FILE, 'w', encoding='utf-8') as f:
#             f.write("# OTP Storage File\n")
#             f.write("# Format: email|otp|timestamp\n")
#         print(f"✓ OTP storage file initialized at: {OTP_STORAGE_FILE}")
#     except Exception as e:
#         print(f"✗ Warning: Could not initialize OTP storage file: {e}")

# Initialize default user if users.json doesn't exist
# def initialize_default_user():
#     if not os.path.exists(USER_DATA_FILE):
#         from werkzeug.security import generate_password_hash
#         default_users = {
#             'cyloixy2610@gmail.com': {
#                 'name': 'Cindy',
#                 'email': 'cyloixy2610@gmail.com',
#                 'password': generate_password_hash('password123'),
#                 # Use bundled default profile picture if none uploaded
#                 'profile_picture': 'default.png'
#             }
#         }
#         with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
#             json.dump(default_users, f, indent=4)
#         print("✓ Default user account created (Cindy / cyloixy2610@gmail.com)")

# initialize_default_user()

# def load_users():
#     if not os.path.exists(USER_DATA_FILE):
#         return {}
#     try:
#         with open(USER_DATA_FILE, 'r', encoding='utf-8') as file:
#             return json.load(file)
#     except (json.JSONDecodeError, FileNotFoundError):
#         return {}


# def save_users(users):
#     with open(USER_DATA_FILE, 'w', encoding='utf-8') as file:
#         json.dump(users, file, indent=4)


def get_current_user():
    """
    Fetch the currently logged-in user from the SQL Server database using session['user_id'].
    Returns a dict with at least: user_id, name, email, profile_picture.
    """
    user_id = session.get('user_id')
    if not user_id:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Adjust selected columns if your Users table has different schema
        cursor.execute(
            "SELECT user_id, name, email FROM Users WHERE user_id = ?",
            user_id
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    # Basic user dict; profile_picture uses a default for now
    return {
        'user_id': row[0],
        'name': row[1],
        'email': row[2],
        'profile_picture': 'default.png',
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
    save_path = os.path.join(app.config['PROFILE_PIC_FOLDER'], stored_name)
    file.save(save_path)
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
        SELECT user_id, email
        FROM Users
        WHERE user_id = ?
    """, user_id)

    row = cursor.fetchone()
    conn.close()

    if not row:
        return redirect(url_for('logout'))

    user = {
        'user_id': row[0],
        'email': row[1]
    }

    return render_template('index.html', user=user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_id, password FROM Users WHERE email = ?",
        email
    )
    row = cursor.fetchone()
    conn.close()

    if row and check_password_hash(row[1], password):
        session['user_id'] = row[0]      # INTERNAL
        session['user_email'] = email   # EMAIL ONLY
        return redirect(url_for('dashboard'))

    return "Invalid credentials", 401


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

            # Check if email already exists
            cursor.execute("""
                SELECT user_id FROM Users WHERE email = ?
            """, email)

            if cursor.fetchone():
                error = 'An account with this email already exists'
            else:
                cursor.execute("""
                    INSERT INTO Users (name, email, password)
                    VALUES (?, ?, ?)
                """, name, email, generate_password_hash(password))

                conn.commit()

                # Get new user_id
                cursor.execute("""
                    SELECT user_id FROM Users WHERE email = ?
                """, email)
                user_id = cursor.fetchone()[0]

                session['user_id'] = user_id
                session['user_email'] = email

                conn.close()
                return redirect(url_for('dashboard'))

            conn.close()

    return render_template('register.html', error=error)

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
    if user and user.get('profile_picture'):
        profile_picture_url = url_for(
            'static', filename=f"{PROFILE_PIC_URL_PREFIX}/{user['profile_picture']}"
        )
    return render_template('profile.html', user=user, profile_picture_url=profile_picture_url)


@app.route('/uploads/<path:filename>')
@login_required
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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

# def load_known_faces(directory):
#     if not os.path.isabs(directory):
#         directory = os.path.join(BASE_DIR, directory)
#     known_face_encodings = []
#     known_face_names = []
#     print(f"Loading encodings for faces from directory: {directory}")
    
#     # Check if directory exists
#     if not os.path.exists(directory):
#         print(f"Creating directory: {directory}")
#         os.makedirs(directory)
#         return np.array([]), []

#     # List all files in directory
#     files = os.listdir(directory)
#     print(f"Found {len(files)} files in directory")
    
#     for filename in files:
#         if filename.lower().endswith((".jpg", ".png")):
#             image_path = os.path.join(directory, filename)
#             name, _ = os.path.splitext(filename)
#             pkl_path = os.path.join(directory, f"{name}.pkl")
            
#             print(f"Processing file: {filename}")
            
#             if os.path.exists(pkl_path):
#                 try:
#                     with open(pkl_path, 'rb') as pkl_file:
#                         encoding = pickle.load(pkl_file)
#                         known_face_encodings.append(encoding)
#                         known_face_names.append(name)
#                         print(f"Loaded encoding from {pkl_path}")
#                 except Exception as e:
#                     print(f"Error loading {pkl_path}: {e}")
#             else:
#                 try:
#                     print(f"Generating new encoding for {image_path}")
#                     image = face_recognition.load_image_file(image_path)
#                     face_encodings = face_recognition.face_encodings(image)
#                     if not encodings:
#                         return jsonify({'error': 'No face detected in uploaded photo'}), 400

#                     encoding_blob = pickle.dumps(encodings[0])

#                     conn = get_db_connection()
#                     cursor = conn.cursor()

#                     cursor.execute("""
#                         INSERT INTO UserFaceEncoding (user_id, face_encoding)
#                         VALUES (?, ?)
#                     """, recipient_user_id, encoding_blob)

#                     conn.commit()
#                     conn.close()

#     known_face_encodings = np.array(known_face_encodings)
#     print(f"Loaded {len(known_face_names)} known faces: {known_face_names}")
#     return known_face_encodings, known_face_names

# def verify_face_with_mediapipe():
#     # Load known faces
#     known_face_encodings, known_face_names = load_known_faces(app.config['KNOWN_FACES'])
    
#     # Initialize camera
#     video_capture = VideoCaptureThread(src=0, width=720, height=500, queue_size=2)
#     video_capture.start()
    
#     try:
#         process_every_n_frames = 2  # Process every 2nd frame
#         frame_count = 0
        
#         # Variables for face tracking
#         previous_face_locations = []
#         previous_face_names = []
        
#         while True:
#             if video_capture.more():
#                 frame = video_capture.read()
#                 frame_count += 1
                
#                 # Only process every nth frame
#                 if frame_count % process_every_n_frames == 0:
#                     # Resize frame to 1/2 size for faster processing
#                     small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#                     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
#                     # Detect faces using HOG (faster than MediaPipe)
#                     face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
                    
#                     if face_locations:
#                         # Get face encodings for the detected faces
#                         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        
#                         for face_encoding in face_encodings:
#                             if len(known_face_encodings) > 0:
#                                 # Compare with known faces using NumPy for faster computation
#                                 distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
#                                 best_match_index = np.argmin(distances)
#                                 dist = distances[best_match_index]
                                
#                                 if dist <= 0.6:
#                                     name = known_face_names[best_match_index]
#                                     confidence = (1 - dist) * 100
#                                     return name, confidence
#                                 else:
#                                     return "Unknown", 0
                    
#                     # Display the frame
#                     for (top, right, bottom, left) in face_locations:
#                         # Scale back up face locations
#                         top *= 2
#                         right *= 2
#                         bottom *= 2
#                         left *= 2
#                         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
#                     cv2.imshow("Face Verification", frame)
                    
#                     if cv2.waitKey(1) & 0xFF == ord("q"):
#                         return "Unknown", 0
    
#     finally:
#         video_capture.stop()
#         video_capture.join()
#         cv2.destroyAllWindows()

# def generate_otp():
#     """Generate a random 6-digit OTP"""
#     otp = ""
#     for i in range(6):
#         otp += str(random.randint(0, 9))
#     return otp

# def store_otp_to_file(email, otp):
#     """Store OTP in a txt file for verification"""
#     otp_storage_file = os.path.join(BASE_DIR, 'otp_storage.txt')
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
#     try:
#         # Ensure BASE_DIR exists
#         os.makedirs(BASE_DIR, exist_ok=True)
        
#         # Check if an OTP was already stored for this email in the current second
#         if os.path.exists(otp_storage_file):
#             try:
#                 with open(otp_storage_file, 'r', encoding='utf-8') as f:
#                     lines = f.readlines()
#                     # Check the last few lines (last 5) for recent entries with same timestamp
#                     for line in reversed(lines[-5:]):
#                         line = line.strip()
#                         if not line or line.startswith('#'):
#                             continue
#                         parts = line.split('|')
#                         if len(parts) >= 3:
#                             stored_email = parts[0].strip().lower()
#                             stored_timestamp = parts[2].strip()
#                             # If same email and same timestamp, skip storing duplicate
#                             if stored_email == email.strip().lower() and stored_timestamp == timestamp:
#                                 print(f"⚠ OTP already stored for {email} at {timestamp}. Skipping duplicate entry.")
#                                 return True
#             except Exception as check_error:
#                 print(f"Warning: Could not check for duplicates: {check_error}")
        
#         # Create file if it doesn't exist, or append if it does
#         with open(otp_storage_file, 'a', encoding='utf-8') as f:
#             data_line = f"{email}|{otp}|{timestamp}\n"
#             f.write(data_line)
#             f.flush()  # Ensure data is written immediately
#             os.fsync(f.fileno())  # Force write to disk
        
#         # Verify file was created/written
#         if os.path.exists(otp_storage_file):
#             file_size = os.path.getsize(otp_storage_file)
#             print(f"✓ OTP stored to file: {email} -> {otp} at {timestamp} (file size: {file_size} bytes)")
#         else:
#             print(f"✗ WARNING: OTP storage file was not created at: {otp_storage_file}")
#         return True
#     except PermissionError as e:
#         print(f"✗ Permission error storing OTP to file: {e}")
#         print(f"  Path: {otp_storage_file}")
#         import traceback
#         traceback.print_exc()
#         return False
#     except Exception as e:
#         print(f"✗ Error storing OTP to file: {e}")
#         print(f"  Attempted path: {otp_storage_file}")
#         import traceback
#         traceback.print_exc()
#         return False

# def get_otp_from_file(email):
#     """Retrieve the latest OTP for an email from the storage file"""
#     otp_storage_file = os.path.join(BASE_DIR, 'otp_storage.txt')
#     if not os.path.exists(otp_storage_file):
#         return None
    
#     try:
#         # Read all lines and find the latest OTP for this email
#         with open(otp_storage_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
        
#         # Find the most recent OTP for this email (read from bottom to top)
#         for line in reversed(lines):
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split('|')
#             if len(parts) >= 2 and parts[0].strip().lower() == email.strip().lower():
#                 return parts[1].strip()  # Return the OTP
#         return None
#     except Exception as e:
#         print(f"Error reading OTP from file: {e}")
#         return None

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
def upload_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document uploaded'}), 400

    file = request.files['document']
    if not allowed_document(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    file_link = url_for('get_uploaded_file', filename=filename, _external=True)

    key = Fernet.generate_key()
    encrypted_link = Fernet(key).encrypt(file_link.encode())

    qr_data = {
        'key': key.decode(),
        'encrypted_link': encrypted_link.decode(),
        'email': session['user_email'],
        'expiry': time.time() + 86400
    }

    qr = qrcode.make(json.dumps(qr_data))
    qr_name = f"qr_{int(time.time())}.png"
    qr_path = os.path.join(UPLOAD_FOLDER, qr_name)
    qr.save(qr_path)

    return jsonify({'qr': url_for('get_uploaded_file', filename=qr_name)})
    # # Collect all receiver photos and emails
    # photos = []
    # emails = []
    # index = 0
    
    # # Collect all photos (photo_0, photo_1, etc.)
    # while f'photo_{index}' in request.files:
    #     photo = request.files[f'photo_{index}']
    #     if photo and photo.filename != '':
    #         photos.append(photo)
    #     index += 1
    
    # # Collect all emails (email_0, email_1, etc.)
    # index = 0
    # while f'email_{index}' in request.form:
    #     email = request.form.get(f'email_{index}', '').strip()
    #     if email:
    #         emails.append(email)
    #     index += 1
    
    # if not photos or not emails:
    #     return jsonify({'error': 'At least one receiver with photo and email is required'}), 400
    
    # if len(photos) != len(emails):
    #     return jsonify({'error': 'Number of photos and emails must match'}), 400
    
    # # Determine file source: direct upload or link
    # file_link = None
    # if 'document' in request.files and request.files['document'].filename != '':
    #     # User uploaded a file directly - save it and use as link
    #     document_file = request.files['document']
    #     if not allowed_document(document_file.filename):
    #         return jsonify({'error': 'Invalid file type. Only PDF and DOCX files are allowed.'}), 400
    #     # Use first email for filename (or generate a generic one)
    #     base_email = emails[0] if emails else 'document'
    #     document_filename = secure_filename(f"{base_email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document_file.filename}")
    #     document_path = os.path.join(app.config['UPLOAD_FOLDER'], document_filename)
    #     document_file.save(document_path)
    #     file_link = url_for('get_uploaded_file', filename=document_filename, _external=True)
    # else:
    #     # User provided a file link
    #     file_link = request.form.get('fileLink')
    
    # if not file_link:
    #     return jsonify({'error': 'Please provide either a document file or a file link'}), 400
    
    # # Get expiration time from form (in hours, default to 24 hours if not provided)
    # expiration_hours = request.form.get('expiration_hours', '24')
    # try:
    #     expiration_hours = float(expiration_hours)
    #     if expiration_hours <= 0:
    #         expiration_hours = 24  # Default to 24 hours if invalid
    # except (ValueError, TypeError):
    #     expiration_hours = 24  # Default to 24 hours if invalid
    
    # # Calculate expiration timestamp
    # expiration_timestamp = time.time() + (expiration_hours * 3600)  # Convert hours to seconds
    
    # # Process each receiver and generate QR codes
    # qr_urls = []
    # for i, (photo, email) in enumerate(zip(photos, emails)):
    #     # Save the photo
    #     photo_filename = secure_filename(f"{email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg")
    #     photo_path = os.path.join(app.config['KNOWN_FACES'], photo_filename)
    #     photo.save(photo_path)
        
    #     # Get recipient user_id using email
    #     conn = get_db_connection()
    #     cursor = conn.cursor()
        
    #     cursor.execute("SELECT user_id FROM [User] WHERE email = ?", email)
    #     row = cursor.fetchone()
    #     if not row:
    #         conn.close()
    #         return jsonify({'error': f'Recipient {email} not registered'}), 400

    #     recipient_user_id = row[0]
        
    #     #  Load image and generate face encoding
    #     image = face_recognition.load_image_file(photo_path)
    #     encodings = face_recognition.face_encodings(image)

    #     if not encodings:
    #         conn.close()
    #         return jsonify({'error': 'No face detected in uploaded photo'}), 400

    #     #  Convert encoding to binary
    #     import pickle
    #     encoding_blob = pickle.dumps(encodings[0])

    #     #  Save encoding to SQL Server
    #     from db import get_db_connection

    #     conn = get_db_connection()
    #     cursor = conn.cursor()

    #     cursor.execute("""
    #         INSERT INTO UserFaceEncoding (user_id, face_encoding)
    #         VALUES (?, ?)
    #     """, recipient_user_id, encoding_blob)

    #     conn.commit()
    #     conn.close()
        
    #     # Generate encryption key for this receiver
    #     key = Fernet.generate_key()
    #     f = Fernet(key)
        
    #     # Encrypt the file link
    #     encrypted_link = f.encrypt(file_link.encode())
        
    #     # Create QR code data
    #     qr_data = {
    #         'key': key.decode(),
    #         'encrypted_link': encrypted_link.decode(),
    #         'email': email,
    #         'expiration_timestamp': expiration_timestamp
    #     }
        
    #     # Generate QR code
    #     qr = qrcode.QRCode(version=1, box_size=10, border=5)
    #     qr.add_data(json.dumps(qr_data))
    #     qr.make(fit=True)
    #     qr_image = qr.make_image(fill_color="black", back_color="white")
        
    #     # Save QR code
    #     qr_filename = f"qr_{email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
    #     qr_path = os.path.join(app.config['UPLOAD_FOLDER'], qr_filename)
    #     qr_image.save(qr_path)
    #     qr_url = url_for('get_uploaded_file', filename=qr_filename)
    #     qr_urls.append({
    #         'email': email,
    #         'qr_path': qr_url
    #     })
    
    # return jsonify({
    #     'success': True,
    #     'message': f'Document processed, {len(qr_urls)} QR code(s) generated',
    #     'qr_codes': qr_urls
    # })
@app.route('/scan', methods=['POST'])
@login_required
def scan_qr():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No QR image'}), 400

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    img = cv2.imread(path)
    decoded = decode(img)

    if not decoded:
        return jsonify({'error': 'Invalid QR'}), 400

    qr_data = json.loads(decoded[0].data.decode())

    if time.time() > qr_data['expiry']:
        return jsonify({'error': 'QR expired'}), 400

    session['qr_data'] = qr_data
    return jsonify({'success': True})
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file provided'}), 400
    
    # file = request.files['file']
    # if file.filename == '':
    #     return jsonify({'error': 'No file selected'}), 400
    
    # # Save the uploaded QR code image
    # filename = secure_filename(file.filename)
    # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # file.save(filepath)
    
    # # Read QR code
    # image = cv2.imread(filepath)
    # decoded_objs = decode(image)
    
    # if not decoded_objs:
    #     return jsonify({'error': 'No QR code found in image'}), 400
    
    # try:
    #     qr_data = json.loads(decoded_objs[0].data.decode())
        
    #     # Check if QR code has expired
    #     expiration_timestamp = qr_data.get('expiration_timestamp')
    #     if expiration_timestamp:
    #         current_time = time.time()
    #         if current_time > expiration_timestamp:
    #             # Calculate how long it has been expired
    #             expired_seconds = current_time - expiration_timestamp
    #             expired_hours = expired_seconds / 3600
    #             expired_days = expired_hours / 24
                
    #             if expired_days >= 1:
    #                 expired_message = f'QR code has expired {expired_days:.1f} days ago'
    #             elif expired_hours >= 1:
    #                 expired_message = f'QR code has expired {expired_hours:.1f} hours ago'
    #             else:
    #                 expired_minutes = expired_seconds / 60
    #                 expired_message = f'QR code has expired {expired_minutes:.1f} minutes ago'
                
    #             return jsonify({
    #                 'error': 'QR code has expired',
    #                 'message': expired_message,
    #                 'expired': True
    #             }), 400
        
    #     session['qr_data'] = qr_data
        
    #     return jsonify({
    #         'success': True,
    #         'message': 'QR code scanned successfully'
    #     })
            
    # except Exception as e:
    #     return jsonify({'error': f'Error processing QR code: {str(e)}'}), 400
    
# =========================
# FACE → OTP
# =========================
@app.route('/verify_face', methods=['POST'])
@login_required
def verify_face():
    user_id = session['user_id']

    result, confidence = face_recognition_pipeline(user_id)

    if result != "Match":
        return jsonify({'error': 'Face verification failed'}), 400

    otp = generate_otp()
    session['otp'] = otp
    session['otp_time'] = time.time()

    send_otp_email(session['user_email'], otp)

    return jsonify({'success': True, 'message': 'OTP sent'})
    # if 'user_id' not in session:
    #     return jsonify({'error': 'Not logged in'}), 401

    # user_id = session['user_id']

    # result, confidence = face_recognition_pipeline(user_id)

    # if result == "Match":
    #     otp = generate_otp()
    #     session['otp'] = otp
    #     session['otp_time'] = time.time()

    #     recipient_email = session['user_email']
    #     send_otp_email(recipient_email, otp)

    #     return jsonify({
    #         'success': True,
    #         'message': 'Face verified. OTP sent to email.'
    #     })

    # return jsonify({'error': 'Face verification failed'}), 400

# @app.route('/send_otp', methods=['POST'])
# @login_required
# def send_otp():
#     """Generate and send OTP after face verification"""
#     try:
#         # Check if OTP was already sent recently (within last 60 seconds)
#         if 'otp_sent_timestamp' in session:
#             time_since_last_otp = time.time() - session['otp_sent_timestamp']
#             if time_since_last_otp < 60:  # 60 seconds cooldown
#                 print(f"OTP already sent {time_since_last_otp:.1f} seconds ago. Skipping duplicate request.")
#                 return jsonify({
#                     'success': True,
#                     'message': 'OTP already sent. Please check your email.',
#                     'already_sent': True
#                 })
        
#         print("Generating and sending OTP after face verification...")
        
#         # Generate OTP
#         otp = generate_otp()
#         print(f"Generated OTP: {otp}")
        
#         # Store OTP in session
#         session['otp'] = otp
#         session['otp_timestamp'] = time.time()
#         session['otp_sent_timestamp'] = time.time()  # Track when OTP was sent
        
#         # Get receiver email from QR data
#         qr_data = session.get('qr_data')
#         if not qr_data:
#             return jsonify({'error': 'QR data not found in session'}), 400
        
#         recipient_email = qr_data.get('email')
#         if not recipient_email:
#             return jsonify({'error': 'Receiver email not found'}), 400
        
#         print(f"OTP will be sent to receiver email: {recipient_email}")
        
#         # Send OTP email (this will also store OTP to file)
#         email_sent = send_otp_email(recipient_email, otp)
        
#         if email_sent:
#             print("OTP email sent successfully")
#             return jsonify({
#                 'success': True,
#                 'message': 'OTP sent to receiver email'
#             })
#         else:
#             print("Failed to send OTP email")
#             # Clear the timestamp if email failed
#             session.pop('otp_sent_timestamp', None)
#             return jsonify({'error': 'Failed to send OTP email. Check server logs for details.'}), 500
            
#     except Exception as e:
#         print(f"Error in send_otp: {str(e)}")
#         session.pop('otp_sent_timestamp', None)
#         return jsonify({'error': f'Error sending OTP: {str(e)}'}), 500

# =========================
# VERIFY OTP → DECRYPT
# =========================
@app.route('/verify_otp', methods=['POST'])
@login_required
def verify_otp():
    user_otp = request.form['otp']

    if 'otp' not in session:
        return jsonify({'error': 'OTP expired'}), 400

    if time.time() - session['otp_time'] > 300:
        session.pop('otp', None)
        return jsonify({'error': 'OTP expired'}), 400

    if user_otp != session['otp']:
        return jsonify({'error': 'Invalid OTP'}), 400

    qr = session['qr_data']
    f = Fernet(qr['key'].encode())
    link = f.decrypt(qr['encrypted_link'].encode()).decode()

    return jsonify({'success': True, 'file_link': link})
    # user_otp = request.form['otp']

    # if 'otp' not in session:
    #     return jsonify({'error': 'OTP expired'}), 400

    # if time.time() - session['otp_time'] > 300:
    #     session.pop('otp', None)
    #     return jsonify({'error': 'OTP expired'}), 400

    # if user_otp != session['otp']:
    #     return jsonify({'error': 'Invalid OTP'}), 400

    # qr_data = session.get('qr_data')
    # fernet = Fernet(qr_data['key'].encode())
    # decrypted_link = fernet.decrypt(qr_data['encrypted_link'].encode()).decode()

    # return jsonify({
    #     'success': True,
    #     'download_link': decrypted_link
    # })

# @app.route('/verify_face_stream', methods=['POST'])
# @login_required
# def verify_face_stream():
#     try:
#         if 'frame' not in request.files:
#             print("No frame received in request")
#             return jsonify({'error': 'No frame provided'}), 400
            
#         frame_file = request.files['frame']
#         # Read the frame as an image
#         frame_array = np.frombuffer(frame_file.read(), np.uint8)
#         frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
#         if frame is None:
#             print("Failed to decode frame")
#             return jsonify({'error': 'Invalid frame data'}), 400
            
#         print("Frame received and decoded successfully")
        
#         # Resize frame for faster processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
#         print("Starting face detection")
#         with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#             results = face_detection.process(rgb_small_frame)
            
#             if results.detections:
#                 print(f"Found {len(results.detections)} faces")
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     ih, iw, _ = rgb_small_frame.shape
#                     x_min = int(bboxC.xmin * iw)
#                     y_min = int(bboxC.ymin * ih)
#                     width = int(bboxC.width * iw)
#                     height = int(bboxC.height * ih)
                    
#                     top = y_min
#                     right = x_min + width
#                     bottom = y_min + height
#                     left = x_min
                    
#                     print(f"Face detected at coordinates: left={left}, top={top}, right={right}, bottom={bottom}")
                    
#                     # Always return face location for visualization
#                     face_location = {
#                         'left': left * 2,
#                         'top': top * 2,
#                         'width': width * 2,
#                         'height': height * 2
#                     }
                    
#                     # Encode the face
#                     face_encoding = face_recognition.face_encodings(rgb_small_frame, [(top, right, bottom, left)])
                    
#                     if face_encoding:
#                         print("Face encoded successfully")
#                         # Load known faces
#                         known_face_encodings, known_face_names = load_known_faces(app.config['KNOWN_FACES'])
#                         print(f"Loaded {len(known_face_names)} known faces")
                        
#                         if len(known_face_encodings) > 0:
#                             distances = np.linalg.norm(known_face_encodings - face_encoding[0], axis=1)
#                             best_match_index = np.argmin(distances)
#                             dist = distances[best_match_index]
                            
#                             print(f"Best match distance: {dist}")
                            
#                             if dist <= 0.6:
#                                 name = known_face_names[best_match_index]
#                                 confidence = float((1 - dist) * 100)  # Convert to Python float
#                                 print(f"Face recognized as {name} with {confidence}% confidence")
#                                 return jsonify({
#                                     'success': True,
#                                     'name': name,
#                                     'confidence': confidence,
#                                     'face_location': face_location,
#                                     'should_close': bool(confidence > 50)  # Convert to Python boolean
#                                 })
#                             else:
#                                 confidence = float((1 - dist) * 100)  # Convert to Python float
#                                 return jsonify({
#                                     'success': False,
#                                     'name': 'Unknown',
#                                     'confidence': confidence,
#                                     'face_location': face_location,
#                                     'should_close': False
#                                 })
#                         else:
#                             return jsonify({
#                                 'success': False,
#                                 'face_location': face_location,
#                                 'should_close': False
#                             })
#                     else:
#                         return jsonify({
#                             'success': False,
#                             'face_location': face_location,
#                             'should_close': False
#                         })
#             else:
#                 return jsonify({'success': False})
            
#     except Exception as e:
#         print(f"Face verification error: {str(e)}")
#         return jsonify({'success': False})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem')) 