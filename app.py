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

# Try to import face_recognition, but continue if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

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

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['KNOWN_FACES'] = os.path.join(BASE_DIR, 'known_faces')
app.config['PROFILE_PIC_FOLDER'] = os.path.join(BASE_DIR, 'static', 'profile_pics')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
USER_DATA_DIR = os.path.join(BASE_DIR, 'data')
USER_DATA_FILE = os.path.join(USER_DATA_DIR, 'users.json')
PROFILE_PIC_URL_PREFIX = 'profile_pics'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'docx'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWN_FACES'], exist_ok=True)
os.makedirs(app.config['PROFILE_PIC_FOLDER'], exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Initialize default user if users.json doesn't exist
def initialize_default_user():
    if not os.path.exists(USER_DATA_FILE):
        from werkzeug.security import generate_password_hash
        default_users = {
            'cyloixy2610@gmail.com': {
                'name': 'Cindy',
                'email': 'cyloixy2610@gmail.com',
                'password': generate_password_hash('password123'),
                # Use bundled default profile picture if none uploaded
                'profile_picture': 'default.png'
            }
        }
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_users, f, indent=4)
        print("✓ Default user account created (Cindy / cyloixy2610@gmail.com)")

initialize_default_user()

def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    try:
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users):
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as file:
        json.dump(users, file, indent=4)


def get_current_user():
    email = session.get('user_email')
    if not email:
        return None
    users = load_users()
    return users.get(email)


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)

    return wrapped_view


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
    user = get_current_user()
    return render_template('index.html', user=user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        users = load_users()
        user = users.get(email)

        if user and check_password_hash(user['password'], password):
            session['user_email'] = email
            return redirect(url_for('dashboard'))

        error = 'Invalid email or password'

    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        profile_picture = request.files.get('profile_picture')

        if not name or not email or not password:
            error = 'All fields are required'
        elif password != confirm_password:
            error = 'Passwords do not match'
        else:
            users = load_users()
            if email in users:
                error = 'An account with this email already exists'
            else:
                profile_filename = save_profile_picture(profile_picture, email)
                # Fallback to bundled default avatar if user didn't upload one
                if not profile_filename:
                    profile_filename = 'default.png'
                users[email] = {
                    'name': name,
                    'email': email,
                    'password': generate_password_hash(password),
                    'profile_picture': profile_filename,
                }
                save_users(users)
                session['user_email'] = email
                return redirect(url_for('dashboard'))

    return render_template('register.html', error=error)


@app.route('/logout')
@login_required
def logout():
    session.pop('user_email', None)
    session.pop('otp', None)
    session.pop('qr_data', None)
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
SENDER_PASSWORD = "jnzb bvae knot evss"  # Your Gmail App Password

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Thread class for video capture
class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0, width=640, height=480, queue_size=2):
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

def load_known_faces(directory):
    if not os.path.isabs(directory):
        directory = os.path.join(BASE_DIR, directory)
    known_face_encodings = []
    known_face_names = []
    print(f"Loading encodings for faces from directory: {directory}")
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)
        return np.array([]), []

    # List all files in directory
    files = os.listdir(directory)
    print(f"Found {len(files)} files in directory")
    
    for filename in files:
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(directory, filename)
            name, _ = os.path.splitext(filename)
            pkl_path = os.path.join(directory, f"{name}.pkl")
            
            print(f"Processing file: {filename}")
            
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as pkl_file:
                        encoding = pickle.load(pkl_file)
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        print(f"Loaded encoding from {pkl_path}")
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
            else:
                try:
                    print(f"Generating new encoding for {image_path}")
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        encoding = face_encodings[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        print(f"Generated and saved encoding for {image_path}")

                        with open(pkl_path, 'wb') as pkl_file:
                            pickle.dump(encoding, pkl_file)
                    else:
                        print(f"No faces found in {image_path}. Skipping.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    known_face_encodings = np.array(known_face_encodings)
    print(f"Loaded {len(known_face_names)} known faces: {known_face_names}")
    return known_face_encodings, known_face_names

def verify_face_with_mediapipe():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(app.config['KNOWN_FACES'])
    
    # Initialize camera
    video_capture = VideoCaptureThread(src=0, width=720, height=720, queue_size=2)
    video_capture.start()
    
    try:
        process_every_n_frames = 2  # Process every 2nd frame
        frame_count = 0
        
        # Variables for face tracking
        previous_face_locations = []
        previous_face_names = []
        
        while True:
            if video_capture.more():
                frame = video_capture.read()
                frame_count += 1
                
                # Only process every nth frame
                if frame_count % process_every_n_frames == 0:
                    # Resize frame to 1/2 size for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces using HOG (faster than MediaPipe)
                    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
                    
                    if face_locations:
                        # Get face encodings for the detected faces
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        
                        for face_encoding in face_encodings:
                            if len(known_face_encodings) > 0:
                                # Compare with known faces using NumPy for faster computation
                                distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                                best_match_index = np.argmin(distances)
                                dist = distances[best_match_index]
                                
                                if dist <= 0.6:
                                    name = known_face_names[best_match_index]
                                    confidence = (1 - dist) * 100
                                    return name, confidence
                                else:
                                    return "Unknown", 0
                    
                    # Display the frame
                    for (top, right, bottom, left) in face_locations:
                        # Scale back up face locations
                        top *= 2
                        right *= 2
                        bottom *= 2
                        left *= 2
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    cv2.imshow("Face Verification", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return "Unknown", 0
    
    finally:
        video_capture.stop()
        video_capture.join()
        cv2.destroyAllWindows()

def generate_otp():
    """Generate a random 6-digit OTP"""
    otp = ""
    for i in range(6):
        otp += str(random.randint(0, 9))
    return otp

def send_otp_email(recipient_email, otp):
    try:
        print(f"Attempting to send email to: {recipient_email}")
        print(f"Connecting to SMTP server: smtp.gmail.com:587")
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
        server.set_debuglevel(1)  # Enable debug output
        
        print("Starting TLS...")
        server.starttls()
        
        from_mail = 'cyloixy2610@gmail.com'
        print(f"Logging in with email: {from_mail}")
        server.login(from_mail, 'jnzb bvae knot evss')
        
        msg = EmailMessage()
        msg['Subject'] = "OTP Verification"
        msg['From'] = from_mail
        msg['To'] = recipient_email
        msg.set_content(f"Your OTP is: {otp}")
        
        print(f"Sending email with OTP: {otp}")
        server.send_message(msg)
        print("Email sent successfully!")
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
        print("Please check if:")
        print("1. 2-Factor Authentication is enabled on Gmail")
        print("2. App Password is correctly generated")
        print("3. 'Less secure app access' is enabled (if not using App Password)")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected email error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_document():
    # Collect all receiver photos and emails
    photos = []
    emails = []
    index = 0
    
    # Collect all photos (photo_0, photo_1, etc.)
    while f'photo_{index}' in request.files:
        photo = request.files[f'photo_{index}']
        if photo and photo.filename != '':
            photos.append(photo)
        index += 1
    
    # Collect all emails (email_0, email_1, etc.)
    index = 0
    while f'email_{index}' in request.form:
        email = request.form.get(f'email_{index}', '').strip()
        if email:
            emails.append(email)
        index += 1
    
    if not photos or not emails:
        return jsonify({'error': 'At least one receiver with photo and email is required'}), 400
    
    if len(photos) != len(emails):
        return jsonify({'error': 'Number of photos and emails must match'}), 400
    
    # Determine file source: direct upload or link
    file_link = None
    if 'document' in request.files and request.files['document'].filename != '':
        # User uploaded a file directly - save it and use as link
        document_file = request.files['document']
        if not allowed_document(document_file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and DOCX files are allowed.'}), 400
        # Use first email for filename (or generate a generic one)
        base_email = emails[0] if emails else 'document'
        document_filename = secure_filename(f"{base_email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document_file.filename}")
        document_path = os.path.join(app.config['UPLOAD_FOLDER'], document_filename)
        document_file.save(document_path)
        file_link = url_for('get_uploaded_file', filename=document_filename, _external=True)
    else:
        # User provided a file link
        file_link = request.form.get('fileLink')
    
    if not file_link:
        return jsonify({'error': 'Please provide either a document file or a file link'}), 400
    
    # Get expiration time from form (in hours, default to 24 hours if not provided)
    expiration_hours = request.form.get('expiration_hours', '24')
    try:
        expiration_hours = float(expiration_hours)
        if expiration_hours <= 0:
            expiration_hours = 24  # Default to 24 hours if invalid
    except (ValueError, TypeError):
        expiration_hours = 24  # Default to 24 hours if invalid
    
    # Calculate expiration timestamp
    expiration_timestamp = time.time() + (expiration_hours * 3600)  # Convert hours to seconds
    
    # Process each receiver and generate QR codes
    qr_urls = []
    for i, (photo, email) in enumerate(zip(photos, emails)):
        # Save the photo
        photo_filename = secure_filename(f"{email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg")
        photo_path = os.path.join(app.config['KNOWN_FACES'], photo_filename)
        photo.save(photo_path)
        
        # Generate encryption key for this receiver
        key = Fernet.generate_key()
        f = Fernet(key)
        
        # Encrypt the file link
        encrypted_link = f.encrypt(file_link.encode())
        
        # Create QR code data
        qr_data = {
            'key': key.decode(),
            'encrypted_link': encrypted_link.decode(),
            'email': email,
            'expiration_timestamp': expiration_timestamp
        }
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR code
        qr_filename = f"qr_{email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
        qr_path = os.path.join(app.config['UPLOAD_FOLDER'], qr_filename)
        qr_image.save(qr_path)
        qr_url = url_for('get_uploaded_file', filename=qr_filename)
        qr_urls.append({
            'email': email,
            'qr_path': qr_url
        })
    
    return jsonify({
        'success': True,
        'message': f'Document processed, {len(qr_urls)} QR code(s) generated',
        'qr_codes': qr_urls
    })


@app.route('/scan', methods=['POST'])
@login_required
def scan_qr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded QR code image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read QR code
    image = cv2.imread(filepath)
    decoded_objs = decode(image)
    
    if not decoded_objs:
        return jsonify({'error': 'No QR code found in image'}), 400
    
    try:
        qr_data = json.loads(decoded_objs[0].data.decode())
        
        # Check if QR code has expired
        expiration_timestamp = qr_data.get('expiration_timestamp')
        if expiration_timestamp:
            current_time = time.time()
            if current_time > expiration_timestamp:
                # Calculate how long it has been expired
                expired_seconds = current_time - expiration_timestamp
                expired_hours = expired_seconds / 3600
                expired_days = expired_hours / 24
                
                if expired_days >= 1:
                    expired_message = f'QR code has expired {expired_days:.1f} days ago'
                elif expired_hours >= 1:
                    expired_message = f'QR code has expired {expired_hours:.1f} hours ago'
                else:
                    expired_minutes = expired_seconds / 60
                    expired_message = f'QR code has expired {expired_minutes:.1f} minutes ago'
                
                return jsonify({
                    'error': 'QR code has expired',
                    'message': expired_message,
                    'expired': True
                }), 400
        
        session['qr_data'] = qr_data
        
        return jsonify({
            'success': True,
            'message': 'QR code scanned successfully'
        })
            
    except Exception as e:
        return jsonify({'error': f'Error processing QR code: {str(e)}'}), 400

@app.route('/verify_face', methods=['POST'])
@login_required
def verify_face():
    try:
        print("Starting face verification...")
        user = get_current_user()
        # Start face recognition using MediaPipe
        name, confidence = verify_face_with_mediapipe()
        print(f"Face verification result - Name: {name}, Confidence: {confidence}")
        
        if name != "Unknown" and confidence > 50:
            print("Face verified successfully, generating OTP...")
            # Generate OTP
            otp = generate_otp()
            print(f"Generated OTP: {otp}")
            
            # Store OTP in session
            session['otp'] = otp
            session['otp_timestamp'] = time.time()
            
            # Send OTP email
            print("Attempting to send OTP email...")
            recipient_email = user['email'] if user else session.get('user_email')
            email_sent = send_otp_email(recipient_email, otp)
            
            if email_sent:
                print("OTP email sent successfully")
                return jsonify({
                    'success': True,
                    'message': 'Face verified, OTP sent to your email'
                })
            else:
                print("Failed to send OTP email")
                return jsonify({'error': 'Failed to send OTP'}), 500
        else:
            print(f"Face verification failed - Name: {name}, Confidence: {confidence}")
            return jsonify({
                'error': 'Face verification failed',
                'message': 'Face not recognized or confidence too low'
            }), 400
            
    except Exception as e:
        print(f"Face verification error: {str(e)}")
        return jsonify({'error': f'Face verification error: {str(e)}'}), 500

@app.route('/verify_otp', methods=['POST'])
@login_required
def verify_otp():
    data = request.get_json()
    user_otp = data.get('otp')
    stored_otp = session.get('otp')
    
    if not user_otp or not stored_otp:
        return jsonify({'error': 'Invalid OTP'}), 400
    
    # Simple OTP comparison
    if user_otp == stored_otp:
        # Clear the OTP from session after successful verification
        session.pop('otp', None)
        
        # Decrypt and return the file link
        qr_data = session.get('qr_data')
        if not qr_data:
            return jsonify({'error': 'Session expired'}), 400
        
        key = qr_data['key'].encode()
        encrypted_link = qr_data['encrypted_link'].encode()
        
        f = Fernet(key)
        decrypted_link = f.decrypt(encrypted_link).decode()
        
        return jsonify({
            'success': True,
            'message': 'OTP verified successfully',
            'file_link': decrypted_link
        })
    else:
        return jsonify({'error': 'Invalid OTP'}), 400

@app.route('/verify_face_stream', methods=['POST'])
@login_required
def verify_face_stream():
    try:
        if 'frame' not in request.files:
            print("No frame received in request")
            return jsonify({'error': 'No frame provided'}), 400
            
        frame_file = request.files['frame']
        # Read the frame as an image
        frame_array = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode frame")
            return jsonify({'error': 'Invalid frame data'}), 400
            
        print("Frame received and decoded successfully")
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        print("Starting face detection")
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_small_frame)
            
            if results.detections:
                print(f"Found {len(results.detections)} faces")
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = rgb_small_frame.shape
                    x_min = int(bboxC.xmin * iw)
                    y_min = int(bboxC.ymin * ih)
                    width = int(bboxC.width * iw)
                    height = int(bboxC.height * ih)
                    
                    top = y_min
                    right = x_min + width
                    bottom = y_min + height
                    left = x_min
                    
                    print(f"Face detected at coordinates: left={left}, top={top}, right={right}, bottom={bottom}")
                    
                    # Always return face location for visualization
                    face_location = {
                        'left': left * 2,
                        'top': top * 2,
                        'width': width * 2,
                        'height': height * 2
                    }
                    
                    # Encode the face
                    face_encoding = face_recognition.face_encodings(rgb_small_frame, [(top, right, bottom, left)])
                    
                    if face_encoding:
                        print("Face encoded successfully")
                        # Load known faces
                        known_face_encodings, known_face_names = load_known_faces(app.config['KNOWN_FACES'])
                        print(f"Loaded {len(known_face_names)} known faces")
                        
                        if len(known_face_encodings) > 0:
                            distances = np.linalg.norm(known_face_encodings - face_encoding[0], axis=1)
                            best_match_index = np.argmin(distances)
                            dist = distances[best_match_index]
                            
                            print(f"Best match distance: {dist}")
                            
                            if dist <= 0.6:
                                name = known_face_names[best_match_index]
                                confidence = float((1 - dist) * 100)  # Convert to Python float
                                print(f"Face recognized as {name} with {confidence}% confidence")
                                return jsonify({
                                    'success': True,
                                    'name': name,
                                    'confidence': confidence,
                                    'face_location': face_location,
                                    'should_close': bool(confidence > 50)  # Convert to Python boolean
                                })
                            else:
                                confidence = float((1 - dist) * 100)  # Convert to Python float
                                return jsonify({
                                    'success': False,
                                    'name': 'Unknown',
                                    'confidence': confidence,
                                    'face_location': face_location,
                                    'should_close': False
                                })
                        else:
                            return jsonify({
                                'success': False,
                                'face_location': face_location,
                                'should_close': False
                            })
                    else:
                        return jsonify({
                            'success': False,
                            'face_location': face_location,
                            'should_close': False
                        })
            else:
                return jsonify({'success': False})
            
    except Exception as e:
        print(f"Face verification error: {str(e)}")
        return jsonify({'success': False})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem')) 