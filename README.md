# Chiper-Share-A-Biometric-Gated-QR-Code-File-Access-System

## Project Overview
Chiper Share is a secure web-based file access system developed as a Final Year Project (FYP 2).  
The system protects sensitive files using **multi-layer authentication**, including **QR code verification, biometric facial recognition, and one-time password (OTP) authentication**.

Files are encrypted using **AES-256-GCM**, and the QR code stores only reference metadata, not the encrypted file or encryption keys. This prevents unauthorized access even if a QR code is intercepted.

---

## Key Features
- AES-256-GCM encrypted file storage  
- QR code–based file access reference  
- Real-time facial recognition via webcam  
- Email-based OTP verification  
- Secure user authentication and session management  
- QR code expiration and access logging  

---

## Technologies Used
- **Backend:** Python, Flask  
- **Frontend:** HTML, CSS, Bootstrap, JavaScript  
- **Biometrics:** OpenCV, face_recognition, MediaPipe  
- **Encryption:** AES-256-GCM (PyCryptodome)  
- **QR Code:** qrcode, pyzbar  
- **Email:** SMTP (Gmail)  
- **Database:** SQLite  

---

## System Flow
1. Sender uploads a file (file is encrypted on the server)  
2. QR code is generated with file reference metadata  
3. Receiver scans the QR code  
4. Facial verification is performed using live webcam feed  
5. OTP is sent to the registered email  
6. File is decrypted and downloaded after successful verification  

---

## How to Run
```bash
pip install -r requirements.txt
python app.py
