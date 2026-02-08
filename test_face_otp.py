import unittest
import os
import json
import pyodbc
import pickle
from datetime import datetime, timedelta

from app import app
from db import get_db_connection
import face_recognition


TEST_EMAIL = "face@test.com"
TEST_FACE = os.path.join("tests", "face.jpg")
QR_PATH = os.path.join("static", "qr_codes", "test_qr.png")


class TestFaceOTP(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

        os.makedirs("static/qr_codes", exist_ok=True)
        open(QR_PATH, "wb").close()   # fake QR image

        conn = get_db_connection()
        cursor = conn.cursor()

        # Clean up safely (FK order)
        cursor.execute("DELETE FROM AccessLog WHERE receiver_email=?", (TEST_EMAIL,))
        cursor.execute("DELETE FROM ReceiverFace WHERE receiver_email=?", (TEST_EMAIL,))
        cursor.execute("DELETE FROM QRCode WHERE receiver_email=?", (TEST_EMAIL,))
        cursor.execute("DELETE FROM Files WHERE uploaded_by IN (SELECT user_id FROM Users WHERE email=?)", (TEST_EMAIL,))
        cursor.execute("DELETE FROM Users WHERE email=?", (TEST_EMAIL,))
        conn.commit()

        # User
        cursor.execute("""
            INSERT INTO Users(name, email, password_hash)
            OUTPUT INSERTED.user_id
            VALUES ('Face Test', ?, 'hash')
        """, (TEST_EMAIL,))
        self.user_id = cursor.fetchone()[0]

        # File
        cursor.execute("""
            INSERT INTO Files(uploaded_by, file_path, expiration_timestamp)
            OUTPUT INSERTED.file_id
            VALUES (?, 'static/uploads/test.enc', DATEADD(hour,1,GETDATE()))
        """, self.user_id)
        self.file_id = cursor.fetchone()[0]

        # QR metadata
        expiry = (datetime.now() + timedelta(hours=1)).isoformat()
        self.qr_data = {
            "file_id": self.file_id,
            "receiver_email": TEST_EMAIL,
            "expiry": expiry
        }

        cursor.execute("""
            INSERT INTO QRCode(file_id, receiver_email, qr_metadata, qr_image_path)
            OUTPUT INSERTED.qr_id
            VALUES (?, ?, ?, ?)
        """, (self.file_id, TEST_EMAIL, json.dumps(self.qr_data), QR_PATH))
        self.qr_id = cursor.fetchone()[0]

        # Face encoding
        img = face_recognition.load_image_file(TEST_FACE)
        enc = face_recognition.face_encodings(img)[0]

        cursor.execute("""
            INSERT INTO ReceiverFace(file_id, receiver_email, face_encoding, photo_path)
            VALUES (?, ?, ?, ?)
        """, (self.file_id, TEST_EMAIL, pickle.dumps(enc), TEST_FACE))

        conn.commit()
        conn.close()

        # Simulate QR scan
        with self.client.session_transaction() as sess:
            sess["qr_data"] = self.qr_data
            sess["otp_sent"] = False
            sess["log_id"] = None

    # -------------------------------------------------
    # T10 – T11
    # -------------------------------------------------
    def test_T10_T11_face_match_generates_single_otp(self):

        # First face scan
        with open(TEST_FACE, "rb") as f:
            r1 = self.client.post("/verify_face_stream", data={
                "frame": f
            }, content_type="multipart/form-data")

        self.assertEqual(r1.status_code, 200)

        # Second scan (should NOT create new OTP)
        with open(TEST_FACE, "rb") as f:
            r2 = self.client.post("/verify_face_stream", data={
                "frame": f
            }, content_type="multipart/form-data")

        self.assertEqual(r2.status_code, 200)

        # Validate database
        conn = get_db_connection()
        cursor = conn.cursor()


        cursor.execute("""
            SELECT COUNT(*)
            FROM AccessLog
            WHERE file_id=? AND receiver_email=?
        """, (self.file_id, TEST_EMAIL))

        count = cursor.fetchone()[0]


        cursor.execute("""
            SELECT TOP 1 face_match_result, otp_status
            FROM AccessLog
            WHERE file_id=? AND receiver_email=?
            ORDER BY access_time DESC
        """, (self.file_id, TEST_EMAIL))

        face_match, otp_status = cursor.fetchone()
        conn.close()

        self.assertEqual(count, 1)
        self.assertEqual(face_match, 1)
        self.assertEqual(otp_status, "Sent")



