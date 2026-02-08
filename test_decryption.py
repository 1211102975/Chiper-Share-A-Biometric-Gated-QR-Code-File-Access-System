import unittest
import json
import io
import cv2
from app import app
from db import get_db_connection
from werkzeug.security import generate_password_hash

class TestSecureDecryption(unittest.TestCase):
    """
    T06 – Ensure encrypted file is successfully decrypted after OTP verification
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # ---------- Clean old test data ----------
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT user_id FROM Users WHERE email=?", ("decrypt@gmail.com",))
        row = cursor.fetchone()

        if row:
            uid = row[0]

            cursor.execute("""
                DELETE FROM AccessLog
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by=?)
            """, uid)

            cursor.execute("""
                DELETE FROM ReceiverFace
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by=?)
            """, uid)

            cursor.execute("""
                DELETE FROM QRCode
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by=?)
            """, uid)

            cursor.execute("""
                DELETE FROM FileKey
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by=?)
            """, uid)

            cursor.execute("DELETE FROM Files WHERE uploaded_by=?", uid)
            cursor.execute("DELETE FROM Users WHERE user_id=?", uid)

        # ---------- Create test user ----------
        cursor.execute("""
            INSERT INTO Users (name, email, password_hash, profile_pic_path)
            VALUES (?, ?, ?, ?)
        """, ("Decrypt User", "decrypt@gmail.com",
              generate_password_hash("Decrypt123"),
              "profile_pics/default.jpg"))
        conn.commit()

        cursor.execute("SELECT user_id FROM Users WHERE email=?", ("decrypt@gmail.com",))
        self.user_id = cursor.fetchone()[0]
        conn.close()

        # ---------- Login ----------
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["user_email"] = "decrypt@gmail.com"

        # ---------- Upload encrypted file ----------
        with open("tests/sample.pdf", "rb") as pdf, open("tests/face.jpg", "rb") as face:
            upload = self.client.post("/upload", data={
                "document": (pdf, "report.pdf"),
                "photo_0": (face, "face.jpg"),
                "email_0": "receiver@test.com",
                "expiration_hours": "24"
            }, content_type="multipart/form-data")

        self.assertEqual(upload.status_code, 200)

        data = json.loads(upload.data)
        self.qr_id = data["qr_codes"][0]["qr_id"]

        # ---------- Get QR ----------
        qr_img = self.client.get(f"/qr/{self.qr_id}")
        self.qr_bytes = bytes(qr_img.data)

    def test_T06_decrypt_after_otp(self):
        # ---------- Scan QR ----------
        qr_file = io.BytesIO(self.qr_bytes)
        qr_file.name = "qr.png"

        scan = self.client.post("/scan", data={
            "file": (qr_file, "qr.png")
        }, content_type="multipart/form-data")

        self.assertEqual(scan.status_code, 200)

        # ---------- Face verification ----------
        img = cv2.imread("tests/face.jpg")
        _, buf = cv2.imencode(".jpg", img)

        frame_file = io.BytesIO(buf.tobytes())
        frame_file.name = "frame.jpg"

        face = self.client.post("/verify_face_stream", data={
            "frame": (frame_file, "frame.jpg")
        }, content_type="multipart/form-data")


        self.assertEqual(face.status_code, 200)

        # ---------- Fetch OTP ----------
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 otp_code, log_id FROM AccessLog ORDER BY log_id DESC")
        otp, log_id = cursor.fetchone()
        conn.close()

        # ---------- Verify OTP ----------
        verify = self.client.post("/verify_otp", json={"otp": otp})
        self.assertEqual(verify.status_code, 200)

        # ---------- Download decrypted file ----------
        download = self.client.get("/download_secure")
        self.assertEqual(download.status_code, 200)

        decrypted = download.data

        with open("tests/sample.pdf", "rb") as f:
            original = f.read()

        # ---------- Compare content ----------
        self.assertEqual(decrypted[:100], original[:100])

        # ---------- Verify log ----------
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT access_result FROM AccessLog WHERE log_id=?", log_id)
        status = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(status, "Success")

