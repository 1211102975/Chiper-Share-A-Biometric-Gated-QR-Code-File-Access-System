import unittest
import os
import json
from app import app
from db import get_db_connection
from werkzeug.security import generate_password_hash

class TestFileUpload(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if test user already exists
        cursor.execute("SELECT user_id FROM Users WHERE email = ?", ("uploadtest@gmail.com",))
        row = cursor.fetchone()

        if row:
            user_id = row[0]

            # Clean child tables first
            cursor.execute("""
                DELETE FROM QRCode 
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by = ?)
            """, user_id)

            cursor.execute("""
                DELETE FROM ReceiverFace 
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by = ?)
            """, user_id)

            cursor.execute("""
                DELETE FROM FileKey 
                WHERE file_id IN (SELECT file_id FROM Files WHERE uploaded_by = ?)
            """, user_id)

            cursor.execute("DELETE FROM Files WHERE uploaded_by = ?", user_id)
            cursor.execute("DELETE FROM Users WHERE user_id = ?", user_id)

        # Re-create test user
        cursor.execute("""
            INSERT INTO Users (name, email, password_hash, profile_pic_path)
            VALUES (?, ?, ?, ?)
        """, ("Uploader", "uploadtest@gmail.com",
              generate_password_hash("Pass1234"),
              "profile_pics/default.jpg"))
        conn.commit()

        cursor.execute("SELECT user_id FROM Users WHERE email=?", ("uploadtest@gmail.com",))
        self.user_id = cursor.fetchone()[0]
        conn.close()

        # Create logged-in session
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["user_email"] = "uploadtest@gmail.com"

    def test_T03_upload_and_encrypt_file(self):
        """
        T03 – Validate authenticated user can upload, encrypt, and generate QR
        """

        pdf = open("tests/sample.pdf", "rb")
        face = open("tests/face.jpg", "rb")

        response = self.client.post("/upload", data={
            "document": (pdf, "sample.pdf"),
            "photo_0": (face, "face.jpg"),
            "email_0": "receiver@test.com",
            "expiration_hours": "24"
        }, content_type="multipart/form-data")

        pdf.close()
        face.close()

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertEqual(len(data["qr_codes"]), 1)

        # Verify encrypted file exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_path FROM Files WHERE uploaded_by = ?
            ORDER BY file_id DESC
        """, self.user_id)
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        encrypted_path = os.path.join(app.root_path, row[0])
        self.assertTrue(os.path.exists(encrypted_path))

