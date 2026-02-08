import unittest
import json
from app import app
from db import get_db_connection
from werkzeug.security import generate_password_hash

class TestURLUpload(unittest.TestCase):
    """
    T04 – Validate system behavior when uploading via URL link
    """

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # Prepare test user
        conn = get_db_connection()
        cursor = conn.cursor()

        # Remove existing test user safely
        cursor.execute("SELECT user_id FROM Users WHERE email = ?", ("urltest@gmail.com",))
        row = cursor.fetchone()

        if row:
            user_id = row[0]

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

        # Create fresh test user
        cursor.execute("""
            INSERT INTO Users (name, email, password_hash, profile_pic_path)
            VALUES (?, ?, ?, ?)
        """, ("URL Tester", "urltest@gmail.com",
              generate_password_hash("UrlPass123"),
              "profile_pics/default.jpg"))
        conn.commit()

        cursor.execute("SELECT user_id FROM Users WHERE email=?", ("urltest@gmail.com",))
        self.user_id = cursor.fetchone()[0]
        conn.close()

        # Login user session
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user_id
            sess["user_email"] = "urltest@gmail.com"

    def test_T04_invalid_url(self):
        """
        Invalid URL must be rejected
        """

        face = open("tests/face.jpg", "rb")

        response = self.client.post("/upload", data={
            "fileLink": "https://invalid/broken.pdf",
            "photo_0": (face, "face.jpg"),
            "email_0": "receiver@test.com",
            "expiration_hours": "24"
        }, content_type="multipart/form-data")

        face.close()

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_T04_valid_url(self):
        """
        Valid URL must still be rejected because encrypted URL uploads
        are not supported by system design
        """

        face = open("tests/face.jpg", "rb")

        response = self.client.post("/upload", data={
            "fileLink": "https://example.com/document.pdf",
            "photo_0": (face, "face.jpg"),
            "email_0": "receiver@test.com",
            "expiration_hours": "24"
        }, content_type="multipart/form-data")

        face.close()

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        self.assertIn("error", data)

