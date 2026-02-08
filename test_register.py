import unittest
import os
import io
from app import app
from db import get_db_connection

class TestUserRegistration(unittest.TestCase):

    def setUp(self):
        # Enable testing mode
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
        self.client = app.test_client()

        # Clean test user if exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Users WHERE email = ?", ("johnlim@gmail.com",))
        conn.commit()
        conn.close()

    def test_T01_register_new_user(self):
        """
        T01 – Verify new user can register with valid credentials
        """

        # Step 1–6: Simulate form submission
        data = {
            "name": "John Lim",
            "email": "johnlim@gmail.com",
            "password": "SecurePass123",
            "confirm_password": "SecurePass123"
        }

        # Fake JPEG image
        image = (io.BytesIO(b"fake image bytes"), "profile.jpg")

        response = self.client.post(
            "/register",
            data={**data, "profile_picture": image},
            content_type="multipart/form-data",
            follow_redirects=False
        )

        # Step 7–9: Verify redirect happened
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.headers["Location"])

        # Verify user was inserted into database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, email, profile_pic_path FROM Users WHERE email = ?",
            ("johnlim@gmail.com",)
        )
        user = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(user)
        self.assertEqual(user[0], "John Lim")
        self.assertEqual(user[1], "johnlim@gmail.com")
        self.assertTrue("profile_pics" in user[2])

