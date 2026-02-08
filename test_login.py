import unittest
from app import app
from db import get_db_connection
from werkzeug.security import generate_password_hash

class TestUserLogin(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # Ensure test user exists
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM Users WHERE email = ?", ("testuser@gmail.com",))

        password_hash = generate_password_hash("Test@123")

        cursor.execute("""
            INSERT INTO Users (name, email, password_hash, profile_pic_path)
            VALUES (?, ?, ?, ?)
        """, ("Test User", "testuser@gmail.com", password_hash, "profile_pics/default.jpg"))

        conn.commit()
        conn.close()

    def test_T02_login_and_access_dashboard(self):
        """
        T02 – Validate registered user can login and access dashboard
        """

        # Step 10–14: Login
        response = self.client.post("/login", data={
            "email": "testuser@gmail.com",
            "password": "Test@123"
        }, follow_redirects=False)

        # Step 16: Must redirect to dashboard
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.headers["Location"])

        # Step 17–18: Access dashboard
        dashboard = self.client.get("/dashboard")

        self.assertEqual(dashboard.status_code, 200)
        html = dashboard.data.decode()

        # Verify navbar user info
        self.assertIn("testuser@gmail.com", html)
        self.assertIn("Test User", html)

        # Verify upload section is enabled
        self.assertTrue(
            "upload" in html.lower(),
            "Upload section not found"
        )

