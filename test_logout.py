import unittest
from app import app


class TestLogout(unittest.TestCase):
    """
    T17 – Logout must clear session and return user to Guest mode
    """

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

        # Simulate logged-in user
        with self.client.session_transaction() as sess:
            sess["user_id"] = 999
            sess["user_email"] = "user@test.com"

    # -------------------------------
    # T17: Logout clears session
    # -------------------------------
    def test_T17_logout_clears_session(self):

        # Step 1 — Call logout
        res = self.client.get("/logout", follow_redirects=True)

        # Step 2 — Must return dashboard
        self.assertIn(res.status_code, (200, 302))

        # Step 3 — Session must be cleared
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)
            self.assertNotIn("user_email", sess)

        # Step 4 — Dashboard should show Guest
        html = res.get_data(as_text=True)
        self.assertTrue(
            "Guest" in html or "Login" in html or "Sign in" in html,
            "Dashboard did not show Guest view"
        )

        # Step 5 — Upload must be blocked
        upload = self.client.post("/upload")
        self.assertNotEqual(upload.status_code, 200)


if __name__ == "__main__":
    unittest.main()
