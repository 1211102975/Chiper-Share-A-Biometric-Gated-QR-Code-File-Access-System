import unittest
from app import app


class TestGuestAccess(unittest.TestCase):
    """
    T16 – Guest users must not access protected features
    """

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    # -------------------------------
    # T16-1: Guest visits dashboard
    # -------------------------------
    def test_guest_dashboard_access(self):
        res = self.client.get("/dashboard")

        # Guest can view limited dashboard
        self.assertIn(res.status_code, (200, 302))

        text = res.get_data(as_text=True)
        self.assertTrue(
            "Guest" in text or "Login" in text or "Sign in" in text,
            "Guest indicator not shown"
        )

    # -------------------------------
    # T16-2: Guest tries upload
    # -------------------------------
    def test_guest_upload_blocked(self):
        res = self.client.post("/upload")

        # Backend must reject request
        self.assertNotEqual(res.status_code, 200)

    # -------------------------------
    # T16-3: Guest tries QR scan
    # -------------------------------
    def test_guest_scan_blocked(self):
        res = self.client.post("/scan")

        # Backend must reject request
        self.assertNotEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
    