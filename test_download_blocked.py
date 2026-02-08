import unittest
from app import app


class TestDownloadBlocked(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        app.config["TESTING"] = True

        # Simulate QR + Face done but OTP NOT verified
        with self.client.session_transaction() as sess:
            sess["qr_data"] = {"file_id": 1, "receiver_email": "test@test.com"}
            sess["otp_verified"] = False

    def test_T15_block_download_without_otp(self):
        response = self.client.get("/download")
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
