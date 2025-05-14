import unittest
from detector.model import PhishingDetector

class TestPhishingDetector(unittest.TestCase):

    def setUp(self):
        self.detector = PhishingDetector()

    def test_train_model(self):
        # Assuming we have a mock dataset for testing
        mock_data = [
            {"text": "Congratulations! You've won a prize!", "label": 1},
            {"text": "Important update regarding your account.", "label": 0}
        ]
        self.detector.train_model(mock_data)
        self.assertIsNotNone(self.detector.model)

    def test_predict_phishing(self):
        self.detector.train_model([
            {"text": "Congratulations! You've won a prize!", "label": 1},
            {"text": "Important update regarding your account.", "label": 0}
        ])
        result = self.detector.predict("You've won a lottery!")
        self.assertEqual(result, 1)

    def test_predict_legitimate(self):
        self.detector.train_model([
            {"text": "Congratulations! You've won a prize!", "label": 1},
            {"text": "Important update regarding your account.", "label": 0}
        ])
        result = self.detector.predict("Your account statement is ready.")
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()