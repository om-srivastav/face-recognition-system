import unittest
from src.matching import classify_match, THRESHOLD

class TestFaceMatching(unittest.TestCase):
    def test_score_below_threshold(self):
        self.assertEqual(classify_match("Om", 0.30), "Unknown")

    def test_score_equal_to_threshold(self):
        self.assertEqual(classify_match("Om", THRESHOLD), "Om")

    def test_score_above_threshold(self):
        self.assertEqual(classify_match("Om", 0.80), "Om")

    def test_custom_threshold(self):
        self.assertEqual(classify_match("Om", 0.70, 0.75), "Unknown")

if __name__ == "__main__":
    unittest.main()
