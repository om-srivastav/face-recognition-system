import os
import tempfile
import unittest
import cv2
import numpy as np

from src.recognition import initialize_models, load_registered_faces


class TestNoFaceImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector, cls.recognizer = initialize_models()

    def test_image_without_face(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "no_face.jpg")
            blank_image = np.zeros((500, 500, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(image_path, blank_image))

            faces = load_registered_faces(
                self.detector, self.recognizer, temp_dir
            )

            self.assertEqual(faces, [])


if __name__ == "__main__":
    unittest.main()
