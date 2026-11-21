
import os
import tempfile
import unittest

from src.recognition import (
    initialize_models,
    load_registered_faces
)


class TestFaceGallery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector, cls.recognizer = initialize_models()

    def test_empty_gallery(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            faces = load_registered_faces(
                self.detector,
                self.recognizer,
                temp_dir
            )

            self.assertEqual(faces, [])

    def test_corrupted_image(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            image_path = os.path.join(
                temp_dir, "corrupted.jpg"
            )

            with open(image_path, "w") as file:
                file.write("This is not a valid image")

            faces = load_registered_faces(
                self.detector,
                self.recognizer,
                temp_dir
            )

            self.assertEqual(faces, [])


if __name__ == "__main__":
    unittest.main()
