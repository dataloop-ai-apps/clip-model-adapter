import os
import unittest
import numpy as np
from PIL import Image
from pathlib import Path

from model_adapter import ClipAdapter

PATH = Path(__file__).resolve().parent / "tests" / "assets" / "unittest"



class TestRunner(unittest.TestCase):
    def setUp(self):
        self.assets_path = os.path.join(PATH, 'assets')
        self.image_features = 'test_img.json'
        self.image_file = 'test_img.jpg'
        self.text_features = 'text.json'

        self.adapter = ClipAdapter()
        self.adapter.load('./')

    def test_image_embedding(self):
        """Test embedding function with an Image object."""
        img = Image.new('RGB', (64, 64), color = 'red')
        result = embedding(img)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_text_embedding(self):
        """Test embedding function with a text string."""
        text = "Hello, world!"
        result = embedding(text)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_invalid_input_type(self):
        """Test embedding function with an invalid input type."""
        invalid_input = 123  # Not an Image or a string
        with self.assertRaises(TypeError):
            embedding(invalid_input)

    def test_empty_text_input(self):
        """Test embedding function with an empty string."""
        text = ""
        result = embedding(text)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_image_embedding_size(self):
        """Test embedding function returns the expected vector size for an image."""
        img = Image.new('RGB', (64, 64), color = 'blue')
        result = embedding(img)
        self.assertEqual(len(result), 3)

    def test_text_embedding_size(self):
        """Test embedding function returns the expected vector size for a string."""
        text = "OpenAI"
        result = embedding(text)
        self.assertEqual(len(result), 3)

if __name__ == "__main__":
    unittest.main()




