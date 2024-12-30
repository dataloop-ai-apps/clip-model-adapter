import os
import unittest
import numpy as np
from types import NoneType

from PIL import Image
from pathlib import Path

from model_adapter import ClipAdapter

PATH = Path(__file__).resolve().parent / "tests" / "assets" / "unittest"

class TestRunner(unittest.TestCase):
    def setUp(self):
        self.adapter = ClipAdapter()
        self.adapter.load('./')

    def test_image_embedding(self):
        """Test embedding function with an Image object."""
        img = Image.new('RGB', (64, 64), color = 'red')
        img_array = np.array(img)
        result = self.adapter.embed([img_array])
        self.assertIsInstance(result[0], list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result[0]))
        self.assertEqual(len(result[0]), 512)

    def test_text_embedding(self):
        """Test embedding function with a text string."""
        text = "Hello, world!"
        result = self.adapter.embed([text])
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result[0]))
        self.assertEqual(len(result[0]), 512)

    def test_unsupported_type(self):
        """Test embedding function with an unsupported input type."""
        invalid_input = 123  # Not an Image or a string
        result = self.adapter.embed([invalid_input])
        self.assertTrue(x.isEmpty() for x in result[0])

if __name__ == "__main__":
    unittest.main()
