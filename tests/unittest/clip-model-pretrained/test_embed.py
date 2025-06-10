import unittest
import numpy as np

from io import BytesIO
from PIL import Image
from model_adapter import ClipAdapter


class MockItem:
    def __init__(self, mimetype):
        self.mimetype = mimetype
        self.name = f'Mock item {self.mimetype}'
        self.id = 1

    def download(self, save_locally=False, to_array=True):
        if self.mimetype == 'image/jpeg':
            item_obj = Image.new('RGB', (64, 64), color='red')
            if to_array is True:
                item_obj = np.array(item_obj)
        elif self.mimetype == 'text/plain':
            item_obj = BytesIO(b"s3://ssml-prd-ext-staging/imerit/images/pcs/1565618/44479741640/frames/frame_000056.jpgs3://ssml-prd-ext-staging/imerit/images/pcs/1565618/44479741640/frames/frame_000254.jpg")
        else:
            item_obj = float()
        return item_obj


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.adapter = ClipAdapter()
        self.adapter.load('./')

    def test_image_embedding(self):
        """Test embedding function with an Image object."""
        item = MockItem(mimetype='image/jpeg')
        result = self.adapter.embed([item])
        self.assertIsInstance(result[0], list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result[0]))
        self.assertEqual(len(result[0]), 512)

    def test_text_embedding(self):
        """Test embedding function with a text string."""
        item = MockItem(mimetype='text/plain')
        result = self.adapter.embed([item])
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in result[0]))
        self.assertEqual(len(result[0]), 512)

    def test_unsupported_type(self):
        """Test embedding function with an unsupported input type."""
        item = MockItem(mimetype='application/pdf')
        result = self.adapter.embed([item])
        self.assertTrue(result[0] is None)


if __name__ == "__main__":
    unittest.main()
