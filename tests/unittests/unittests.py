import json
import pickle
import unittest
import dtlpy as dl

from PIL import Image
from pathlib import Path
from model_adapter import ClipAdapter

TEST_DATA_DIR = Path(__file__).resolve().parent / "assets"

# class TestRunner(unittest.TestCase):
#     def setUp(self):
#         self.assets_path = os.path.join(PATH, 'assets')
#         self.image_features = 'test_img.json'
#         self.image_file = 'test_img.jpg'
#         self.text_features = 'text.json'
#
#         self.adapter = ClipAdapter()
#         self.adapter.load('./')
#
#     def test_embed(self):
#         image = Image.open(os.path.join(self.assets_path, self.image_file))
#         test_embeddings = self.adapter.embed(batch = [image])
#         with open(os.path.join(self.assets_path, f'{self.image_features}'), 'r') as f:
#             ref_embedding = json.load(f)
#         self.assertEqual(ref_embedding, test_embeddings)
#         assert len(test_embeddings[0]) == 512
#
# def test_embed(self):
#     with open('test_img.jpg', 'rb') as f:
#         image = Image.open(f)
#         embeddings = self.adapter.embed([image])
#         assert embeddings is not None
#         assert len(embeddings) == 512


class MockItem:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_name_no_ext = Path(file_name).stem
        self.file_path = TEST_DATA_DIR / file_name

    def download(self, overwrite=True):
        """Mock download method returning the local file path."""
        return str(self.file_path)


class TestAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("dataloop.json", "r") as f:
            config = json.load(f)
        model_json = config.get("components", {}).get("models", [])[0]
        model = dl.Model.from_json(
            _json=model_json,
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package(),
        )
        cls.adapter = ClipAdapter(model_entity=model)

    def load_expected_output(self, file_name):
        """Helper function to load expected output from a pickle file."""
        expected_output_path = TEST_DATA_DIR / file_name
        with open(expected_output_path, "rb") as f:
            return pickle.load(f)

    def test_load(self):
        """Test the adapter's model loading functionality."""
        self.adapter.load(local_path=".")
        expected = self.load_expected_output("test_load_model.pkl")
        self.assertEqual(
            self.adapter.model.names, expected, "Model labels do not match expected labels."
        )

    def test_embed(self):
        """Test the adapter's item embedding."""
        file_names = ["test_img.jpg", "test_text.txt"]
        for file_name in file_names:
            with self.subTest(file_name=file_name):
                mock_item = MockItem(file_name=file_name)
                test_input = self.adapter.prepare_item_func(item=mock_item)
                expected = self.load_expected_output(
                    f"test_prepare_item_func_{mock_item.file_name_no_ext}.pkl"
                )
                self.assertEqual(
                    test_input,
                    expected,
                    f"Prepared input for {mock_item.file_name} does not match expected output.",
                )
        item = MockItem()
        prompt_item = self.adapter.prepare_item_func(item)
        assert prompt_item is not None

    def test_predict(self):
        """Test the adapter's prediction functionality."""
        file_names = ["test_1.jpg"]
        self.adapter.load(local_path=".")
        for file_name in file_names:
            with self.subTest(file_name=file_name):
                mock_item = MockItem()
                image = self.adapter.prepare_item_func(item=mock_item)
                results = self.adapter.predict([image])
                expected = self.load_expected_output(
                    f"test_predict_{mock_item.file_name_no_ext}.pkl"
                )
                self.assertEqual(
                    results,
                    expected,
                    f"Predictions for {mock_item.file_name} do not match expected output.",
                )


if __name__ == '__main__':
    unittest.main()