import os
import json
import datetime
import unittest
import dtlpy as dl
from model_adapter import ClipAdapter

PATH = os.path.dirname(os.path.abspath(__file__))

class TestRunner(unittest.TestCase):
    def setUp(self):
        dl.setenv('rc')
        self.assets_path = os.path.join(PATH, 'assets')
        self.project = dl.projects.get('smart image search')
        self.item_id = '676a843467fa453515e6433c'
        self.model_entity = self.project.models.get(model_name='openai-clip')
        self.adapter = ClipAdapter(model_entity=self.model_entity)
        self.adapter.load_from_model()

    def test_embed_items(self):
        self.item = self.project.items.get(item_id=self.item_id)
        _, test_embeddings = self.adapter.embed_items(items=[self.item], upload_features=False)
        with open(os.path.join(self.assets_path, self.model_entity.id, f'{self.item_id}.json'), 'r') as f:
            ref_embedding = json.load(f)
        self.assertEqual(ref_embedding, test_embeddings)
        # assert len(test_embeddings[0]) == 512

if __name__ == '__main__':
    unittest.main()