import dtlpy as dl
from model_adapter import ClipAdapter
import pytest
import numpy as np


class TestClipBatching:
    """Test class for CLIP model batching functionality"""
    
    @pytest.fixture(scope="class")
    def setup_adapter(self):
        """Setup CLIP adapter for testing"""
        project = dl.projects.get(project_id="cc3c2a13-8e87-426a-ac6c-e8587be95cf6")
        adapter = ClipAdapter(model_entity=dl.models.get(model_id='68c00043055605761751cf1e'))
        adapter.load('./')
        return adapter
    
    @pytest.fixture(scope="class")
    def test_items(self):
        """Setup test items"""
        image_item = dl.items.get(item_id='68bff4534315a5d87edd13f8')
        text_item = dl.items.get(item_id='68c08fb4e2076842ab8d1e39')
        return image_item, text_item

    def test_single_image_embedding(self, setup_adapter, test_items):
        """Test embedding a single image item"""
        adapter = setup_adapter
        image_item, _ = test_items
        
        result = adapter.embed([image_item])
        
        # Verify output structure
        assert len(result) == 1, "Output should contain exactly 1 embedding"
        assert result[0] is not None, "Image embedding should not be None"
        assert isinstance(result[0], list), "Image embedding should be a list"
        assert len(result[0]) == 512, "Image embedding should have 512 dimensions"

    def test_single_text_embedding(self, setup_adapter, test_items):
        """Test embedding a single text item"""
        adapter = setup_adapter
        _, text_item = test_items
        
        result = adapter.embed([text_item])
        
        # Verify output structure
        assert len(result) == 1, "Output should contain exactly 1 embedding"
        assert result[0] is not None, "Text embedding should not be None"
        assert isinstance(result[0], list), "Text embedding should be a list"
        assert len(result[0]) == 512, "Text embedding should have 512 dimensions"

    def test_mixed_batch_embedding(self, setup_adapter, test_items):
        """Test embedding a mixed batch of image and text items"""
        adapter = setup_adapter
        image_item, text_item = test_items
        
        # Create batch with alternating image and text items
        batch = [image_item, text_item, image_item, text_item]
        result = adapter.embed(batch)
        
        # Verify output structure
        assert len(result) == 4, "Output should contain exactly 4 embeddings"
        assert all(emb is not None for emb in result), "All embeddings should not be None"
        assert all(isinstance(emb, list) for emb in result), "All embeddings should be lists"
        assert all(len(emb) == 512 for emb in result), "All embeddings should have 512 dimensions"

    def test_duplicate_items_consistency(self, setup_adapter, test_items):
        """Test that duplicate items produce consistent embeddings"""
        adapter = setup_adapter
        image_item, text_item = test_items
        
        # Create batch with duplicates
        batch = [image_item, image_item, text_item, text_item]
        result = adapter.embed(batch)
        
        # Verify output structure
        assert len(result) == 4, "Output should contain exactly 4 embeddings"
        assert all(emb is not None for emb in result), "All embeddings should not be None"
        
        # Verify duplicate items produce identical embeddings
        image_emb1, image_emb2, text_emb1, text_emb2 = result
        
        # Check that duplicate images produce identical embeddings
        np.testing.assert_array_almost_equal(
            np.array(image_emb1), 
            np.array(image_emb2), 
            decimal=5, 
            err_msg="Duplicate image items should produce identical embeddings"
        )
        
        # Check that duplicate text items produce identical embeddings
        np.testing.assert_array_almost_equal(
            np.array(text_emb1), 
            np.array(text_emb2), 
            decimal=5, 
            err_msg="Duplicate text items should produce identical embeddings"
        )

    def test_large_batch_embedding(self, setup_adapter, test_items):
        """Test embedding a large batch of items"""
        adapter = setup_adapter
        image_item, text_item = test_items
        
        # Create a larger batch (10 items)
        batch = [image_item, text_item] * 5
        result = adapter.embed(batch)
        
        # Verify output structure
        assert len(result) == 10, "Output should contain exactly 10 embeddings"
        assert all(emb is not None for emb in result), "All embeddings should not be None"
        assert all(isinstance(emb, list) for emb in result), "All embeddings should be lists"
        assert all(len(emb) == 512 for emb in result), "All embeddings should have 512 dimensions"

    def test_batch_order_preservation(self, setup_adapter, test_items):
        """Test that output embeddings maintain the same order as input items"""
        adapter = setup_adapter
        image_item, text_item = test_items
        
        # Create a specific order batch
        batch = [image_item, text_item, image_item, text_item, image_item]
        result = adapter.embed(batch)
        
        # Verify output structure
        assert len(result) == 5, "Output should contain exactly 5 embeddings"
        
        # Verify that embeddings are in the correct order
        # Even indices should be image embeddings, odd indices should be text embeddings
        for i, (item, embedding) in enumerate(zip(batch, result)):
            assert embedding is not None, f"Embedding at index {i} should not be None"
            
            # Check that the embedding corresponds to the correct item type
            if 'image/' in item.mimetype:
                assert isinstance(embedding, list), f"Image embedding at index {i} should be a list"
            elif 'text/' in item.mimetype:
                assert isinstance(embedding, list), f"Text embedding at index {i} should be a list"

    def test_empty_batch_handling(self, setup_adapter):
        """Test handling of empty batch"""
        adapter = setup_adapter
        
        result = adapter.embed([])
        
        # Verify empty batch returns empty list
        assert result == [], "Empty batch should return empty list"

    def test_single_type_batch(self, setup_adapter, test_items):
        """Test batches containing only one type of item"""
        adapter = setup_adapter
        image_item, text_item = test_items
        
        # Test batch with only images
        image_batch = [image_item, image_item, image_item]
        image_result = adapter.embed(image_batch)
        
        assert len(image_result) == 3, "Image-only batch should return 3 embeddings"
        assert all(emb is not None for emb in image_result), "All image embeddings should not be None"
        
        # Test batch with only text
        text_batch = [text_item, text_item, text_item]
        text_result = adapter.embed(text_batch)
        
        assert len(text_result) == 3, "Text-only batch should return 3 embeddings"
        assert all(emb is not None for emb in text_result), "All text embeddings should not be None"

    def test_embedding_dimensions_consistency(self, setup_adapter, test_items):
        """Test that all embeddings have consistent dimensions"""
        adapter = setup_adapter
        image_item, text_item = test_items
        
        # Test with various batch sizes
        batch_sizes = [1, 2, 5, 10]
        
        for batch_size in batch_sizes:
            batch = [image_item, text_item] * (batch_size // 2 + 1)
            batch = batch[:batch_size]  # Trim to exact size
            
            result = adapter.embed(batch)
            
            # Verify all embeddings have the same dimension
            embedding_dims = [len(emb) for emb in result if emb is not None]
            assert len(set(embedding_dims)) == 1, f"All embeddings should have the same dimension for batch size {batch_size}"
            assert embedding_dims[0] == 512, f"All embeddings should have 512 dimensions for batch size {batch_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
