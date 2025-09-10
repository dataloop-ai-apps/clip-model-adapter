import dtlpy as dl
from model_adapter import ClipAdapter
import pytest
import torch
import time


class TestClipDevices:
    """Test class for CLIP model device handling (CPU/CUDA)"""
    
    @pytest.fixture(scope="class")
    def test_items(self):
        """Setup test items"""
        image_item = dl.items.get(item_id='68bff4534315a5d87edd13f8')
        text_item = dl.items.get(item_id='68c08fb4e2076842ab8d1e39')
        return image_item, text_item

    def test_cpu_device(self, test_items):
        """Test CLIP adapter on CPU device"""
        # Setup adapter
        project = dl.projects.get(project_id="cc3c2a13-8e87-426a-ac6c-e8587be95cf6")
        adapter = ClipAdapter(model_entity=dl.models.get(model_id='68c00043055605761751cf1e'))
        adapter.load('./')
        
        # Force CPU device
        adapter.device = torch.device("cpu")
        adapter.model.to(adapter.device)
        
        # Verify device is set to CPU
        assert adapter.device.type == 'cpu', "Device should be set to CPU"
        
        # Test embedding on CPU with timing
        image_item, text_item = test_items
        batch = [image_item, text_item]
        
        # Time the embedding operation
        start_time = time.time()
        result = adapter.embed(batch)
        end_time = time.time()
        
        cpu_time = end_time - start_time
        
        # Print timing results
        print(f"\nCPU Embedding Results:")
        print(f"  Device: {adapter.device}")
        print(f"  Time: {cpu_time:.3f} seconds")
        print(f"  Items processed: {len(batch)}")
        print(f"  Time per item: {cpu_time/len(batch):.3f} seconds")
        
        # Verify results
        assert len(result) == 2, "Should return 2 embeddings"
        assert all(emb is not None for emb in result), "All embeddings should not be None"
        assert all(isinstance(emb, list) for emb in result), "All embeddings should be lists"
        assert all(len(emb) == 512 for emb in result), "All embeddings should have 512 dimensions"
        
        # Verify model is on CPU
        for param in adapter.model.parameters():
            assert param.device.type == 'cpu', f"Model parameter should be on CPU, but is on {param.device}"

    def test_cuda_device(self, test_items):
        """Test CLIP adapter on CUDA device (if available)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available, skipping CUDA test")
        
        # Setup adapter
        project = dl.projects.get(project_id="cc3c2a13-8e87-426a-ac6c-e8587be95cf6")
        adapter = ClipAdapter(model_entity=dl.models.get(model_id='68c00043055605761751cf1e'))
        adapter.load('./')
        
        # Force CUDA device
        adapter.device = torch.device("cuda:0")
        adapter.model.to(adapter.device)
        
        # Verify device is set to CUDA
        assert adapter.device.type == 'cuda', "Device should be set to CUDA"
        assert adapter.device.index == 0, "Should use CUDA device 0"
        
        # Test embedding on CUDA with timing
        image_item, text_item = test_items
        batch = [image_item, text_item]
        
        # Clear CUDA cache and synchronize before timing
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Time the embedding operation
        start_time = time.time()
        result = adapter.embed(batch)
        torch.cuda.synchronize()  # Ensure CUDA operations are complete
        end_time = time.time()
        
        cuda_time = end_time - start_time
        
        # Print timing results
        print(f"\nCUDA Embedding Results:")
        print(f"  Device: {adapter.device}")
        print(f"  Time: {cuda_time:.3f} seconds")
        print(f"  Items processed: {len(batch)}")
        print(f"  Time per item: {cuda_time/len(batch):.3f} seconds")
        print(f"  CUDA Memory Allocated: {torch.cuda.memory_allocated(adapter.device) / (1024**2):.2f} MB")
        print(f"  CUDA Memory Cached: {torch.cuda.memory_reserved(adapter.device) / (1024**2):.2f} MB")
        
        # Verify results
        assert len(result) == 2, "Should return 2 embeddings"
        assert all(emb is not None for emb in result), "All embeddings should not be None"
        assert all(isinstance(emb, list) for emb in result), "All embeddings should be lists"
        assert all(len(emb) == 512 for emb in result), "All embeddings should have 512 dimensions"
        
        # Verify model is on CUDA
        for param in adapter.model.parameters():
            assert param.device.type == 'cuda', f"Model parameter should be on CUDA, but is on {param.device}"
            assert param.device.index == 0, f"Model parameter should be on CUDA:0, but is on {param.device}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
