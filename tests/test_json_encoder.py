"""
Unit tests for JSONEncoder model.

Tests the core JSONEncoder functionality including field-wise embeddings,
multi-categorical field processing, MLP layers, and L2 normalization.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from models.json_encoder import JSONEncoder


class TestJSONEncoder:
    """Test cases for JSONEncoder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vocab_sizes = {
            'category': 10,
            'style': 20,
            'silhouette': 8,
            'material': 15,
            'detail': 25
        }
        
        # 테스트용 작은 차원 사용 (빠른 실행)
        self.encoder = JSONEncoder(
            vocab_sizes=self.vocab_sizes,
            embedding_dim=64,   # 축소된 임베딩 차원
            hidden_dim=128,     # 축소된 은닉층 차원
            output_dim=512,
            dropout_rate=0.1
        )
        
        # Set to eval mode to disable dropout for consistent testing
        self.encoder.eval()
    
    def _create_test_batch(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Create a test batch with proper structure."""
        max_style_len = 3
        max_material_len = 2
        max_detail_len = 4
        
        batch = {
            # Single categorical fields (must be LongTensor for embedding)
            'category': torch.randint(0, self.vocab_sizes['category'], (batch_size,), dtype=torch.long),
            'silhouette': torch.randint(0, self.vocab_sizes['silhouette'], (batch_size,), dtype=torch.long),
            
            # Multi categorical fields (must be LongTensor for embedding)
            'style': torch.randint(0, self.vocab_sizes['style'], (batch_size, max_style_len), dtype=torch.long),
            'material': torch.randint(0, self.vocab_sizes['material'], (batch_size, max_material_len), dtype=torch.long),
            'detail': torch.randint(0, self.vocab_sizes['detail'], (batch_size, max_detail_len), dtype=torch.long),
            
            # Padding masks (1 for valid tokens, 0 for padding)
            'style_mask': torch.ones(batch_size, max_style_len, dtype=torch.float),
            'material_mask': torch.ones(batch_size, max_material_len, dtype=torch.float),
            'detail_mask': torch.ones(batch_size, max_detail_len, dtype=torch.float)
        }
        
        return batch
    
    def test_initialization(self):
        """Test JSONEncoder initialization."""
        assert isinstance(self.encoder.category_emb, nn.Embedding)
        assert isinstance(self.encoder.style_emb, nn.Embedding)
        assert isinstance(self.encoder.silhouette_emb, nn.Embedding)
        assert isinstance(self.encoder.material_emb, nn.Embedding)
        assert isinstance(self.encoder.detail_emb, nn.Embedding)
        
        # Check embedding dimensions
        assert self.encoder.category_emb.num_embeddings == self.vocab_sizes['category']
        assert self.encoder.category_emb.embedding_dim == 64  # 축소된 차원
        
        # Check MLP structure
        assert isinstance(self.encoder.mlp, nn.Sequential)
        assert len(self.encoder.mlp) == 4  # Linear, ReLU, Dropout, Linear
    
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        batch = self._create_test_batch(batch_size=4)
        
        output = self.encoder(batch)
        
        # Should output [batch_size, 512]
        assert output.shape == (4, 512)
        assert output.dtype == torch.float32
    
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            batch = self._create_test_batch(batch_size=batch_size)
            output = self.encoder(batch)
            assert output.shape == (batch_size, 512)
    
    def test_output_normalization(self):
        """Test that output vectors are L2 normalized."""
        batch = self._create_test_batch(batch_size=4)
        
        output = self.encoder(batch)
        
        # Check L2 normalization
        norms = torch.norm(output, p=2, dim=-1)
        expected_norms = torch.ones_like(norms)
        
        # Allow small numerical tolerance
        assert torch.allclose(norms, expected_norms, atol=1e-6)
    
    def test_single_categorical_processing(self):
        """Test processing of single categorical fields."""
        batch_size = 2
        batch = {
            'category': torch.tensor([1, 5], dtype=torch.long),
            'silhouette': torch.tensor([2, 7], dtype=torch.long),
            'style': torch.zeros(batch_size, 1, dtype=torch.long),
            'material': torch.zeros(batch_size, 1, dtype=torch.long),
            'detail': torch.zeros(batch_size, 1, dtype=torch.long),
            'style_mask': torch.ones(batch_size, 1, dtype=torch.float),
            'material_mask': torch.ones(batch_size, 1, dtype=torch.float),
            'detail_mask': torch.ones(batch_size, 1, dtype=torch.float)
        }
        
        output = self.encoder(batch)
        
        # Should produce different outputs for different inputs
        assert not torch.allclose(output[0], output[1])
        assert output.shape == (batch_size, 512)
    
    def test_multi_categorical_processing(self):
        """Test processing of multi-categorical fields with mean pooling."""
        batch_size = 2
        
        # Create batch with different multi-categorical patterns
        batch = {
            'category': torch.zeros(batch_size, dtype=torch.long),
            'silhouette': torch.zeros(batch_size, dtype=torch.long),
            'style': torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),  # Different lengths
            'material': torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            'detail': torch.tensor([[1, 0, 0, 0], [2, 3, 4, 5]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float),  # Mask padding
            'material_mask': torch.ones(batch_size, 2, dtype=torch.float),
            'detail_mask': torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float)
        }
        
        output = self.encoder(batch)
        
        # Should handle different sequence lengths correctly
        assert output.shape == (batch_size, 512)
        assert not torch.allclose(output[0], output[1])
    
    def test_padding_mask_effect(self):
        """Test that padding masks correctly exclude padded tokens."""
        batch_size = 2
        
        # Create identical sequences but with different masks
        batch1 = {
            'category': torch.zeros(batch_size, dtype=torch.long),
            'silhouette': torch.zeros(batch_size, dtype=torch.long),
            'style': torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long),
            'material': torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
            'detail': torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long),
            'style_mask': torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float),  # Different masks
            'material_mask': torch.ones(batch_size, 2, dtype=torch.float),
            'detail_mask': torch.ones(batch_size, 4, dtype=torch.float)
        }
        
        output1 = self.encoder(batch1)
        
        # Should produce different outputs due to different masking
        assert not torch.allclose(output1[0], output1[1])
    
    def test_empty_multi_categorical_handling(self):
        """Test handling of empty multi-categorical fields."""
        batch_size = 1
        
        batch = {
            'category': torch.tensor([1], dtype=torch.long),
            'silhouette': torch.tensor([1], dtype=torch.long),
            'style': torch.zeros(batch_size, 3, dtype=torch.long),
            'material': torch.zeros(batch_size, 2, dtype=torch.long),
            'detail': torch.zeros(batch_size, 4, dtype=torch.long),
            'style_mask': torch.zeros(batch_size, 3, dtype=torch.float),  # All masked (empty)
            'material_mask': torch.zeros(batch_size, 2, dtype=torch.float),  # All masked (empty)
            'detail_mask': torch.zeros(batch_size, 4, dtype=torch.float)  # All masked (empty)
        }
        
        # Should not crash and should produce valid output
        output = self.encoder(batch)
        assert output.shape == (batch_size, 512)
        
        # Check normalization still works
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_process_multi_categorical_method(self):
        """Test the _process_multi_categorical method directly."""
        batch_size = 2
        max_len = 3
        
        # Create test data
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)  # Second sequence shorter
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float)  # Mask padding
        
        # Use style embedding layer for testing
        result = self.encoder._process_multi_categorical(ids, mask, self.encoder.style_emb)
        
        assert result.shape == (batch_size, 64)  # 축소된 embedding_dim
        
        # Results should be different for different inputs
        assert not torch.allclose(result[0], result[1])
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        batch = self._create_test_batch(batch_size=2)
        
        # Enable training mode
        self.encoder.train()
        
        output = self.encoder(batch)
        loss = output.sum()  # Dummy loss
        loss.backward()
        
        # Check that gradients exist for trainable parameters
        for name, param in self.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Zero gradient for parameter: {name}"
    
    def test_deterministic_output(self):
        """Test that the model produces deterministic output in eval mode."""
        batch = self._create_test_batch(batch_size=2)
        
        self.encoder.eval()
        
        # Run forward pass twice
        output1 = self.encoder(batch)
        output2 = self.encoder(batch)
        
        # Should be identical in eval mode
        assert torch.allclose(output1, output2)
    
    def test_vocab_size_validation(self):
        """Test that model handles vocabulary sizes correctly."""
        # Test with minimum vocab sizes
        min_vocab_sizes = {field: 2 for field in self.vocab_sizes.keys()}
        
        encoder = JSONEncoder(vocab_sizes=min_vocab_sizes, embedding_dim=64)
        batch = self._create_test_batch(batch_size=1)
        
        # Adjust batch to use valid indices
        for field in ['category', 'style', 'silhouette', 'material', 'detail']:
            if field in ['category', 'silhouette']:
                batch[field] = torch.zeros_like(batch[field], dtype=torch.long)
            else:
                batch[field] = torch.zeros_like(batch[field], dtype=torch.long)
        
        output = encoder(batch)
        assert output.shape == (1, 512)
    
    def test_embedding_dimensions(self):
        """Test different embedding dimensions."""
        for emb_dim in [64, 128, 256]:
            encoder = JSONEncoder(
                vocab_sizes=self.vocab_sizes,
                embedding_dim=emb_dim,
                hidden_dim=256
            )
            
            batch = self._create_test_batch(batch_size=2)
            output = encoder(batch)
            
            # Output should always be 512-dimensional
            assert output.shape == (2, 512)
    
    def test_hidden_dimensions(self):
        """Test different hidden layer dimensions."""
        for hidden_dim in [128, 256, 512]:
            encoder = JSONEncoder(
                vocab_sizes=self.vocab_sizes,
                embedding_dim=128,
                hidden_dim=hidden_dim
            )
            
            batch = self._create_test_batch(batch_size=2)
            output = encoder(batch)
            
            # Output should always be 512-dimensional
            assert output.shape == (2, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])