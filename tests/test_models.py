"""
Tests for model implementations
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseModel
from src.models.neural_network import NeuralNetwork


class TestBaseModel:
    """Test cases for BaseModel class"""
    
    def test_base_model_initialization(self):
        """Test BaseModel can be initialized with correct dimensions"""
        model = BaseModel(input_dim=10, hidden_dim=20, output_dim=5)
        assert model.input_dim == 10
        assert model.hidden_dim == 20
        assert model.output_dim == 5
    
    def test_base_model_forward_not_implemented(self):
        """Test that BaseModel forward raises NotImplementedError"""
        model = BaseModel(input_dim=10, hidden_dim=20, output_dim=5)
        x = torch.randn(1, 10)
        with pytest.raises(NotImplementedError):
            model(x)


class TestNeuralNetwork:
    """Test cases for NeuralNetwork class"""
    
    def test_neural_network_initialization(self):
        """Test NeuralNetwork can be initialized"""
        model = NeuralNetwork(input_dim=10, hidden_dim=20, output_dim=5)
        assert model.input_dim == 10
        assert model.hidden_dim == 20
        assert model.output_dim == 5
    
    def test_neural_network_forward(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        input_dim = 10
        output_dim = 5
        
        model = NeuralNetwork(input_dim=input_dim, hidden_dim=20, output_dim=output_dim)
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
    
    def test_neural_network_dropout(self):
        """Test dropout is applied during training"""
        model = NeuralNetwork(input_dim=10, hidden_dim=20, output_dim=5, dropout_rate=0.5)
        x = torch.randn(2, 10)
        
        # Training mode
        model.train()
        out1 = model(x)
        out2 = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(out1, out2)
        
        # Eval mode
        model.eval()
        out3 = model(x)
        out4 = model(x)
        
        # Outputs should be the same in eval mode
        assert torch.allclose(out3, out4)
    
    def test_save_load_model(self, tmp_path):
        """Test model can be saved and loaded"""
        model = NeuralNetwork(input_dim=10, hidden_dim=20, output_dim=5)
        model_path = tmp_path / "test_model.pth"
        
        # Save model
        model.save(str(model_path))
        assert model_path.exists()
        
        # Load model
        new_model = NeuralNetwork(input_dim=10, hidden_dim=20, output_dim=5)
        new_model.load(str(model_path))
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__])
