"""
Tests for data utilities
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MLDLDataset
from src.data.dataloader import create_dataloaders


class TestMLDLDataset:
    """Test cases for MLDLDataset class"""
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized"""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 5, 100)
        
        dataset = MLDLDataset(features, labels)
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 5, 100)
        
        dataset = MLDLDataset(features, labels)
        feature, label = dataset[0]
        
        assert feature.shape == (10,)
        assert isinstance(label.item(), int)
    
    def test_dataset_length(self):
        """Test dataset length is correct"""
        features = np.random.randn(50, 10)
        labels = np.random.randint(0, 5, 50)
        
        dataset = MLDLDataset(features, labels)
        assert len(dataset) == 50


class TestDataLoader:
    """Test cases for dataloader utilities"""
    
    def test_create_dataloaders(self):
        """Test dataloaders can be created"""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 5, 100)
        
        train_loader, test_loader = create_dataloaders(
            features, labels, batch_size=16, test_size=0.2
        )
        
        assert len(train_loader.dataset) == 80
        assert len(test_loader.dataset) == 20
    
    def test_dataloader_batch_size(self):
        """Test dataloader respects batch size"""
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 5, 100)
        
        batch_size = 16
        train_loader, test_loader = create_dataloaders(
            features, labels, batch_size=batch_size, test_size=0.2
        )
        
        # Get first batch
        for batch_features, batch_labels in train_loader:
            assert batch_features.shape[0] <= batch_size
            assert batch_labels.shape[0] <= batch_size
            break


if __name__ == '__main__':
    pytest.main([__file__])
