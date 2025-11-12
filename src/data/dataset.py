"""
Dataset class for MLDL-HW3
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MLDLDataset(Dataset):
    """
    Custom Dataset class for MLDL-HW3.
    
    This class wraps data for use with PyTorch DataLoader.
    """
    
    def __init__(self, features, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            features (numpy.ndarray): Feature data
            labels (numpy.ndarray): Label data
            transform (callable, optional): Optional transform to apply to samples
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        """
        Return the size of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (feature, label) pair
        """
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label
