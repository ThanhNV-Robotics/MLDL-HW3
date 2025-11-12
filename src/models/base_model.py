"""
Base model class for MLDL-HW3
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class that all models should inherit from.
    
    This class provides a common interface for all models in the project.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the base model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save(self, path):
        """
        Save model parameters to a file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """
        Load model parameters from a file.
        
        Args:
            path (str): Path to load the model from
        """
        self.load_state_dict(torch.load(path))
