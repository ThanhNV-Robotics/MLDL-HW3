"""
Neural Network model implementation for MLDL-HW3
"""

import torch
import torch.nn as nn
from .base_model import BaseModel


class NeuralNetwork(BaseModel):
    """
    Simple feedforward neural network model.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension
            dropout_rate (float): Dropout rate for regularization
        """
        super(NeuralNetwork, self).__init__(input_dim, hidden_dim, output_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
