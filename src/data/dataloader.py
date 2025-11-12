"""
DataLoader utilities for MLDL-HW3
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import MLDLDataset


def create_dataloaders(features, labels, batch_size=32, test_size=0.2, 
                      random_state=42, num_workers=0):
    """
    Create train and test dataloaders from features and labels.
    
    Args:
        features (numpy.ndarray): Feature data
        labels (numpy.ndarray): Label data
        batch_size (int): Batch size for dataloaders
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    # Create datasets
    train_dataset = MLDLDataset(X_train, y_train)
    test_dataset = MLDLDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
