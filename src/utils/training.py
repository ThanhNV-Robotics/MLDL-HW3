"""
Training utilities for MLDL-HW3
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(dataloader, desc='Training'):
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Evaluation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Evaluating'):
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy
