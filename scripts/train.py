"""
Main training script for MLDL-HW3
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.neural_network import NeuralNetwork
from src.data.dataloader import create_dataloaders
from src.utils.logger import setup_logger
from src.utils.training import train_epoch, evaluate


def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Setup logger
    logger = setup_logger('train', args.log_file)
    logger.info("Starting training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # TODO: Load your data here
    # For demonstration, we'll create dummy data
    import numpy as np
    features = np.random.randn(1000, args.input_dim)
    labels = np.random.randint(0, args.output_dim, 1000)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        features, labels,
        batch_size=args.batch_size,
        test_size=args.test_size
    )
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    # Create model
    model = NeuralNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout_rate=args.dropout
    ).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = Path(args.save_dir) / 'best_model.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            logger.info(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLDL-HW3 model')
    
    # Data parameters
    parser.add_argument('--input-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=10, help='Output dimension (number of classes)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Other parameters
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    
    args = parser.parse_args()
    main(args)
