"""
Evaluation script for MLDL-HW3
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.neural_network import NeuralNetwork
from src.utils.logger import setup_logger
from src.utils.training import evaluate


def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
    """
    # Setup logger
    logger = setup_logger('evaluate')
    logger.info("Starting evaluation...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # TODO: Load your test data here
    # For demonstration, we'll create dummy data
    import numpy as np
    features = np.random.randn(200, args.input_dim)
    labels = np.random.randint(0, args.output_dim, 200)
    
    # Create dataset and dataloader directly
    from src.data.dataset import MLDLDataset
    from torch.utils.data import DataLoader
    
    test_dataset = MLDLDataset(features, labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = NeuralNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    
    # Load model weights
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    model.load(str(model_path))
    logger.info(f"Loaded model from: {model_path}")
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MLDL-HW3 model')
    
    # Data parameters
    parser.add_argument('--input-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=10, help='Output dimension (number of classes)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    
    args = parser.parse_args()
    main(args)
