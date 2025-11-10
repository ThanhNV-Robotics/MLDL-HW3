# MLDL-HW3

Machine Learning and Deep Learning Homework 3 - Source Code Repository

## Project Structure

```
MLDL-HW3/
├── src/                      # Source code directory
│   ├── models/              # Model implementations
│   │   ├── base_model.py   # Base model class
│   │   └── neural_network.py  # Neural network implementation
│   ├── data/                # Data loading and processing
│   │   ├── dataset.py      # Dataset class
│   │   └── dataloader.py   # DataLoader utilities
│   └── utils/               # Utility functions
│       ├── config.py        # Configuration utilities
│       ├── logger.py        # Logging utilities
│       └── training.py      # Training/evaluation utilities
├── scripts/                 # Executable scripts
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── data/                   # Data directory
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── models/                 # Saved models
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ThanhNV-Robotics/MLDL-HW3.git
cd MLDL-HW3
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model with default parameters:
```bash
python scripts/train.py
```

To train with custom parameters:
```bash
python scripts/train.py --epochs 20 --batch-size 64 --lr 0.0001
```

Available training arguments:
- `--input-dim`: Input dimension (default: 784)
- `--output-dim`: Number of output classes (default: 10)
- `--hidden-dim`: Hidden layer dimension (default: 256)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.5)
- `--test-size`: Test set proportion (default: 0.2)
- `--save-dir`: Directory to save models (default: 'models')
- `--log-file`: Path to log file (optional)

### Evaluation

To evaluate a trained model:
```bash
python scripts/evaluate.py --model-path models/best_model.pth
```

### Using Configuration File

You can modify the `config.yaml` file to set default parameters for your experiments.

## Project Components

### Models
- **BaseModel**: Abstract base class for all models
- **NeuralNetwork**: Feedforward neural network implementation

### Data
- **MLDLDataset**: Custom PyTorch Dataset class
- **create_dataloaders**: Utility to create train/test dataloaders

### Utilities
- **Config**: Load/save configuration files
- **Logger**: Setup logging for experiments
- **Training**: Training and evaluation functions

## Development

### Adding a New Model

1. Create a new file in `src/models/`
2. Inherit from `BaseModel`
3. Implement the `forward()` method
4. Update `src/models/__init__.py`

### Adding Tests

Add unit tests in the `tests/` directory:
```bash
pytest tests/
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0

See `requirements.txt` for complete list of dependencies.

## License

This project is for educational purposes as part of MLDL coursework.

## Author

ThanhNV-Robotics