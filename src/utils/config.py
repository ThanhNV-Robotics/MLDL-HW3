"""
Configuration utilities for MLDL-HW3
"""

import yaml
import json
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config


def save_config(config, config_path):
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the configuration file
        
    Raises:
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
