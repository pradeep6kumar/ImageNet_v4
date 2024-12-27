#!/usr/bin/env python3
"""
This file contains the configuration classes for the training, model, and data.
"""
# Standard Library Imports
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    """
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 5.54E-02
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    num_workers: int = 8


@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    num_classes: int = 100
    dropout_rate: float = 0.1


@dataclass
class DataConfig:
    """
    Configuration for the data.
    """
    train_path: str = "train.beton"
    val_path: str = "val.beton"
    input_size: int = 224
