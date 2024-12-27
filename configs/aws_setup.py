#!/usr/bin/env python3
"""
AWS EC2 Setup and Utility Functions for Training
"""
# Standard Library Imports
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

# Third Party Imports
import torch


@dataclass
class EC2Config:
    """
    Configuration for the EC2 instance.
    """
    # g4dn.2xlarge specs
    num_gpus: int = 1
    gpu_memory_gb: int = 16
    num_vcpus: int = 8
    ram_gb: int = 32
    
    # Training configs
    num_workers: int = 4  # Typically num_vcpus/2 for g4dn.2xlarge
    batch_size: int = 128
    gradient_accumulation_steps: int = 4
    
    # Storage paths
    data_path: str = "/data"
    checkpoint_path: str = "/checkpoints"
    
    def __post_init__(self):
        """
        Create directories if they don't exist
        """
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    def get_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers based on CPU cores
        """
        return min(self.num_vcpus - 2, self.num_workers)
    
    def get_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on GPU memory
        """
        # Rough estimation - adjust these values based on your model
        gb_per_image = 0.0125  # Approximate memory per image for ResNet50
        max_images = int((self.gpu_memory_gb * 0.8) / gb_per_image)  # Using 80% of GPU memory
        return min(max_images, self.batch_size)


def setup_gpu_environment():
    """
    Setup CUDA environment variables for optimal performance
    """
    # Enable JIT compilation
    torch.backends.cudnn.benchmark = True
    
    # Set environment variables for better GPU utilization
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'  # Helpful for debugging
    
    # Return GPU info
    return {
        'gpu_count': torch.cuda.device_count(),
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
    }


def monitor_gpu_usage():
    """
    Monitor GPU usage using nvidia-smi
    """
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except Exception as e:
        return f"Error getting GPU stats: {str(e)}"


def calculate_training_params(config: EC2Config):
    """
    Calculate optimal training parameters based on hardware
    """
    batch_size = config.get_optimal_batch_size()
    num_workers = config.get_optimal_workers()
    
    return {
        'batch_size': batch_size,
        'effective_batch_size': batch_size * config.gradient_accumulation_steps,
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': 2,
    }


def setup_training_environment(debug: bool = False):
    """
    Setup complete training environment
    """
    # Initialize config
    config = EC2Config()
    
    # Setup GPU environment
    gpu_info = setup_gpu_environment()
    
    # Get optimal training parameters
    train_params = calculate_training_params(config)
    
    if debug:
        print("GPU Information:")
        print(gpu_info)
        print("\nTraining Parameters:")
        print(train_params)
        print("\nInitial GPU Usage:")
        print(monitor_gpu_usage())
    
    return config, train_params


# Example usage in your training script:
if __name__ == "__main__":
    config, train_params = setup_training_environment(debug=True)