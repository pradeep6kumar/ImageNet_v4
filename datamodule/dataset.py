#!/usr/bin/env python3
"""
Dataset and DataLoader classes for ImageNet-1K using FFCV
"""
# Standard Library Imports
from typing import Tuple, Optional, Callable, Dict

# Third Party Imports
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToTorchImage, Convert, RandomHorizontalFlip, Resize, CenterCrop
from ffcv.transforms.common import Squeeze
from ffcv.transforms.functional import ToDevice, NormalizeImage


class ImageNetDataset:
    """
    ImageNet-1K dataset using FFCV for faster data loading
    """
    def __init__(
        self,
        beton_path: str,
        mode: str = 'train',
        device: str = 'cuda',
        batch_size: int = 128,
        num_workers: int = 8,
        transforms: Optional[dict] = None,
        shuffle_buffer: int = 1024 * 8,
    ) -> None:
        """
        Initialize ImageNet dataset
        
        :param beton_path: Path to beton file
        :param mode: Either 'train' or 'val'
        :param device: Device to load data to ('cuda' or 'cpu')
        :param batch_size: Batch size for the loader
        :param num_workers: Number of workers for loading
        :param transforms: Custom transforms for data
        """
        self.mode = mode
        
        # Set default transforms if none provided
        if transforms is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            
            image_pipeline = [
                SimpleRGBImageDecoder(),
                Resize(256),
                CenterCrop(224),
                RandomHorizontalFlip() if mode == 'train' else Operation(),
                ToTensor(),
                ToDevice(device, non_blocking=True),
                NormalizeImage(mean, std, np_dtype=torch.float32),
                ToTorchImage(torch_dtype=torch.float16)
            ]
            
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True)
            ]
            
            transforms = {
                'image': image_pipeline,
                'label': label_pipeline
            }
        
        # Configure sampling strategy
        if mode == 'train':
            # Use a large shuffle buffer to ensure better randomization across classes
            order = OrderOption.QUASI_RANDOM
            self.loader = Loader(
                beton_path,
                batch_size=batch_size,
                num_workers=num_workers,
                order=order,
                pipelines=transforms,
                drop_last=True,
                os_cache=True,
                shuffle_buffer=shuffle_buffer
            )
        else:
            order = OrderOption.SEQUENTIAL
            self.loader = Loader(
                beton_path,
                batch_size=batch_size,
                num_workers=num_workers,
                order=order,
                pipelines=transforms,
                drop_last=False
            )

    def __iter__(self):
        """Return iterator over the dataset"""
        return iter(self.loader)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.loader)
