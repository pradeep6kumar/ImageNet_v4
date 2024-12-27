#!/usr/bin/env python3
"""
DataLoader for ImageNet-1K dataset
"""
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Optional
from datamodule.dataset import ImageNetDataset


class ImageNetDataLoader(DataLoader):
    """
    DataLoader for ImageNet-1K dataset using FFCV
    """
    def __init__(
        self,
        beton_path: str,
        mode: str = 'train',
        batch_size: int = 128,
        num_workers: int = 8,
        device: str = 'cuda',
        transforms: Optional[dict] = None,
    ) -> None:
        """
        Initialize ImageNet DataLoader
        
        :param beton_path: Path to beton file
        :param mode: Either 'train' or 'val'
        :param batch_size: Number of samples per batch
        :param num_workers: Number of workers for data loading
        :param device: Device to load data to
        :param transforms: Custom transforms pipeline
        """
        self.dataset = ImageNetDataset(
            beton_path=beton_path,
            mode=mode,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=transforms
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


"""
# Example usage:
if __name__ == '__main__':

    # Create training and validation loaders
    train_loader = ImageNetDataLoader(
        beton_path='path/to/train.beton',
        mode='train',
        batch_size=128,
        num_workers=8
    )

    val_loader = ImageNetDataLoader(
        beton_path='path/to/val.beton',
        mode='val',
        batch_size=128,
        num_workers=8
    )

    # Training loop remains the same
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Training step
            pass
"""
