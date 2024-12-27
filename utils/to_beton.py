#!/usr/bin/env python3
"""
Script to convert datasets to FFCV's beton format
"""
import os
from typing import Tuple, Optional
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder
import argparse


def convert_to_beton(
    data_dir: str,
    write_path: str,
    max_resolution: int = 256,
    num_workers: int = 8,
    chunk_size: int = 100,
    jpeg_quality: int = 90,
) -> None:
    """
    Convert a dataset to FFCV's beton format
    
    :param data_dir: Path to the dataset directory
    :param write_path: Path where the beton file will be written
    :param max_resolution: Maximum resolution of images
    :param num_workers: Number of workers for parallel processing
    :param chunk_size: Chunk size for writing
    :param jpeg_quality: Quality of JPEG compression
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    
    # Load the dataset
    dataset = ImageFolder(data_dir)
    
    # Configure the fields for the writer
    writer = DatasetWriter(
        write_path,
        {
            'image': RGBImageField(
                max_resolution=max_resolution,
                jpeg_quality=jpeg_quality,
            ),
            'label': IntField(),
        },
        num_workers=num_workers,
    )
    
    # Write the dataset
    writer.from_indexed_dataset(dataset)


def get_args() -> argparse.Namespace:
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Convert dataset to FFCV format')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--write-path', type=str, required=True,
                      help='Path where the beton file will be written')
    parser.add_argument('--max-resolution', type=int, default=256,
                      help='Maximum resolution of images')
    parser.add_argument('--num-workers', type=int, default=8,
                      help='Number of workers for parallel processing')
    parser.add_argument('--chunk-size', type=int, default=100,
                      help='Chunk size for writing')
    parser.add_argument('--jpeg-quality', type=int, default=90,
                      help='Quality of JPEG compression')
    
    return parser.parse_args()


def main() -> None:
    """
    Main function
    """
    args = get_args()
    
    print(f"Converting dataset at {args.data_dir} to beton format...")
    print(f"Writing to {args.write_path}")
    
    convert_to_beton(
        data_dir=args.data_dir,
        write_path=args.write_path,
        max_resolution=args.max_resolution,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        jpeg_quality=args.jpeg_quality,
    )
    
    print("Conversion complete!")


if __name__ == '__main__':
    main() 