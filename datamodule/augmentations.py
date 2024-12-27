#!/usr/bin/env python3
"""
Augmentations to be applied during training on ImageNet-1K training dataset
"""
# Third Party Imports
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from ffcv.transforms import ToTensor, ToTorchImage, Convert, RandomHorizontalFlip
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder, IntDecoder
from ffcv.transforms.common import Squeeze


class ImageNetAugmentations:
    """
    ImageNet Augmentations
    """
    def __init__(self):
        """
        Constructor for ImageNetAugmentations
        """
        # Rotations, Shiftscalerotate, Randombrightness and coardsedropout
        self.train_transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16),
        ])

    def __call__(self, image, label):
        """
        Call method for ImageNetAugmentations
        """
        return self.train_transforms(image=image, label=label)

    def val_transforms(self):
        """
        Validation transforms for ImageNetAugmentations
        """
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
        ])

    def test_transforms(self):
        """
        Test transforms for ImageNetAugmentations
        """
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
        ])

    def get_transforms(self, mode: str) -> dict:
        """
        Get transforms for ImageNetAugmentations
        :param mode: Mode to get transforms for ('train', 'val', 'test')
        """
        if mode == 'train':
            return {
                'image': [
                    RandomResizedCropRGBImageDecoder((224, 224)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    ToTorchImage(),
                    Convert(torch.float32)
                ],
                'label': [
                    IntDecoder(),
                    ToTensor(),
                    Squeeze()
                ]
            }
        else:
            return {
                'image': [
                    CenterCropRGBImageDecoder((224, 224), ratio=224/256),
                    ToTensor(),
                    ToTorchImage(),
                    Convert(torch.float32)
                ],
                'label': [
                    IntDecoder(),
                    ToTensor(),
                    Squeeze()
                ]
            }
