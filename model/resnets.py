#!/usr/bin/env python3
"""
Script for ResNet50 model
Date: Dec 23, 2024
"""
# Third Party Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports


class ResNet(nn.Module):
    """
    ResNet model with support for regular and bottleneck blocks
    """
    def __init__(
        self,
        num_classes: int = 1000,
        blocks: list[int] = [2, 2, 2, 2],
        block_type: str = 'regular',
        dropout_rate: float = 0.1
    ) -> None:
        """
        Initialize the ResNet model
        :param num_classes: Number of classes to classify
        :param blocks: List of number of blocks in each layer
        :param block_type: Type of residual block ('regular' or 'bottleneck')
        :param dropout_rate: Dropout rate after convolutions
        """
        super().__init__()

        self.blocks = blocks
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        self.block_type = block_type
        self.expansion = 4 if block_type == 'bottleneck' else 1

        # Initial layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, blocks[0], is_initial_block=False)
        self.layer2 = self._make_layer(128, blocks[1], is_initial_block=True)
        self.layer3 = self._make_layer(256, blocks[2], is_initial_block=True)
        self.layer4 = self._make_layer(512, blocks[3], is_initial_block=True)

        # Final layers
        final_channels = 512 * self.expansion
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, is_initial_block: bool) -> nn.Sequential:
        """
        Create a layer of ResNet blocks
        :param out_channels: Number of output channels for the layer
        :param blocks: Number of blocks in the layer
        :param is_initial_block: Whether the first block is an initial block
        """
        layers = []
        layers.append(
            self.resnet_block(
                is_initial_block=is_initial_block,
                in_channels=self.in_channels,
                out_channels=out_channels
            )
        )
        self.in_channels = out_channels * self.expansion

        for _ in range(1, blocks):
            layers.append(
                self.resnet_block(
                    is_initial_block=False,
                    in_channels=self.in_channels,
                    out_channels=out_channels
                )
            )

        return nn.Sequential(*layers)

    def resnet_block(self, is_initial_block: bool, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create either regular or bottleneck block based on block_type
        :param is_initial_block: Whether the first block is an initial block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        if self.block_type == 'bottleneck':
            return self._bottleneck_block(is_initial_block, in_channels, out_channels)
        return self._regular_block(is_initial_block, in_channels, out_channels)

    def _regular_block(self, is_initial_block: bool, in_channels: int, out_channels: int) -> nn.Module:
        """
        Regular ResNet block with two 3x3 convolutions
        :param is_initial_block: Whether the first block is an initial block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        initial_conv_stride = 2 if is_initial_block else 1

        # Main branch
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=initial_conv_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=self.dropout_rate)
        )

        # Shortcut connection
        shortcut = nn.Sequential()
        if is_initial_block or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_conv_stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(p=self.dropout_rate)
            )

        return ResNetBlock(layers, shortcut)

    def _bottleneck_block(self, is_initial_block: bool, in_channels: int, out_channels: int) -> nn.Module:
        """
        Bottleneck block with 1x1, 3x3, 1x1 convolutions
        :param is_initial_block: Whether the first block is an initial block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        initial_conv_stride = 2 if is_initial_block else 1
        expanded_channels = out_channels * self.expansion

        # Main branch
        layers = nn.Sequential(
            # 1x1 conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),

            # 3x3 conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=initial_conv_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),

            # 1x1 conv for expanding dimensions
            nn.Conv2d(out_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.Dropout2d(p=self.dropout_rate)
        )

        # Shortcut connection
        downsample = nn.Sequential()
        if is_initial_block or in_channels != expanded_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=initial_conv_stride, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.Dropout2d(p=self.dropout_rate)
            )

        return BottleneckBlock(layers, downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model
        :param x: Input tensor
        """
        # Initial layers
        x = self.relu(self.bn_init(self.conv_init(x)))
        x = self.dropout(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BottleneckBlock(nn.Module):
    """
    Basic ResNet block with skip connection
    """
    def __init__(self, layers: nn.Sequential, downsample: nn.Sequential):
        """
        Initialize the ResNet block
        :param layers: Main layers of the block
        :param downsample: Shortcut connection of the block
        """
        super().__init__()
        self.conv1 = layers[0]
        self.bn1 = layers[1]
        self.conv2 = layers[3]
        self.bn2 = layers[4]
        self.conv3 = layers[6]
        self.bn3 = layers[7]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet block
        :param x: Input tensor
        """
        identity = x
        
        # Match PyTorch's forward pass structure
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------- ResNet Variants ----------------------
class ResNet50(ResNet):
    """
    ResNet50 model
    """
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.1) -> None:
        """
        Initialize the ResNet50 model
        :param num_classes: Number of classes to classify
        """
        super().__init__(
            num_classes=num_classes,
            blocks=[3, 4, 6, 3],
            block_type='bottleneck',
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet50 model
        :param x: Input tensor
        """
        super().forward(x)


class ResNet18(ResNet):
    """
    ResNet18 model
    """
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.1) -> None:
        """
        Initialize the ResNet18 model
        :param num_classes: Number of classes to classify
        """
        super().__init__(
            num_classes=num_classes,
            blocks=[2, 2, 2, 2],
            block_type='regular',
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet18 model
        :param x: Input tensor
        """
        super().forward(x)


class ResNet34(ResNet):
    """
    ResNet34 model
    """
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.1) -> None:
        """
        Initialize the ResNet34 model
        :param num_classes: Number of classes to classify
        """
        super().__init__(
            num_classes=num_classes,
            blocks=[3, 4, 6, 3],
            block_type='regular',
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet34 model
        :param x: Input tensor
        """
        super().forward(x)


class ResNet101(ResNet):
    """
    ResNet101 model
    """
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.1) -> None:
        """
        Initialize the ResNet101 model
        :param num_classes: Number of classes to classify
        """
        super().__init__(
            num_classes=num_classes,
            blocks=[3, 4, 23, 3],
            block_type='bottleneck',
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet101 model
        :param x: Input tensor
        """
        super().forward(x)


class ResNet152(ResNet):
    """
    ResNet152 model
    """
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.1) -> None:
        """
        Initialize the ResNet152 model
        :param num_classes: Number of classes to classify
        """
        super().__init__(
            num_classes=num_classes,
            blocks=[3, 8, 36, 3],
            block_type='bottleneck',
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet152 model
        :param x: Input tensor
        """
        super().forward(x)
