"""
CIFAR-10 Dataset Module

This module provides functionality to download, parse, and load the CIFAR-10 dataset.
"""

from .cifar10_loader import (
    load_cifar10,
    download_cifar10,
    CIFAR10_CLASSES,
    flatten,
)

__all__ = [
    'load_cifar10',
    'download_cifar10',
    'CIFAR10_CLASSES',
    'flatten',
]
