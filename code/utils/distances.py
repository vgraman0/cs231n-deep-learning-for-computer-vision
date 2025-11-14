"""
Distance functions for machine learning algorithms.

This module provides various distance metrics that can be used with
k-nearest neighbors and other algorithms.
"""

import numpy as np
from typing import Callable


def l1_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute L1 (Manhattan) distance between x and each row of y.
    
    Args:
        x: Single sample of shape (D,).
        y: Multiple samples of shape (N, D).
    
    Returns:
        Distances of shape (N,) where each element is the L1 distance
        between x and the corresponding row of y.
    """
    return np.sum(np.abs(x - y), axis=1)


def l2_square_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute squared L2 (Euclidean) distance between x and each row of y.
    
    Args:
        x: Single sample of shape (D,).
        y: Multiple samples of shape (N, D).
    
    Returns:
        Distances of shape (N,) where each element is the squared L2 distance
        between x and the corresponding row of y.
    """
    return np.sum((x - y) ** 2, axis=1)

