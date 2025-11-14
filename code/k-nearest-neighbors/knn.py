"""
K-Nearest Neighbors (KNN) Classifier

This module implements a K-Nearest Neighbors classifier for image classification.
"""

import numpy as np
from typing import Callable

class KNearestNeighbors:

    
    def __init__(self, distance: Callable, k: int = 1):
        """
        Initialize the KNN classifier.
        
        Args:
            distance: Distance function that computes distances between samples.
                     Should accept two numpy arrays and return distances.
            k (int): Number of nearest neighbors to use for prediction. Default is 1.
        """
        self.distance = distance
        self.k = k

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the KNN classifier by storing the training data.
        
        Note: KNN is a lazy learner, so "training" just means storing the data.
        No actual model fitting happens here.
        
        Args:
            X: Training data of shape (N, D) where N is the number of samples
               and D is the number of features. Each row is a training sample.
               Must be a numpy array.
            y: Training labels of shape (N,) where each element is a class label.
               Must be a numpy array.
        
        Raises:
            ValueError: If k is greater than the number of training samples (N).
        """
        N = X.shape[0]
        if self.k <= 0:
            raise ValueError(f"k must be positive, got k={self.k}")
        if self.k > N:
            raise ValueError(
                f"k ({self.k}) cannot be greater than the number of training samples ({N}). "
                f"Please use 1 <= k <= {N}."
            )
        self.Xtr = X
        self.ytr = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data using k-nearest neighbors.
        
        Args:
            X: Test data of shape (M, D) where M is the number of test samples
               and D is the number of features. Each row is an example we want
               to predict the label for. Must be a numpy array.
        
        Returns:
            Predicted labels of shape (M,) for each test sample as numpy array.
        """
        M = X.shape[0]
        y_pred = np.zeros(M, dtype=self.ytr.dtype)
        
        for i in range(M):
            distances = self.distance(X[i], self.Xtr) # (N,) array of distances
            closest_points_idx = np.argpartition(distances, self.k - 1)[:self.k] # (k, ) array of closest point indices
            labels = self.ytr[closest_points_idx] # (k, ) array of top labels
            y_pred[i] = self.majority(self.k, labels)
        
        return y_pred

    @staticmethod
    def majority(k: int, labels: np.ndarray) -> int:
        """
        Determine the majority class label from k nearest neighbors.
        
        Args:
            k: Number of nearest neighbors.
            labels: Array of labels from the k nearest neighbors, shape (k,).
        
        Returns:
            The majority class label. If k=1, returns that single label.
            If k>1, returns the most frequently occurring label (majority vote).
        """
        if k == 1:
            return labels[0]
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            return unique_labels[np.argmax(counts)]
