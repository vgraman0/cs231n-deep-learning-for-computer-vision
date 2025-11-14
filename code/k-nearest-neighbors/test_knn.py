"""
Unit tests for K-Nearest Neighbors classifier.

To run tests, install dependencies first:
    pip install -r requirements.txt
"""

import numpy as np
import pytest  # type: ignore[import-untyped]
import sys
from pathlib import Path

# Add parent directory to path to import from sibling directories
sys.path.insert(0, str(Path(__file__).parent.parent))

from knn import KNearestNeighbors
from utils import l1_distance, l2_square_distance


class TestKNearestNeighbors:
    """Test suite for KNearestNeighbors class."""
    
    def test_initialization(self):
        """Test that KNN initializes correctly."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=5)
        assert knn.k == 5
        assert knn.distance == l2_square_distance
    
    def test_train_stores_data(self):
        """Test that train() stores the training data."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=3)
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.array([0, 1, 0])
        
        knn.train(X_train, y_train)
        
        np.testing.assert_array_equal(knn.Xtr, X_train)
        np.testing.assert_array_equal(knn.ytr, y_train)
    
    def test_predict_k1_single_neighbor(self):
        """Test prediction with k=1 (single nearest neighbor)."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=1)
        
        # Simple 2D training data
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_train = np.array([0, 1, 2])
        knn.train(X_train, y_train)
        
        # Test point closest to first training point
        X_test = np.array([[0.1, 0.1]])
        y_pred = knn.predict(X_test)
        
        assert y_pred[0] == 0
        assert len(y_pred) == 1
    
    def test_predict_k1_multiple_test_points(self):
        """Test prediction with k=1 for multiple test points."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=1)
        
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_train = np.array([0, 1, 2])
        knn.train(X_train, y_train)
        
        X_test = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
        y_pred = knn.predict(X_test)
        
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(y_pred, expected)
    
    def test_predict_k3_majority_vote(self):
        """Test prediction with k=3 using majority voting."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=3)
        
        # Training data with clear clusters
        X_train = np.array([
            [0.0, 0.0],  # class 0
            [0.1, 0.1],  # class 0
            [0.2, 0.2],  # class 0
            [1.0, 1.0],  # class 1
            [1.1, 1.1],  # class 1
        ])
        y_train = np.array([0, 0, 0, 1, 1])
        knn.train(X_train, y_train)
        
        # Test point closer to class 0 cluster
        X_test = np.array([[0.15, 0.15]])
        y_pred = knn.predict(X_test)
        
        # With k=3, should get 3 nearest neighbors (all class 0)
        assert y_pred[0] == 0
    
    def test_predict_k2_tie_breaking(self):
        """Test prediction with k=2 when there's a tie."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=2)
        
        X_train = np.array([
            [0.0, 0.0],  # class 0
            [1.0, 0.0],  # class 1
            [0.0, 1.0],  # class 0
        ])
        y_train = np.array([0, 1, 0])
        knn.train(X_train, y_train)
        
        # Test point equidistant from class 0 and class 1
        X_test = np.array([[0.5, 0.5]])
        y_pred = knn.predict(X_test)
        
        # Should return one of the classes (implementation dependent)
        assert y_pred[0] in [0, 1]
    
    def test_majority_static_method_k1(self):
        """Test majority() static method with k=1."""
        labels = np.array([5])
        result = KNearestNeighbors.majority(1, labels)
        assert result == 5
    
    def test_majority_static_method_k3_unanimous(self):
        """Test majority() static method with k=3, all same label."""
        labels = np.array([2, 2, 2])
        result = KNearestNeighbors.majority(3, labels)
        assert result == 2
    
    def test_majority_static_method_k3_majority(self):
        """Test majority() static method with k=3, majority vote."""
        labels = np.array([1, 1, 0])
        result = KNearestNeighbors.majority(3, labels)
        assert result == 1
    
    def test_majority_static_method_k5_majority(self):
        """Test majority() static method with k=5, clear majority."""
        labels = np.array([0, 0, 0, 1, 2])
        result = KNearestNeighbors.majority(5, labels)
        assert result == 0
    
    def test_different_distance_metrics(self):
        """Test that different distance metrics work correctly."""
        # Test with L1 distance
        knn_l1 = KNearestNeighbors(distance=l1_distance, k=1)
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0, 1])
        knn_l1.train(X_train, y_train)
        
        X_test = np.array([[0.1, 0.1]])
        y_pred_l1 = knn_l1.predict(X_test)
        assert y_pred_l1[0] == 0
        
        # Test with L2 squared distance
        knn_l2 = KNearestNeighbors(distance=l2_square_distance, k=1)
        knn_l2.train(X_train, y_train)
        y_pred_l2 = knn_l2.predict(X_test)
        assert y_pred_l2[0] == 0
    
    def test_custom_distance_function(self):
        """Test with a custom distance function (L3 norm)."""
        def l3_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.sum(np.abs(x - y) ** 3, axis=1)
        
        knn = KNearestNeighbors(distance=l3_distance, k=1)
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0, 1])
        knn.train(X_train, y_train)
        
        X_test = np.array([[0.1, 0.1]])
        y_pred = knn.predict(X_test)
        assert y_pred[0] == 0
    
    def test_empty_training_data_error(self):
        """Test that predict fails gracefully if not trained."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=1)
        X_test = np.array([[1.0, 2.0]])
        
        # Should raise AttributeError when trying to access self.Xtr
        with pytest.raises(AttributeError):
            knn.predict(X_test)
    
    def test_single_training_sample(self):
        """Test prediction with only one training sample."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=1)
        X_train = np.array([[1.0, 2.0]])
        y_train = np.array([5])
        knn.train(X_train, y_train)
        
        X_test = np.array([[1.1, 2.1], [10.0, 20.0]])
        y_pred = knn.predict(X_test)
        
        # Both should predict the same label (only one training sample)
        assert y_pred[0] == 5
        assert y_pred[1] == 5
    
    def test_k_larger_than_training_samples(self):
        """Test that ValueError is raised when k is larger than number of training samples."""
        knn = KNearestNeighbors(distance=l2_square_distance, k=10)
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0, 1])
        
        # Should raise ValueError when k > N
        with pytest.raises(ValueError, match="k.*cannot be greater than"):
            knn.train(X_train, y_train)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

