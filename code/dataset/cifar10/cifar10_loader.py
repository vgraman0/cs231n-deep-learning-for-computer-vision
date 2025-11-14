"""
CIFAR-10 Dataset Loader

This module provides functionality to download, parse, and load the CIFAR-10 dataset.
CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
There are 50,000 training images and 10,000 test images.
"""

import os
import pickle
import tarfile
import urllib.request
import numpy as np
from pathlib import Path


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def download_cifar10(data_dir=None):
    """
    Download and extract CIFAR-10 dataset if it doesn't already exist.
    
    Args:
        data_dir (str): Directory to store the dataset. 
                       Default is 'data' relative to this module's location.
    
    Returns:
        str: Path to the extracted CIFAR-10 data directory.
    """
    # Default to 'data' directory relative to this module
    if data_dir is None:
        module_dir = Path(__file__).parent
        data_dir = module_dir / 'data'
    else:
        data_dir = Path(data_dir)
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cifar_dir = data_dir / 'cifar-10-batches-py'
    
    # Check if data already exists - verify all required batch files are present
    required_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 
                      'data_batch_5', 'test_batch']
    
    if cifar_dir.exists():
        missing_files = [f for f in required_files if not (cifar_dir / f).exists()]
        if not missing_files:
            # All files exist
            print(f"✓ Found existing CIFAR-10 data at: {cifar_dir}")
            print("  Using existing dataset files (no download needed).")
            return str(cifar_dir)
        else:
            # Some files are missing
            print(f"Warning: CIFAR-10 data directory exists but some files are missing: {missing_files}")
            print("  Will download and extract the complete dataset.")
    else:
        print(f"CIFAR-10 data not found at: {cifar_dir}")
        print("  Will download the dataset.")
    
    # Download the dataset
    tar_path = data_dir / 'cifar-10-python.tar.gz'
    print(f"Downloading CIFAR-10 dataset from {url}...")
    
    try:
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete. Extracting...")
    except Exception as e:
        raise RuntimeError(f"Failed to download CIFAR-10 dataset: {e}")
    
    # Extract the tar.gz file
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print(f"Extraction complete. Dataset available at {cifar_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract CIFAR-10 dataset: {e}")
    finally:
        # Clean up the tar file
        if tar_path.exists():
            tar_path.unlink()
    
    return str(cifar_dir)


def unpickle(file):
    """
    Unpickle a CIFAR-10 batch file.
    
    Args:
        file (str): Path to the pickle file.
    
    Returns:
        dict: Dictionary containing 'data', 'labels', 'filenames', and 'batch_label'.
    """
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


def load_cifar10_batch(data_dir, batch_files):
    """
    Load one or more CIFAR-10 batch files.
    
    Args:
        data_dir (str): Directory containing CIFAR-10 batch files.
        batch_files (list): List of batch file names to load.
    
    Returns:
        tuple: (images, labels) as numpy arrays.
    """
    data_dir = Path(data_dir)
    images = []
    labels = []
    
    for batch_file in batch_files:
        batch_path = data_dir / batch_file
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_path}")
        
        batch_dict = unpickle(batch_path)
        
        # CIFAR-10 data is stored as bytes, convert to numpy array
        # Each image is 3072 bytes (32*32*3), stored as a flat array
        batch_data = np.array(batch_dict[b'data'], dtype=np.uint8)
        batch_labels = np.array(batch_dict[b'labels'])
        
        images.append(batch_data)
        labels.append(batch_labels)
    
    # Concatenate all batches
    images = np.vstack(images)
    labels = np.hstack(labels)
    
    return images, labels


def flatten(images):
    """
    Reshape flat image arrays (N, 3072) to image format (N, 32, 32, 3).
    
    Args:
        images (np.ndarray): Images as flat arrays of shape (N, 3072).
    
    Returns:
        np.ndarray: Images reshaped to (N, 32, 32, 3).
    """
    N = images.shape[0]
    # Reshape from (N, 3072) to (N, 32, 32, 3)
    # CIFAR-10 stores images as RRR...GGG...BBB (row-major order)
    images = images.reshape(N, 3, 32, 32)
    # Transpose to (N, 32, 32, 3) - channels last format
    images = images.transpose(0, 2, 3, 1)
    return images


def load_cifar10(data_dir=None, flatten_images=True, normalize=False):
    """
    Load CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to store/load the dataset. 
                       Default is 'data' relative to this module's location.
        flatten_images (bool): If True, return images as flat arrays (N, 3072).
                               If False, return images in image format (N, 32, 32, 3).
                               Default is True (useful for k-NN).
        normalize (bool): If True, normalize pixel values from [0, 255] to [0, 1].
                         Default is False.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) where:
            - X_train: Training images as numpy array
            - y_train: Training labels as numpy array (0-9)
            - X_test: Test images as numpy array
            - y_test: Test labels as numpy array (0-9)
    """
    # Download dataset if needed
    cifar_dir = download_cifar10(data_dir)
    
    # Load training batches
    train_batches = [f'data_batch_{i}' for i in range(1, 6)]
    X_train, y_train = load_cifar10_batch(cifar_dir, train_batches)
    
    # Load test batch
    X_test, y_test = load_cifar10_batch(cifar_dir, ['test_batch'])
    
    # Reshape images if not flattening
    if not flatten_images:
        X_train = flatten(X_train)
        X_test = flatten(X_test)
    
    # Normalize pixel values if requested
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    
    print(f"Loaded CIFAR-10 dataset:")
    print(f"  Training set: {X_train.shape[0]} images, shape {X_train.shape}")
    print(f"  Test set: {X_test.shape[0]} images, shape {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # Example usage
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10(flatten_images=True, normalize=False)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"\nClass names: {CIFAR10_CLASSES}")
    print(f"Label range: {y_train.min()} to {y_train.max()}")


