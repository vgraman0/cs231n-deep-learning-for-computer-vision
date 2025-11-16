import numpy as np
from typing import Callable, Tuple

def softmax_loss(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    Softmax loss function.
    
    Args:
        W: Weight matrix of shape (D, C) where D is feature dimension and C is number of classes
        X: Input data of shape (N, D) where N is number of samples
        y: Labels of shape (N,) containing class indices
        reg: Regularization strength (float)
    
    Returns:
        Tuple of (loss: float, gradient: np.ndarray of shape (D, C))
    """
    num_train = X.shape[0]

    scores = X @ W # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True) # (N, C)
    log_probs = np.log(probs)

    data_loss = -log_probs[np.arange(num_train), y].sum() / num_train 
    regularization_loss = reg * np.sum(W * W)
    loss = data_loss + regularization_loss

    dscores = probs.copy() # (N, C)
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train + 2 * reg * W
    
    return loss, dW





