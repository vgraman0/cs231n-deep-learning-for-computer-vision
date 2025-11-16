"""
Optimizer implementations from scratch.

This module contains implementations of common first-order optimization algorithms:
- SGD (Stochastic Gradient Descent)
- SGD with Momentum
- RMSProp
- Adam

All optimizers inherit from the abstract Optimizer base class and implement the step method.
"""

import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Abstract base class for optimizers.
    
    All optimizers must implement the step method which updates parameters
    based on the gradient.
    """
    
    @abstractmethod
    def step(self, theta, grad, state, t):
        """Perform one optimization step.
        
        Parameters
        ----------
        theta : ndarray
            Current parameter vector.
        grad : ndarray
            Gradient at current point.
        state : dict
            State dictionary for storing optimizer-specific information.
        t : int
            Current iteration number (starting from 1).
            
        Returns
        -------
        new_theta : ndarray
            Updated parameter vector.
        state : dict
            Updated state dictionary.
        """
        pass


class SGD(Optimizer):
    """Plain stochastic gradient descent.
    
    Parameters
    ----------
    lr : float, default=0.1
        Learning rate.
    """
    
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def step(self, theta, grad, state, t):
        """Perform SGD update: theta = theta - lr * grad.
        
        Parameters
        ----------
        theta : ndarray, shape (2,)
            Current parameter vector.
        grad : ndarray, shape (2,)
            Gradient at current point.
        state : dict
            Unused, but kept for API compatibility.
        t : int
            Current iteration number (starting from 1).
            
        Returns
        -------
        new_theta : ndarray, shape (2,)
            Updated parameter vector.
        state : dict
            State dictionary (unchanged).
        """
        new_theta = theta - self.lr * grad
        return new_theta, state


class Momentum(Optimizer):
    """SGD with momentum.
    
    Parameters
    ----------
    lr : float, default=0.1
        Learning rate.
    beta : float, default=0.9
        Momentum coefficient.
    """
    
    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
    
    def step(self, theta, grad, state, t):
        """Perform momentum update.
        
        Update rule:
        - v = beta * v + (1 - beta) * grad
        - theta = theta - lr * v
        
        Parameters
        ----------
        theta : ndarray, shape (2,)
            Current parameter vector.
        grad : ndarray, shape (2,)
            Gradient at current point.
        state : dict
            State dictionary. Should contain:
            * state["v"] : velocity vector, same shape as theta
        t : int
            Current iteration number (starting from 1).
            
        Returns
        -------
        new_theta : ndarray, shape (2,)
            Updated parameter vector.
        state : dict
            Updated state dictionary with new velocity.
        """
        if "v" not in state:
            state["v"] = np.zeros_like(theta)
        
        v = state["v"]
        
        # Update velocity: v = beta * v + (1 - beta) * grad
        v = self.beta * v + (1 - self.beta) * grad
        # Update parameters: theta = theta - lr * v
        theta = theta - self.lr * v
        state["v"] = v
        
        return theta, state


class RMSProp(Optimizer):
    """RMSProp optimizer.
    
    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    rho : float, default=0.9
        Decay rate for moving average of squared gradients.
    eps : float, default=1e-8
        Small constant for numerical stability.
    """
    
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
    
    def step(self, theta, grad, state, t):
        """Perform RMSProp update.
        
        Update rule:
        - s = rho * s + (1 - rho) * grad^2
        - theta = theta - lr * grad / (sqrt(s) + eps)
        
        Parameters
        ----------
        theta : ndarray, shape (2,)
            Current parameter vector.
        grad : ndarray, shape (2,)
            Gradient at current point.
        state : dict
            State dictionary. Should contain:
            * state["s"] : running average of squared gradients
        t : int
            Current iteration number (starting from 1).
            
        Returns
        -------
        new_theta : ndarray, shape (2,)
            Updated parameter vector.
        state : dict
            Updated state dictionary with new squared gradient average.
        """
        if "s" not in state:
            state["s"] = np.zeros_like(theta)
        
        s = state["s"]
        
        # Update running average of squared gradients
        s = self.rho * s + (1 - self.rho) * (grad ** 2)
        # Update parameters with adaptive learning rate
        theta = theta - self.lr * grad / (s ** 0.5 + self.eps)
        state["s"] = s
        
        return theta, state


class Adam(Optimizer):
    """Adam optimizer with bias correction.
    
    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    eps : float, default=1e-8
        Small constant for numerical stability.
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def step(self, theta, grad, state, t):
        """Perform Adam update with bias correction.
        
        Update rule:
        - m = beta1 * m + (1 - beta1) * grad
        - v = beta2 * v + (1 - beta2) * grad^2
        - m_hat = m / (1 - beta1^t)
        - v_hat = v / (1 - beta2^t)
        - theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        
        Parameters
        ----------
        theta : ndarray, shape (2,)
            Current parameter vector.
        grad : ndarray, shape (2,)
            Gradient at current point.
        state : dict
            State dictionary. Should contain:
            * state["m"] : first moment (mean of gradients)
            * state["v"] : second moment (mean of squared gradients)
        t : int
            Current iteration number (starting from 1).
            
        Returns
        -------
        new_theta : ndarray, shape (2,)
            Updated parameter vector.
        state : dict
            Updated state dictionary with new moments.
        """
        if "m" not in state:
            state["m"] = np.zeros_like(theta)
        if "v" not in state:
            state["v"] = np.zeros_like(theta)
        
        m = state["m"]
        v = state["v"]
        
        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * grad
        # Update biased second moment estimate
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        
        # Bias correction
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        
        # Update parameters
        theta = theta - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        
        state["m"] = m
        state["v"] = v
        
        return theta, state


# Backward compatibility: function-based API
# These functions create optimizer instances and call their step methods
def sgd_step(theta, grad, state, t, lr=0.1):
    """Plain stochastic gradient descent (function-based API for backward compatibility)."""
    optimizer = SGD(lr=lr)
    return optimizer.step(theta, grad, state, t)


def momentum_step(theta, grad, state, t, lr=0.1, beta=0.9):
    """SGD with momentum (function-based API for backward compatibility)."""
    optimizer = Momentum(lr=lr, beta=beta)
    return optimizer.step(theta, grad, state, t)


def rmsprop_step(theta, grad, state, t, lr=0.01, rho=0.9, eps=1e-8):
    """RMSProp optimizer (function-based API for backward compatibility)."""
    optimizer = RMSProp(lr=lr, rho=rho, eps=eps)
    return optimizer.step(theta, grad, state, t)


def adam_step(theta, grad, state, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer (function-based API for backward compatibility)."""
    optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    return optimizer.step(theta, grad, state, t)
