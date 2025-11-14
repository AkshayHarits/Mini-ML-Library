import numpy as np
from .autograd import Value # Need Value to check parameter type

class SGD:
    """Implements stochastic gradient descent."""

    def __init__(self, params, lr=0.01):
        """Initializes the SGD optimizer."""
        # ... [Boilerplate init code 1574-1589] ...
        self.params = list(params)
        self.lr = lr
        for p in self.params:
            if not isinstance(p, Value):
                raise TypeError("Optimizer parameters must be Value objects.")
            if not hasattr(p, 'grad') or p.grad is None:
                p.grad = np.zeros_like(p.data, dtype=np.float64)

    def step(self):
        """Performs a single optimization step (parameter update)."""
        # --- Implement the SGD update rule ---
        for p in self.params:
            if p.grad is not None:
                # Update parameter data in place
                # p.data = p.data - self.lr * p.grad
                p.data -= self.lr * p.grad
            else:
                print("Warning: Parameter has no gradient (p.grad is None).")

    def zero_grad(self):
        """Sets the gradients (.grad attribute) of all managed parameters to zero."""
        # --- Implement gradient zeroing ---
        for p in self.params:
            if isinstance(p, Value):
                # Set gradient to zeros, matching data shape
                p.grad = np.zeros_like(p.data, dtype=np.float64)

    def __repr__(self):
        """Provides a string representation of the optimizer."""
        return f"SGD(lr={self.lr})"