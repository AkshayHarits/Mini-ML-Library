from .base import Module
from my_ml_lib.nn.autograd import Value

class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise."""
    def __call__(self, x: Value) -> Value:
        """Forward pass: Applies ReLU activation."""
        # Call the .relu() method on the input Value object
        return x.relu()

    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    """Applies the Sigmoid function element-wise."""
    def __call__(self, x: Value) -> Value:
        """Forward pass: Applies Sigmoid activation."""
        # Call the .sigmoid() method on the input Value object
        # (We added this method to Value in autograd.py)
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"