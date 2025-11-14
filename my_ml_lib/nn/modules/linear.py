import numpy as np
from .base import Module
from my_ml_lib.nn.autograd import Value

class Linear(Module):
    """Applies a linear transformation to the incoming data: y = X @ W + b"""

    def __init__(self, in_features, out_features, bias=True):
        """Initializes the Linear layer."""
        # --- [Boilerplate init code: 1935-1957] ---
        super().__init__() # CRITICAL
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameter
        scale = np.sqrt(2.0 / in_features)
        # Note: Boilerplate shape is (in_features, out_features)
        self.weight = Value(
            scale * np.random.randn(in_features, out_features), 
            label='weight'
        )

        # Initialize bias parameter
        if bias:
            # Bias b has shape (out_features,)
            self.bias = Value(np.zeros(out_features), label='bias')
        else:
            self.register_parameter('bias', None)

    def __call__(self, x: Value) -> Value:
        """Defines the forward pass of the Linear layer."""
        # --- Implement the forward pass ---
        # x shape: (batch_size, in_features)
        # W shape: (in_features, out_features)
        # b shape: (out_features,)

        # --- Step 1: Matrix Multiplication ---
        # out = x @ self.weight
        # (batch_size, in_features) @ (in_features, out_features) -> (batch_size, out_features)
        out = x @ self.weight

        # --- Step 2: Add Bias (Optional) ---
        if self.bias is not None:
            # out = out + self.bias
            # (batch_size, out_features) + (out_features,) -> (batch_size, out_features)
            out = out + self.bias

        return out

    def __repr__(self):
        """Provides a developer-friendly string representation."""
        has_bias = self._parameters.get('bias', None) is not None
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={has_bias})"