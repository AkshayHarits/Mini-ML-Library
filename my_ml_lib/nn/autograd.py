import numpy as np
import math # For exp, log etc.

class Value:
    """
    Stores a scalar or numpy array and its gradient.
    Builds a computation graph for automatic differentiation (backpropagation).
    Inspired by micrograd: https://github.com/karpathy/micrograd
    """

    def __init__(self, data, _parents=(), _op='', label=''):
        """Initializes a Value object."""
        # ... [Boilerplate init code 1112-1126] ...
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(f"Data must be convertible to a numpy array, got {type(data)}")
        
        if not np.issubdtype(data.dtype, np.floating):
            # print(f"Warning: Casting data from {data.dtype} to float64.")
            data = data.astype(np.float64)

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label

    def __repr__(self):
        data_str = f"array(shape={self.data.shape})" if self.data.ndim > 0 else f"scalar({self.data.item():.4f})"
        grad_str = f"array(shape={self.grad.shape})" if self.grad.ndim > 0 else f"scalar({self.grad.item():.4f})"
        return f"Value(data={data_str}, grad={grad_str}, op='{self._op}')"

    def _unbroadcast(self, grad_in, original_shape):
        """Helper to sum gradient back to an original broadcasted shape."""
        if grad_in.shape == original_shape:
            return grad_in
        
        # 1. Sum over new axes
        ndim_diff = grad_in.ndim - len(original_shape)
        if ndim_diff > 0:
            grad_in = np.sum(grad_in, axis=tuple(range(ndim_diff)))

        # 2. Sum over singleton dimensions
        singleton_dims = tuple(i for i, dim in enumerate(original_shape) if dim == 1)
        if singleton_dims:
            grad_in = np.sum(grad_in, axis=singleton_dims, keepdims=True)

        return grad_in

    def __add__(self, other):
        """Addition operation. Handles broadcasting."""
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')

        def _backward():
            # Gradient of '+' is 1. Chain rule: self.grad += 1 * out.grad
            # Handle broadcasting
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Multiplication operation. Handles broadcasting."""
        other = other if isinstance(other, Value) else Value(other)

        # --- Step 1 Forward Pass ---
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')

        # --- Step 2 Define _backward for Multiplication ---
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            # Chain rule: self.grad += other.data * out.grad
            
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            
            # Handle broadcasting
            self.grad += self._unbroadcast(grad_self, self.data.shape)
            other.grad += self._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    # --- Commutative operations ---
    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    # --- Other necessary math operations ---
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        return self * (other**-1)

    def __rtruediv__(self, other): # other / self
        return other * (self**-1)

    def __pow__(self, other):
        """Power operation (only supports scalar power)."""
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        
        # --- Step 1 Forward Pass ---
        out_data = self.data ** other
        out = Value(out_data, (self,), f'**{other}')

        # --- Step 2 Define _backward for Power ---
        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    # --- Activation Functions ---
    def relu(self):
        """Rectified Linear Unit (ReLU) activation."""
        # --- Step 1 Forward Pass ---
        out_data = np.maximum(0, self.data)
        out = Value(out_data, (self,), 'ReLU')

        # --- Step 2 Define _backward for ReLU ---
        def _backward():
            # d(relu(a))/da = 1 if a > 0, else 0
            self.grad += (self.data > 0).astype(self.data.dtype) * out.grad

        out._backward = _backward
        return out

    # --- Elementary Functions (exp, log) ---
    def exp(self):
        """Exponential function."""
        # --- Step 1 Forward Pass ---
        # Clip data for stability
        clipped_data = np.clip(self.data, -500, 700)
        out_data = np.exp(clipped_data)
        out = Value(out_data, (self,), 'exp')

        # --- Step 2 Define _backward for exp ---
        def _backward():
            # d(exp(a))/da = exp(a) = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """Natural logarithm function (log base e)."""
        # --- Step 1 - Forward Pass ---
        # Ensure numerical stability
        self._stable_data_for_log = np.maximum(self.data, 1e-15)
        out_data = np.log(self._stable_data_for_log)
        out = Value(out_data, (self,), 'log')

        # --- Step 2 Define _backward for log ---
        def _backward():
            # d(log(a))/da = 1/a
            # Use the stable data from forward pass
            self.grad += (1.0 / self._stable_data_for_log) * out.grad

        out._backward = _backward
        return out

    # --- Matrix Multiplication ---
    def __matmul__(self, other):
        """Matrix multiplication (@ operator)."""
        other = other if isinstance(other, Value) else Value(other)

        # --- Step 1 - Forward Pass ---
        out_data = self.data @ other.data
        out = Value(out_data, (self, other), '@')

        # --- Step 2 - Define _backward for matmul ---
        def _backward():
            # A @ B = C
            # dL/dA = dL/dC @ B.T
            # dL/dB = A.T @ dL/dC
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            # Note: This simple version doesn't handle batch dimensions > 2D

        out._backward = _backward
        return out

    # --- Reduction Operations (sum, mean) ---
    def sum(self, axis=None, keepdims=False):
        """Summation operation."""
        # --- Step 1 Forward Pass ---
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'sum')

        # --- Step 2 Define _backward for sum ---
        def _backward():
            # Gradient of sum is 1, distributed back
            # If keepdims=False, out.grad needs to be reshaped
            grad_out = out.grad
            if not keepdims and axis is not None:
                grad_out = np.expand_dims(out.grad, axis)
            
            self.grad += np.ones_like(self.data) * grad_out

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean operation."""
        # --- Step 1 Forward Pass ---
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'mean')

        # --- Step 2 Define _backward for mean ---
        def _backward():
            # Gradient of mean is 1/N
            # Calculate N
            if axis is None:
                N = self.data.size
            else:
                N = self.data.shape[axis]
            
            grad_out = out.grad
            if not keepdims and axis is not None:
                grad_out = np.expand_dims(out.grad, axis)
                
            self.grad += (np.ones_like(self.data) * grad_out) / N

        out._backward = _backward
        return out

    # --- BACKPROPAGATION ---
    def backward(self):
        """Performs backpropagation starting from this Value node."""
        # --- Step 1 Topological Sort ---
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        
        build_topo(self)

        # --- Step 2 Initialize Gradient ---
        # Initialize gradient of the final node to ones
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # --- Step 3 Backward Pass ---
        # Iterate through the topo list in reverse order
        for node in reversed(topo):
            node._backward()

    # --- Add sigmoid (from activations) here for BCE loss ---
    def sigmoid(self):
        """Sigmoid activation function."""
        # sigmoid(x) = 1 / (1 + exp(-x))
        # We can write this using our existing ops
        return (1.0 + (-self).exp())**-1