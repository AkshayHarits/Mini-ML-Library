from .base import Module
from collections import OrderedDict
from my_ml_lib.nn.autograd import Value # Import Value for type hinting

class Sequential(Module):
    """A sequential container for stacking Modules."""

    def __init__(self, *args):
        """Initializes the Sequential container."""
        # --- [Boilerplate init code: 1862-1880] ---
        super().__init__() # CRITICAL

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # If an OrderedDict is passed
            for key, module in args[0].items():
                if not isinstance(module, Module):
                    raise TypeError(f"Value for key '{key}' is not an nn.Module subclass: {type(module)}")
                self.add_module(key, module)
        else:
            # If positional arguments are passed
            for i, module in enumerate(args):
                if not isinstance(module, Module):
                    raise TypeError(f"Argument {i} is not an nn.Module subclass: {type(module)}")
                self.add_module(str(i), module)

    def __call__(self, x: Value) -> Value:
        """Defines the forward pass through the sequential layers."""
        # --- Implement the sequential forward pass ---
        # Iterate through the modules stored in self._modules
        for module in self._modules.values():
            # Pass the output of one module as the input to the next
            x = module(x)
        
        return x # Return the output of the final layer

    # --- [Boilerplate String Representation: 1911-1916] ---
    def __repr__(self):
        """Provides a developer-friendly string representation."""
        layer_strs = [f"  ({name}): {module}" for name, module in self._modules.items()]
        return f"Sequential(\n" + "\n".join(layer_strs) + "\n)"