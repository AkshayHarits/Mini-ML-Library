import numpy as np

# We must import the exact same modules used by the model
from my_ml_lib.nn import Module, Sequential, Linear, ReLU

#
# This class definition MUST BE IDENTICAL
# to the one used for training in your notebook.
#
class MyMLP(Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        # Define the network architecture
        self.network = Sequential(
            Linear(n_features, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, n_classes)
        )

    def __call__(self, X):
        # Pass input through the network
        return self.network(X)

    def __repr__(self):
        # Delegate the representation to the internal Sequential network
        return f"{self.__class__.__name__}(\n{repr(self.network)}\n)"


def initialize_best_model():
    """
    This function MUST return an instance of your best model's architecture.
    It must match the architecture that was saved in 'best_model.npz'.
    """
    # Fashion-MNIST has 784 features and 10 classes
    N_FEATURES = 784
    N_CLASSES = 10
    
    model = MyMLP(N_FEATURES, N_CLASSES)
    return model