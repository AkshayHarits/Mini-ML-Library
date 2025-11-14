# Import key components from submodules for easier access like nn.Linear
from .autograd import Value
from .modules import Module, Linear, ReLU, Sigmoid, Sequential
from . import optim
from . import losses
# CHANGE HERE: Import the class directly
from .losses import CrossEntropyLoss, BinaryCrossEntropyLoss
from .optim import SGD # Also import SGD directly

__all__ = [
    'Value',
    'Module',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Sequential',
    'optim',
    'losses',
    'CrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'SGD'
]