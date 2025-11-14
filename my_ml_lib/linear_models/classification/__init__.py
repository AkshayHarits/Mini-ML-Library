# Fix: Use relative imports
from ._lda import LinearDiscriminantAnalysis
from ._logistic import LogisticRegression
# Optional imports
from ._least_squares import LeastSquaresClassifier
from ._perceptron import Perceptron

__all__ = [
    'LinearDiscriminantAnalysis',
    'LogisticRegression',
    'LeastSquaresClassifier', # Optional
    'Perceptron'              # Optional
]