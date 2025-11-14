import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self._combinations = None

    def fit(self, X, y=None):
        """
        Compute the combinations of features that will be used.
        """
        n_features = X.shape[1]
        
        # Generate all combinations of feature indices for degrees 0 to self.degree
        start_degree = 0 if self.include_bias else 1
        self._combinations = []
        for d in range(start_degree, self.degree + 1):
            self._combinations.extend(
                combinations_with_replacement(range(n_features), d)
            )
            
        return self

    def transform(self, X):
        """Transform data to polynomial features."""
        if self._combinations is None:
            raise RuntimeError("This PolynomialFeatures instance is not fitted yet.")
        
        n_samples = X.shape[0]
        n_output_features = len(self._combinations)
        
        X_new = np.empty((n_samples, n_output_features))
        
        for i, indices in enumerate(self._combinations):
            if not indices:
                # This is the bias term (degree 0)
                X_new[:, i] = 1.0
            else:
                # Compute the product of features for this combination
                X_new[:, i] = np.prod(X[:, indices], axis=1)
                
        return X_new

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)