import numpy as np

class GaussianBasisFeatures:
    # Note: The init signature in your boilerplate is different from A2.
    # This implementation follows the boilerplate.
    def __init__(self, n_centers=10, sigma=1.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state # For reproducibility if using

    def fit(self, X, y=None):
        """Select n_centers random points from X as RBF centers."""
        # Strategy: Randomly sample n_centers points from X
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(X.shape[0], self.n_centers, replace=False)
        self.centers_ = X[indices]
        return self

    def transform(self, X):
        """
        Transform input X into Gaussian RBF features.
        Output shape: (n_samples, n_centers)
        """
        if self.centers_ is None:
            raise RuntimeError("Transformer is not fitted yet.")

        # Apply the RBF formula: exp(-(||X - center||^2 / (2 * sigma^2)))

        # Compute squared Euclidean distance between each sample and each center
        # X shape: (n_samples, n_features)
        # centers shape: (n_centers, n_features)
        
        # Use broadcasting to compute squared distances
        # (a-b)^2 = a^2 - 2ab + b^2
        X_sq = np.sum(X**2, axis=1).reshape(-1, 1) # (n_samples, 1)
        centers_sq = np.sum(self.centers_**2, axis=1) # (n_centers,)
        X_dot_centers = X @ self.centers_.T # (n_samples, n_centers)
        
        # dist_sq shape: (n_samples, n_centers)
        dist_sq = X_sq - 2 * X_dot_centers + centers_sq
        
        # Apply Gaussian RBF formula
        return np.exp(-dist_sq / (2 * self.sigma**2))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)