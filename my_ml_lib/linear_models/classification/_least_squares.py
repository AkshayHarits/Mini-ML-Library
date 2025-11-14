import numpy as np

class LeastSquaresClassifier:
    """Classifier using the least squares approach.
    Fits a linear model by minimizing the squared error between
    predictions and target labels (e.g., encoded as +1/-1).
    """
    def __init__(self):
        self.w_ = None # Weight vector (including bias)
        self.classes_ = None # Store class labels

    def fit(self, X, y):
        """Fit the least squares classifier model.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
                            Assumed to be binary (e.g., 0 and 1).
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LeastSquaresClassifier supports only binary classification.")

        # 1. Choose a target encoding T for the labels y (e.g., +1/-1)
        # Assume classes_[1] is the positive class (+1) and classes_[0] is negative (-1)
        T = np.where(y == self.classes_[1], 1, -1)

        # 2. Augment X with a bias column.
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        # 3. Solve for weights using np.linalg.pinv for stability
        # w = pinv(X_aug) @ T
        self.w_ = np.linalg.pinv(X_aug) @ T
        
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # 1. Augment X with a bias column.
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        # 2. Calculate scores: scores = X_augmented @ self.w_
        scores = X_aug @ self.w_

        # 3. Determine predicted class based on scores (sign)
        # 4. Map back to original class labels.
        predictions = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        
        return predictions