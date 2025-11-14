import numpy as np

class Perceptron:
    """Perceptron classifier.
    Simple algorithm for binary linear classification.
    Uses iterative updates based on misclassified points.
    """
    def __init__(self, learning_rate=0.01, max_iters=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.random_state = random_state
        self.w_ = None # Weights (including bias)
        self.errors_ = [] # To store number of misclassifications per epoch
        self.classes_ = None # Store class labels

    def fit(self, X, y):
        """Fit perceptron model using the Pocket Algorithm.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels, e.g., 0/1), 
                            shape (n_samples,).
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1. Initialize weights (e.g., randomly or to zeros).
        w_current = rng.normal(loc=0.0, scale=0.01, size=1 + n_features)
        w_best = w_current.copy() # Pocket: store best weights
        best_errors = n_samples

        # 2. Augment X with bias column.
        X_aug = np.hstack([np.ones((n_samples, 1)), X])

        # 3. Ensure y labels are +1/-1
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Perceptron supports only binary classification.")
        y_conv = np.where(y == self.classes_[1], 1, -1)

        self.errors_ = []

        # 4. Iterate up to max_iters:
        for _ in range(self.max_iters):
            current_errors = 0
            # Loop through shuffled samples
            indices = rng.permutation(n_samples)
            for idx in indices:
                xi = X_aug[idx]
                target = y_conv[idx]
                
                # Make a prediction
                prediction_score = np.dot(xi, w_current)
                
                # Check if misclassified (target * score <= 0)
                if target * prediction_score <= 0:
                    # Update weights
                    update = self.learning_rate * target * xi
                    w_current = w_current + update
            
            # After epoch, evaluate w_current
            predictions = np.sign(X_aug @ w_current)
            # Fix for sign(0) = 0
            predictions[predictions == 0] = -1 
            current_errors = np.sum(predictions != y_conv)
            self.errors_.append(current_errors)

            # Pocket algorithm: check if current weights are better
            if current_errors < best_errors:
                best_errors = current_errors
                w_best = w_current.copy()

            # Stop if perfectly classified
            if best_errors == 0:
                break

        self.w_ = w_best # Store the best weights found
        return self

    def _predict_raw(self, X):
        """Calculate net input (scores)."""
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")
        # Augment X
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_augmented @ self.w_

    def predict(self, X):
        """Return class label after unit step."""
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")
            
        # Apply activation function (step function) to raw predictions
        scores = self._predict_raw(X)
        # Predict +1 or -1
        predictions_conv = np.where(scores >= 0.0, 1, -1)
        
        # Map output to the original class format (e.g., 0/1)
        return np.where(predictions_conv == 1, self.classes_[1], self.classes_[0])