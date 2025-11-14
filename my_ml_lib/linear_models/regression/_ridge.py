import numpy as np

class RidgeRegression:
    """Linear least squares with L2 regularization (Ridge Regression).
    Minimizes objective function: ||y - Xw||^2 + alpha * ||w||^2
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        Args:
            alpha (float): Regularization strength; must be a positive float.
                         Larger values specify stronger regularization.
            fit_intercept (bool): Whether to calculate the intercept
                                  for this model. If set to False,
                                  no intercept will be used.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None  # Weights for the features
        self.intercept_ = None # Intercept (bias term)

    def fit(self, X, y):
        """Fit Ridge regression model.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # --- Step 1: Handle fit_intercept ---
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            n_features_total = n_features + 1
        else:
            X_aug = X
            n_features_total = n_features

        # --- Step 2 & 3: Create regularization matrix ---
        # Create the identity matrix I
        reg_matrix = self.alpha * np.eye(n_features_total)
        
        # Important: Do not regularize the intercept term
        if self.fit_intercept:
            reg_matrix[0, 0] = 0.0

        # --- Step 4: Solve the normal equation ---
        # w = (X^T X + alpha*I)^(-1) @ X^T @ y
        try:
            A = X_aug.T @ X_aug + reg_matrix
            b = X_aug.T @ y
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Warning: Solving normal equation failed. Using pseudo-inverse.")
            A_inv = np.linalg.pinv(A)
            weights = A_inv @ b

        # --- Store coefficients ---
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights

        return self

    def predict(self, X):
        """Predict using the linear model.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values, shape (n_samples,).
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- Implement prediction: y_pred = X @ coef_ + intercept_ ---
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred