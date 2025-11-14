import numpy as np

class LogisticRegression:
    """L2-Regularized Logistic Regression classifier.
    
    This version is trained using Stochastic Gradient Descent (SGD).
    """
    
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-5, fit_intercept=True, 
                 learning_rate=0.01, batch_size=64, random_state=None):
        """
        Args:
            alpha (float): L2 regularization strength.
            max_iter (int): Maximum number of epochs for SGD.
            tol (float): Tolerance for stopping criterion (not used in this SGD impl.).
            fit_intercept (bool): Whether to add a bias term.
            learning_rate (float): Learning rate for SGD.
            batch_size (int): Size of mini-batches for SGD.
            random_state (int): Seed for shuffling data in SGD.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol # Note: tol is not used by this SGD implementation
        self.fit_intercept = fit_intercept
        self.w_ = None # Learned weights
        
        # --- New parameters required for SGD ---
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        # Clip z to avoid overflow/underflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

#     # --- START of COMMENTED-OUT IRLS METHOD ---
#     # This was the original Newton's method (IRLS), which can be slow.
#     #
#     def fit(self, X, y):
#         """Fit the L2-regularized logistic regression model using IRLS.
# 
#         Args:
#             X (np.ndarray): Training data, shape (n_samples, n_features).
#             y (np.ndarray): Target values (0 or 1), shape (n_samples,).
#         """
#         n_samples, n_features = X.shape
#         w_old = None # Keep track of previous weights for convergence check
# 
#         # --- Step 1 Add intercept term (bias) ---
#         if self.fit_intercept:
#             # Augment X with a column of ones
#             X_aug = np.hstack([np.ones((n_samples, 1)), X])
#             # Initialize self.w as a zero vector of size (n_features + 1)
#             self.w_ = np.zeros(n_features + 1)
#         else:
#             # Use X directly as X_aug
#             X_aug = X
#             # Initialize self.w as a zero vector of size n_features
#             self.w_ = np.zeros(n_features)
# 
#         # --- Step 2 Regularization setup for IRLS ---
#         # Create the regularization matrix: alpha * Identity
#         reg_matrix = self.alpha * np.eye(self.w_.shape[0])
#         
#         # If fitting an intercept, make sure the first element is 0
#         if self.fit_intercept:
#             reg_matrix[0, 0] = 0.0
# 
#         # --- Step 3 IRLS Iterations ---
#         for i in range(self.max_iter):
#             w_old = self.w_.copy() # Store weights from previous iteration
# 
#             # --- Step 3a Calculate predictions (h) ---
#             z = X_aug @ self.w_
#             h = self._sigmoid(z)
# 
#             # --- Step 3b Calculate gradient (grad_L) ---
#             reg_grad = self.alpha * self.w_
#             if self.fit_intercept:
#                 reg_grad[0] = 0.0
#             gradient = X_aug.T @ (h - y) + reg_grad
# 
#             # --- Step 3c Calculate weight matrix R (diagonal) ---
#             r_diag = h * (1 - h)
#             r_diag = np.maximum(r_diag, 1e-10)
# 
#             # --- Step 3d Calculate Hessian (H) ---
#             hessian = (X_aug.T * r_diag) @ X_aug + reg_matrix
# 
#             # --- Step 3e Update weights ---
#             try:
#                 delta_w = np.linalg.solve(hessian, gradient)
#             except np.linalg.LinAlgError:
#                 print(f"Warning: Hessian is singular at iter {i}. Using pseudo-inverse.")
#                 delta_w = np.linalg.pinv(hessian) @ gradient
# 
#             self.w_ = w_old - delta_w
# 
#             # --- Step 3f Check for convergence ---
#             weight_change = np.linalg.norm(self.w_ - w_old)
#             if weight_change < self.tol:
#                 break
#         else:
#             print(f"Warning: IRLS did not converge within {self.max_iter} iterations.")
#             
#         return self
#     # --- END of COMMENTED-OUT IRLS METHOD ---


    def fit(self, X, y):
        """Fit the L2-regularized logistic regression model using Mini-Batch SGD.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values (0 or 1), shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # --- Step 1: Add intercept term (bias) ---
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            self.w_ = np.zeros(n_features + 1)
        else:
            X_aug = X
            self.w_ = np.zeros(n_features)
            
        # For reproducibility of batches
        rng = np.random.RandomState(self.random_state)
        
        # --- Step 2: SGD Iterations (Epochs) ---
        for epoch in range(self.max_iter):
            # Shuffle data for each epoch
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            X_aug_shuffled = X_aug[indices]
            y_shuffled = y[indices]
            
            # --- Loop over mini-batches ---
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_aug_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                if X_batch.shape[0] == 0:
                    continue
                
                # --- Step 2a: Calculate predictions (h) ---
                z = X_batch @ self.w_
                h = self._sigmoid(z)
                
                # --- Step 2b: Calculate gradient (grad_L) ---
                # Gradient of log-loss: (1/m) * X.T @ (h - y)
                # Gradient of L2-reg: alpha * w (with w[0] = 0)
                
                reg_grad = self.alpha * self.w_
                if self.fit_intercept:
                    reg_grad[0] = 0.0 # Don't regularize bias
                
                # Average gradient over the batch
                gradient = (1.0 / X_batch.shape[0]) * (X_batch.T @ (h - y_batch)) + reg_grad
                
                # --- Step 2c: Update weights ---
                self.w_ = self.w_ - self.learning_rate * gradient
                
        return self


    def predict_proba(self, X):
        """Predict class probabilities for samples in X.
        (This method works for both IRLS and SGD fitted weights)
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- Step 1 Augment X if fitting intercept ---
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X

        # --- Step 2 Calculate P(y=1|X) ---
        z = X_aug @ self.w_
        prob_y1 = self._sigmoid(z)

        # --- Step 3 Calculate P(y=0|X) ---
        prob_y0 = 1.0 - prob_y1

        # --- Step 4 Stack probabilities ---
        return np.column_stack([prob_y0, prob_y1])

    def predict(self, X):
        """Predict class labels (0 or 1) for samples in X.
        (This method works for both IRLS and SGD fitted weights)
        """
        # --- Step 1 Get P(y=1|X) ---
        probabilities_y1 = self.predict_proba(X)[:, 1]

        # --- Step 2 Apply threshold ---
        return (probabilities_y1 >= 0.5).astype(int)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)