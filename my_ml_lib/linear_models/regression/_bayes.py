import numpy as np
# May need scipy.stats.multivariate_normal if not implementing PDF from scratch

class BayesianRegression:
    """Bayesian Linear Regression with Gaussian basis functions.
    Assumes a Gaussian likelihood and a Gaussian prior on weights.
    Computes the posterior distribution over weights and the
    posterior predictive distribution. (Based on A2 logic).
    """
    def __init__(self, n_basis=25, basis_sigma_fraction=0.1, alpha=1.0,
                 beta=100.0):
        """
        Args:
            n_basis (int): Number of Gaussian basis functions (including bias).
            basis_sigma_fraction (float): Width sigma as fraction of
                                          center spacing.
            alpha (float): Precision of the Gaussian prior on weights
                           (1/variance).
            beta (float): Precision of the Gaussian likelihood noise
                          (1/variance).
        """
        self.n_basis = n_basis
        self.basis_sigma_fraction = basis_sigma_fraction
        self.alpha = alpha
        self.beta = beta
        self.basis_centers_ = None # mu_j
        self.basis_sigma_ = None # sigma
        self.posterior_mean_ = None # mN
        self.posterior_cov_ = None # SN

    def _gaussian_basis(self, X):
        """Transforms input X using Gaussian basis functions."""
        if self.basis_centers_ is None or self.basis_sigma_ is None:
            raise RuntimeError("Basis functions parameters not set. Call fit first.")
        
        # Ensure X is (n_samples, 1) for broadcasting
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # --- Implement the Gaussian basis transformation from A2 ---
        # phi_j(x) = exp(-(x - mu_j)^2 / (2 * sigma^2))
        
        # Use broadcasting: (n_samples, 1) - (n_basis-1,) -> (n_samples, n_basis-1)
        dist_sq = (X - self.basis_centers_)**2
        phi_no_bias = np.exp(-dist_sq / (2 * self.basis_sigma_**2))
        
        # Handle the bias term (phi_0 = 1).
        # Output should be Phi(X) of shape (n_samples, n_basis)
        phi = np.hstack([np.ones((X.shape[0], 1)), phi_no_bias])
        
        return phi

    def fit(self, X, y):
        """
        Compute the posterior distribution over weights.
        ...
        (Assuming 1D input based on A2 sine example).
        """
        # --- Step 1: Determine basis function centers and width ---
        # Based on A2, centers are spaced evenly over the data range
        # We have n_basis total, so n_basis-1 for non-bias centers
        n_non_bias_centers = self.n_basis - 1
        
        # Space centers evenly from min to max of X
        self.basis_centers_ = np.linspace(np.min(X), np.max(X), n_non_bias_centers)
        
        # Calculate sigma based on center spacing
        if n_non_bias_centers > 1:
            center_spacing = self.basis_centers_[1] - self.basis_centers_[0]
        else:
            center_spacing = np.max(X) - np.min(X) # Default if only one center
            
        self.basis_sigma_ = center_spacing * self.basis_sigma_fraction
        if self.basis_sigma_ == 0:
            self.basis_sigma_ = 0.1 # Avoid division by zero if all X are same

        # --- Step 2: Transform X using basis functions ---
        Phi = self._gaussian_basis(X)

        # --- Step 3: Calculate posterior covariance S_N ---
        # SN_inv = alpha*I + beta * Phi^T @ Phi
        I = np.eye(self.n_basis)
        SN_inv = (self.alpha * I) + (self.beta * Phi.T @ Phi)
        
        # SN = inv(SN_inv)
        self.posterior_cov_ = np.linalg.inv(SN_inv)

        # --- Step 4: Calculate posterior mean m_N ---
        # mN = beta * SN @ Phi^T @ y
        self.posterior_mean_ = self.beta * (self.posterior_cov_ @ Phi.T @ y)
        
        return self

    def predict_dist(self, X):
        """Compute the posterior predictive distribution for new inputs X."""
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- Step 1: Transform X using basis functions ---
        Phi_new = self._gaussian_basis(X)

        # --- Step 2: Calculate predictive mean ---
        # m(x) = Phi_new @ m_N
        pred_mean = Phi_new @ self.posterior_mean_

        # --- Step 3: Calculate predictive variance ---
        # s^2(x) = 1/beta + diag(Phi_new @ SN @ Phi_new^T)
        
        # Calculate the variance term for each sample individually
        # np.diag(A @ B @ A.T) is equivalent to np.sum((A @ B) * A, axis=1)
        variance_from_weights = np.sum((Phi_new @ self.posterior_cov_) * Phi_new, axis=1)
        
        pred_var = (1.0 / self.beta) + variance_from_weights

        return pred_mean, pred_var

    def predict(self, X):
        """
        Predict target values using the mean of the posterior
        predictive distribution.
        """
        pred_mean, _ = self.predict_dist(X)
        return pred_mean