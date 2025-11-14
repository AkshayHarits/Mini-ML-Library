import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes (GNB) classifier.
    Assumes features are conditionally independent given the class,
    and each feature follows a Gaussian distribution within each class.
    """
    def __init__(self, var_smoothing=1e-9):
        self.classes_ = None
        self.class_priors_ = None # P(y=k)
        self.theta_ = None # Mean of each feature per class (n_classes, n_features)
        self.var_ = None   # Variance of each feature per class (n_classes, n_features)
        # self.epsilon = 1e-9 # Renamed to var_smoothing for clarity
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y."""
        n_samples, n_features = X.shape
        
        # 1. Find unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize storage
        self.class_priors_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            n_k = X_k.shape[0]
            
            # 2. Calculate class priors: P(y=k) = Nk / N
            self.class_priors_[i] = n_k / n_samples
            
            # 3. Calculate mean (self.theta_) and variance (self.var_)
            self.theta_[i, :] = np.mean(X_k, axis=0)
            self.var_[i, :] = np.var(X_k, axis=0)

        # Add smoothing to variance
        # Note: This is different from A2's epsilon addition.
        # Standard GNB adds smoothing relative to the max variance.
        # We will just add the small epsilon as per the boilerplate hint.
        self.var_ += self.var_smoothing 

        return self

    def _gaussian_log_pdf(self, X, class_idx):
        """Calculate log probability density function for Gaussian."""
        mean = self.theta_[class_idx]
        var = self.var_[class_idx]
        
        # log P(x_j | y=k) = -0.5 * log(2*pi*var_j) - 0.5 * ((x_j - mean_j)^2 / var_j)
        
        # Calculate for all features at once using broadcasting
        # X shape: (n_samples, n_features)
        # mean/var shape: (n_features,)
        
        log_prob_const = -0.5 * np.log(2. * np.pi * var)
        log_prob_exp = -0.5 * ((X - mean) ** 2) / var
        
        # Sum log probabilities across features
        # This gives log P(x | y=k) for each sample
        return np.sum(log_prob_const + log_prob_exp, axis=1) # Shape (n_samples,)

    def predict_log_proba(self, X):
        """Calculate log probability estimates for samples in X."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")

        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))

        for i, k in enumerate(self.classes_):
            # log P(y=k|x) propto log P(y=k) + log P(x|y=k)
            # log P(x|y=k) = sum over features j [ log P(x_j|y=k) ]
            
            class_conditional_log_prob = self._gaussian_log_pdf(X, i)
            
            joint_log_likelihood[:, i] = np.log(self.class_priors_[i]) + class_conditional_log_prob

        return joint_log_likelihood

    def _stable_softmax(self, X):
        """Compute softmax values for each sets of scores in X."""
        max_log_proba = np.max(X, axis=1, keepdims=True)
        e_x = np.exp(X - max_log_proba)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def predict_proba(self, X):
        """Calculate probability estimates for samples in X."""
        log_proba = self.predict_log_proba(X)
        # Convert log probabilities to probabilities using stable softmax
        return self._stable_softmax(log_proba)

    def predict(self, X):
        """Perform classification on samples in X."""
        log_proba = self.predict_log_proba(X)
        # Find the class with the highest log probability
        indices = np.argmax(log_proba, axis=1)
        return self.classes_[indices]