import numpy as np

class BernoulliNaiveBayes:
    """Bernoulli Naive Bayes (BNB) classifier.
    Assumes features are binary (0 or 1), conditionally independent
    given the class, and follow a Bernoulli distribution within each class.
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter
                         (0 for no smoothing). Corresponds to a
                         Beta(alpha, alpha) prior.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None # Log P(y=k)
        self.feature_log_prob_ = None # Log P(x_j=1 | y=k), shape (n_classes, n_features)

    def fit(self, X, y):
        """Fit Bernoulli Naive Bayes according to X, y.
        Assumes X contains binary features (0 or 1).
        """
        n_samples, n_features = X.shape
        
        # 1. Find unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize storage
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, k in enumerate(self.classes_):
            X_k = X[y == k] # Samples belonging to class k
            n_k = X_k.shape[0] # Number of samples in class k (Nk)
            
            # 2. Calculate class log priors: log(Nk / N)
            self.class_log_prior_[i] = np.log(n_k / n_samples)
            
            # 3. For each class k and feature j, estimate P(x_j=1 | y=k)
            # Count occurrences N_kj = count(x_j=1 and y=k)
            n_kj = np.sum(X_k, axis=0) # Sums features (0 or 1)
            
            # Apply Laplace smoothing: P(x_j=1 | y=k) = (N_kj + alpha) / (Nk + 2 * alpha)
            prob = (n_kj + self.alpha) / (n_k + 2 * self.alpha)
            
            # 4. Store the log of these probabilities
            self.feature_log_prob_[i, :] = np.log(prob)
            
        return self

    def predict_log_proba(self, X):
        """Calculate log probability estimates for samples in X."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")
            
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes))

        # We need log P(x_j=0 | y=k) = log(1 - P(x_j=1 | y=k))
        # log(1 - exp(log_p))
        log_prob_x1 = self.feature_log_prob_ # (n_classes, n_features)
        log_prob_x0 = np.log(1.0 - np.exp(log_prob_x1)) # (n_classes, n_features)

        for i, k in enumerate(self.classes_):
            # log P(x|yk) = sum_j [ x_j * log P(x_j=1|yk) + (1-x_j) * log P(x_j=0|yk) ]
            # Use matrix multiplication for efficiency
            # X @ log_prob_x1[i].T + (1-X) @ log_prob_x0[i].T
            log_p_x_given_k = X @ log_prob_x1[i].T + (1 - X) @ log_prob_x0[i].T
            
            # log P(y=k|x) propto log P(y=k) + log P(x|y=k)
            joint_log_likelihood[:, i] = self.class_log_prior_[i] + log_p_x_given_k

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