import numpy as np

class KFold:
    """
    K-Folds cross-validator.
    ...
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        """Initializes the KFold splitter."""
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer greater than 1.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean value.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        ...
        Yields:
            tuple: (train_indices, test_indices) arrays for each split.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # --- Implement shuffling logic ---
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        # --- Implement the logic to calculate fold sizes ---
        
        # Determine fold sizes, distributing remainder samples across first folds
        base_fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits
        
        # Create an array of fold sizes
        fold_sizes = np.full(self.n_splits, base_fold_size)
        fold_sizes[:remainder] += 1

        current = 0
        # yield to make it a generator
        for fold_size in fold_sizes:
            # Define start and end indices for the current test fold
            start = current
            stop = current + fold_size
            
            # Select test indices for this fold
            test_indices = indices[start:stop]
            
            # Select train indices (all indices *except* the test ones)
            # Create a mask for test indices
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_indices] = True
            
            train_indices = indices[~test_mask]
            
            yield train_indices, test_indices
            
            # Move to the start of the next fold
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits