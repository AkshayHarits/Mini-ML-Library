import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """Split X and y arrays into random train and test subsets.

    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        test_size (float or int): Proportion (0.0-1.0) or absolute
                                  number for the test split.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # --- Step 1 Calculate n_test and n_train ---
    if isinstance(test_size, float) and 0.0 < test_size < 1.0:
        n_test = int(np.ceil(n_samples * test_size))
    elif isinstance(test_size, int) and 0 < test_size < n_samples:
        n_test = test_size
    else:
        raise ValueError(f"Invalid test_size: {test_size}. "
                         "Must be float (0, 1) or int (0, n_samples).")
    
    n_train = n_samples - n_test

    # --- Step 2 Create and Shuffle Indices ---
    indices = np.arange(n_samples)
    
    if shuffle:
        # Initialize a RandomState generator
        rng = np.random.RandomState(random_state)
        # Shuffle indices in place
        rng.shuffle(indices)

    # --- Step 3 Split Indices ---
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # --- Step 4 Split Arrays ---
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def train_test_val_split(X, y,
                           train_size=0.7,
                           val_size=0.15,
                           test_size=0.15,
                           shuffle=True,
                           random_state=None):
    """
    Split X and y arrays into random train, validation, and test subsets.
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # --- Step 1 Validate Proportions ---
    if not (0.0 < train_size < 1.0 and 0.0 <= val_size < 1.0 and 0.0 <= test_size < 1.0):
        raise ValueError("Proportions must be between 0.0 and 1.0.")
    
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0.")

    # --- Step 2 Calculate Split Sizes ---
    n_train = int(np.floor(n_samples * train_size))
    n_val = int(np.floor(n_samples * val_size))
    n_test = n_samples - n_train - n_val # Ensure all samples are used
    
    if n_train == 0 or n_test == 0:
        raise ValueError("Train or test split size is zero. Adjust proportions.")
    # n_val can be 0, which is fine.

    # --- Step 3 Create and Shuffle Indices ---
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    # --- Step 4 Split Indices ---
    train_end = n_train
    val_end = n_train + n_val
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:] # Goes to the end

    # --- Step 5 Split Arrays ---
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test