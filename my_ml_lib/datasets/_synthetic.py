import numpy as np

def make_noisy_sine(n_samples=100, noise=0.1, random_state=None):
    """Generates the noisy sine wave dataset from A2.

    Args:
        n_samples (int): Number of data points to generate.
        noise (float): Standard deviation of the Gaussian noise added.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple: (X, y) numpy arrays. X is (n_samples, 1), y is (n_samples,).
    """
    rng = np.random.RandomState(random_state)
    
    # Generate X values uniformly from 0 to 2*pi
    X = rng.uniform(0, 2 * np.pi, size=(n_samples, 1))
    
    # Generate y values as sin(X) + Gaussian noise
    y = np.sin(X).ravel() + rng.normal(0, noise, size=n_samples)
    
    return X, y