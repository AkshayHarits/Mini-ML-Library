import numpy as np
import pandas as pd
import os # Useful for joining paths
class DatasetNotFoundError(Exception):
    """Custom exception for when a dataset file is not found."""
    pass
def load_spambase(data_folder="data", filename="spambase.data",
                  download_url=None):
    """Loads the UCI Spambase dataset from a local file.

    Args:
        data_folder (str): The folder where dataset files are stored.
        filename (str): The name of the spambase data file.
        download_url (str, optional): URL to download from if file not
    found. (Implementation of download is
    optional).

    Returns:
        tuple: (X, y) numpy arrays, features and labels.

    Raises:
        DatasetNotFoundError: If the dataset file cannot be found.
    """
    file_path = os.path.join(data_folder, filename)

    if not os.path.exists(file_path):
        # Optional: Add code here to download from download_url if provided
        raise DatasetNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please download it from the UCI ML Repository and place it in the '{data_folder}' directory."
        )

    # The spambase data has no header and is comma-separated
    # Using pandas is robust for CSVs
    try:
        data = pd.read_csv(file_path, header=None).values
    except Exception as e:
        raise IOError(f"Error loading data file {file_path}. Error: {e}")

    # All columns except the last one are features
    X = data[:, :-1]
    # The last column is the label
    y = data[:, -1]
    
    return X, y

def load_fashion_mnist(data_folder="data",
                       train_filename="fashion-mnist_train.csv",
                       test_filename="fashion-mnist_test.csv",
                       kind='train', normalize=True):
    """
    Loads the Fashion-MNIST dataset from local CSV files.

    Args:
        data_folder (str): Folder where dataset CSV files are stored.
        train_filename (str): Name of the training CSV file.
        test_filename (str): Name of the testing CSV file.
        kind (str): 'train' or 'test' to specify which dataset to load.
        normalize (bool): If True, scale pixel values from 0-255 to 0-1.

    Returns:
        tuple: (X, y) numpy arrays, features (images flattened) and labels.

    Raises:
        DatasetNotFoundError: If the specified dataset file cannot be found.
        ValueError: If kind is not 'train' or 'test'.
    """
    if kind == 'train':
        filename = train_filename
    elif kind == 'test':
        filename = test_filename
    else:
        raise ValueError("kind must be 'train' or 'test'")

    file_path = os.path.join(data_folder, filename)

    if not os.path.exists(file_path):
        raise DatasetNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please download the Fashion MNIST CSV files from Kaggle and place them in the '{data_folder}' directory."
        )

    # Load data using pandas
    try:
        data_df = pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error loading data file {file_path}. Error: {e}")

    # The first column is 'label'
    y = data_df['label'].values
    # The rest are pixel values
    X = data_df.drop('label', axis=1).values

    # Normalize the data if normalize is True (pixel values range from 0 to 255)
    if normalize:
        X = X.astype(np.float64) / 255.0

    return X, y