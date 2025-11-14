import numpy as np
import os
import matplotlib.pyplot as plt  # <--- ADDED THIS IMPORT

# --- Boilerplate: Imports ---
try:
    from my_ml_lib.datasets import load_spambase, DatasetNotFoundError
    from my_ml_lib.preprocessing import StandardScaler
    from my_ml_lib.linear_models.classification import LogisticRegression
    from my_ml_lib.model_selection import KFold, train_test_split
except ImportError as e:
    print(f"Error importing library components: {e}")
    print("Please ensure your my_ml_lib structure and __init__.py files are correct.")
    exit()
# --- End Boilerplate ---

# --- Boilerplate: Configuration ---
DATA_FOLDER = "data/" # Directory containing spambase.data
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS_CV = 5
ALPHAS_TO_TEST = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # L2 regularization strengths to test
# --- End Boilerplate ---

# --- Helper Function for Cross-Validation (MODIFIED) ---
def find_best_alpha(X_train_cv, y_train_cv, alphas, n_splits, random_state):
    """
    Performs K-Fold CV to find the best alpha for Logistic Regression.
    
    MODIFIED: Now returns the list of mean accuracies for plotting.
    """
    
    print(f"Running {n_splits}-fold CV for alphas: {alphas}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    mean_accuracies = []
    
    for alpha in alphas:
        fold_accuracies = []
        for train_index, val_index in kf.split(X_train_cv):
            X_fold_train, X_fold_val = X_train_cv[train_index], X_train_cv[val_index]
            y_fold_train, y_fold_val = y_train_cv[train_index], y_train_cv[val_index]
            
            # --- MODIFIED: Using new SGD constructor ---
            model = LogisticRegression(
                alpha=alpha, 
                max_iter=100, # 100 epochs is enough for CV
                learning_rate=0.01,
                batch_size=128,
                random_state=random_state
            )
            model.fit(X_fold_train, y_fold_train)
            
            acc = model.score(X_fold_val, y_fold_val)
            fold_accuracies.append(acc)
        
        mean_acc = np.mean(fold_accuracies)
        mean_accuracies.append(mean_acc)
        print(f"  Alpha={alpha:<10} | Mean Val Accuracy={mean_acc:.4f}")

    best_alpha_found = alphas[np.argmax(mean_accuracies)]
    print(f"-> Best alpha found: {best_alpha_found}")
    
    # --- MODIFIED: Return mean_accuracies as well ---
    return best_alpha_found, mean_accuracies


def main():
    # --- Step 1: Load Data ---
    print(f"Loading Spambase data from '{DATA_FOLDER}'...")
    try:
        X, y = load_spambase(data_folder=DATA_FOLDER, filename="spambase.data")
        print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
    except DatasetNotFoundError as e:
        print(e)
        print("Exiting.")
        return

    # --- Step 2: Split Data into Train and Test ---
    print(f"Splitting data into {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # --- Step 4: Experiment with RAW Data (MODIFIED) ---
    print("\n--- Experiment: Raw Data ---")
    # --- Capture the returned accuracy list ---
    best_alpha_raw, accs_raw = find_best_alpha(
        X_train, y_train, ALPHAS_TO_TEST, N_SPLITS_CV, RANDOM_STATE
    )

    # --- Step 5: Train and Evaluate Final RAW Model (MODIFIED) ---
    print("Training final raw model on full training set...")
    # --- MODIFIED: Using new SGD constructor ---
    model_raw = LogisticRegression(
        alpha=best_alpha_raw, 
        max_iter=500, # More epochs for final model
        learning_rate=0.01,
        batch_size=128,
        random_state=RANDOM_STATE
    )
    model_raw.fit(X_train, y_train)
    
    acc_train_raw = model_raw.score(X_train, y_train)
    acc_test_raw = model_raw.score(X_test, y_test)
    train_error_raw = 1.0 - acc_train_raw
    test_error_raw = 1.0 - acc_test_raw
    print(f"Raw Model Train Error: {train_error_raw:.4f} (Accuracy: {acc_train_raw:.4f})")
    print(f"Raw Model Test Error:  {test_error_raw:.4f} (Accuracy: {acc_test_raw:.4f})")

    # --- Step 6: Experiment with STANDARDIZED Data (MODIFIED) ---
    print("\n--- Experiment: Standardized Data ---")
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    print("Transforming train and test data...")
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # --- Capture the returned accuracy list ---
    best_alpha_std, accs_std = find_best_alpha(
        X_train_std, y_train, ALPHAS_TO_TEST, N_SPLITS_CV, RANDOM_STATE
    )

    # --- Step 7: Train and Evaluate Final STANDARDIZED Model (MODIFIED) ---
    print("Training final standardized model on full standardized training set...")
    # --- MODIFIED: Using new SGD constructor ---
    model_std = LogisticRegression(
        alpha=best_alpha_std, 
        max_iter=500, # More epochs for final model
        learning_rate=0.01,
        batch_size=128,
        random_state=RANDOM_STATE
    )
    model_std.fit(X_train_std, y_train)
    
    acc_train_std = model_std.score(X_train_std, y_train)
    acc_test_std = model_std.score(X_test_std, y_test)
    train_error_std = 1.0 - acc_train_std
    test_error_std = 1.0 - acc_test_std
    print(f"Std. Model Train Error: {train_error_std:.4f} (Accuracy: {acc_train_std:.4f})")
    print(f"Std. Model Test Error:  {test_error_std:.4f} (Accuracy: {acc_test_std:.4f})")

    # --- [NEW] Step 8: Generate and Save Plot ---
    print("\n--- Generating Alpha vs. Accuracy Plot ---")
    plt.figure(figsize=(10, 6))
    # Plot x-axis on a log scale
    plt.semilogx(ALPHAS_TO_TEST, accs_raw, 'o-', label='Raw Data')
    plt.semilogx(ALPHAS_TO_TEST, accs_std, 's-', label='Standardized Data')
    
    plt.title('Validation Accuracy vs. Alpha (L2 Regularization)')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Mean 5-Fold Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file for your report
    plot_filename = 'p3_alpha_vs_accuracy.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'. Please include this in your report.")
    # plt.show() # You can uncomment this if you want the plot to pop up

    # --- Boilerplate: Report Results ---
    print("\n--- Summary Results ---")
    print(f"Preprocessing  | Best Alpha | Train Error | Test Error")
    print(f":---------------|:-----------|:------------|:-----------")
    print(f"Raw            | {best_alpha_raw:<10} | {train_error_raw:<11.4f} | {test_error_raw:<10.4f}")
    print(f"Standardized   | {best_alpha_std:<10} | {train_error_std:<11.4f} | {test_error_std:<10.4f}")
    print("\nNOTE: Ensure the results above reflect your actual computed values.")


if __name__ == "__main__":
    main()