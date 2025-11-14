# FOML Assignment 3: The "Mini-ML" Library and Autograd Engine

**Name:** Akshay S Harits

---

## 1. Project Overview

This project, for the "Foundations of Machine Learning" course, involves the creation of a Python machine learning library, `my_ml_lib`, built from scratch using only NumPy.

The project is divided into two main parts:
1.  **Library Implementation:** Building the `my_ml_lib` package, which includes modules for data preprocessing, model selection, linear models, Naive Bayes, and a complete neural network autograd engine (Problem 1, 2, 4).
2.  **Experimental Analysis:** Using this custom library to conduct two major experiments:
    * **Problem 3:** A spam classification task (`spambase.data`) to evaluate the performance and necessity of `StandardScaler` with `LogisticRegression`.
    * **Problem 5:** A "Capstone Showdown" on the Fashion-MNIST dataset to compare four different classification models and identify the best-performing architecture.

This document serves as a complete guide to setting up the project, running all experiments, and understanding the final results and analysis.

---

## 2. `my_ml_lib` Library Overview

The `my_ml_lib` package is the core of this project. Its components are organized as follows:

| Module | Contents | Where It's Used |
| :--- | :--- | :--- |
| **`preprocessing`** | `StandardScaler`, `GaussianBasisFeatures`, `PolynomialFeatures`. | `StandardScaler` is used in `run_spam_experiment.py` and `capstone_showdown.ipynb`. `GaussianBasisFeatures` is used in `capstone_showdown.ipynb` for Model 3. |
| **`datasets`** | `load_spambase`, `load_fashion_mnist`, `_synthetic`. | `load_spambase` is used in `run_spam_experiment.py`. `load_fashion_mnist` is used in `capstone_showdown.ipynb`. |
| **`model_selection`** | `train_test_split`, `KFold`. | `KFold` and `train_test_split` are used in `run_spam_experiment.py`. `train_test_split` is also used in `capstone_showdown.ipynb`. |
| **`linear_models`** | `LogisticRegression` (with `fit_sgd` and `fit_irls`), `RidgeRegression`, `LDA`, `Perceptron`. | The `LogisticRegression` module is the core of `run_spam_experiment.py` and Model 1 in `capstone_showdown.ipynb`. |
| **`naive_bayes`** | `GaussianNaiveBayes`, `BernoulliNaiveBayes`. | (These models are built as part of the library but were not required for the final P3/P5 experiments.) |
| **`nn` (Autograd)** | `Value` (autograd node), `Module` (base class), `Sequential` (container), `Linear` (layer), `ReLU` (activation), `CrossEntropyLoss` (loss), `SGD` (optimizer). | This is the core of P4/P5. `visualize.py` tests the graph. `capstone_showdown.ipynb` uses all components to build, train, and save Models 2, 3, and 4. `create_best_model.py` uses the layers to define the final architecture. |

---

## 3. Setup and Installation

These steps detail how to set up the environment from scratch.

### Step 3.1: Data Population

**IMPORTANT:** For submission, the `data/` folder must be **empty**. To run the scripts, you must download and place the required datasets into the `data/` folder.

* **Spambase (for Problem 3):**
    1.  Download the file from: [https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data)
    2.  Place it in the `data/` folder, ensuring the final path is `data/spambase.data`.

* **Fashion-MNIST (for Problem 5):**
    1.  Go to the Kaggle dataset page: [https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
    2.  Download **only** these two files:
        * `fashion-mnist_train.csv`
        * `fashion-mnist_test.csv`
    3.  Place both files inside the `data/` folder.

### Step 3.2: Environment Setup

1.  Open a terminal (e.g., PowerShell) and navigate to the project root (`FOML_A3_ES23BTECH11002/`).
2.  Create a Python virtual environment:
    ```bash
    python -m venv venv
    ```
3.  Activate the environment:
    ```powershell
    # On PowerShell
    .\venv\Scripts\Activate.ps1
    ```
4.  Install all required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Step 3.3: Install Graphviz (for Problem 4)

The `visualize.py` script requires the Graphviz system library to render graphs.

1.  Go to [https://graphviz.org/download/](https://graphviz.org/download/).
2.  Download and run the installer for your OS.
3.  **CRITICAL:** During installation, you **must** select the option to "Add Graphviz to the system PATH".
4.  **After installation, you must completely restart your terminal and VS Code** for the new PATH to be recognized.

### Step 3.4: Configure VS Code Kernel (for Problem 5)

This project uses a Jupyter Notebook (`.ipynb`). To run this in VS Code using your `venv`:

1.  Open the `FOML_A3_ES23BTECH11002` folder in VS Code.
2.  Open the `capstone_showdown.ipynb` file.
3.  Click the "Select Kernel" button in the top-right corner.
4.  Select your `venv` (e.g., `venv (Python 3.1x.x)`).
5.  VS Code will prompt you to install the `ipykernel` package. Click **Yes** or run `pip install ipykernel` in your terminal. This installs the necessary bridge between VS Code and your environment.

---

## 4. How to Run & Analyze Results

Follow these steps in order to reproduce all project results.

### Step 4.1: Problem 3 (Spam Experiment)

* **How to Run:** In your activated terminal, run:
    ```bash
    python run_spam_experiment.py
    ```
* **What it Does:** This script runs the P3 experiment. It uses 5-fold CV to find the best `alpha` for `LogisticRegression` on both raw and standardized data. It then prints a final summary table and saves the CV plot as `p3_alpha_vs_accuracy.png`.
* **Expected Results:**
    ```
    --- Summary Results ---
    Preprocessing  | Best Alpha | Train Error | Test Error
    :---------------|:-----------|:------------|:-----------
    Raw            | 0.1        | 0.2899      | 0.2932
    Standardized   | 0.001      | 0.0720      | 0.0727
    ```
* **Analysis of Results:**
    The effect of `StandardScaler` is dramatic. The raw data model fails with a **29.32% test error**, while the standardized model succeeds with a **7.27% test error**. The console output (showing `RuntimeWarning: overflow`) proves that the unscaled data is numerically unstable for SGD; the large feature values cause the optimizer to fail. Standardization solves this by creating a stable optimization landscape.

* **Note on Implementation:** The `_logistic.py` file contains two fit methods: `fit_irls()` (from the boilerplate) and `fit_sgd()`. For the large Q5 experiment, `fit_sgd()` is necessary for performance. For consistency, this P3 script also defaults to `fit_sgd()`. The commented-out `fit_irls()` method is available in the library and produces similar (but slower) results, confirming the conclusion.

### Step 4.2: Problem 4 (Visualization)

* **How to Run:** In your terminal, run:
    ```bash
    python visualize.py
    ```
* **What it Does:** This script tests the autograd engine by building a tiny MLP (`Linear(2,3) -> ReLU`), running a backward pass, and saving the computation graph as `computation_graph.png`.
* **Analysis of Result:**
    The `computation_graph.png` image shows the full graph. Rectangles are `Value` nodes (storing `data` and `grad`), and ovals are `op` nodes (like `@`, `+`, `ReLU`). This visual confirms that the `backward()` pass can correctly trace dependencies from the loss back to all parameters.

### Step 4.3: Problem 5 (Capstone Showdown)

* **How to Run:**
    1.  Open `capstone_showdown.ipynb` in VS Code.
    2.  Ensure your `venv` kernel is selected.
    3.  Click the **"Run All" (▶️▶️)** button.
* **What it Does:** This script performs the full P5 analysis. It takes several minutes to:
    1.  Train all four candidate models (OvR Logistic, Softmax, Softmax+RBF, and MLP).
    2.  Generate plots for loss and accuracy (saved in the notebook output).
    3.  Print a final comparison table to find the winner.
    4.  Automatically re-train *only* the winning model and save its final parameters to `saved_models/best_model.npz`.
* **Expected Results:**
    ```
    --- Final Model Comparison ---
    Model                     | Best Validation Accuracy
    :------------------------|:--------------------
    MLP                       | 89.27%
    Softmax (Raw)             | 85.83%
    OvR Logistic (SGD)        | 84.11%
    Softmax (RBF)             | 10.20%
    
    WINNER: MLP (Accuracy: 89.27%)
    ```
* **Analysis of Results:**
    The **MLP is the clear winner**. Its `ReLU` activation functions allow it to learn complex, non-linear patterns in the pixel data that the linear models cannot. The "surprise" of the experiment was the total failure of the `Softmax (RBF)` model (10.20% accuracy), proving that this form of dimensionality reduction was not suitable for this task.

### Step 4.4: Verify Final Submission

* **How to Run:** After running the notebook (Step 4.3), run this in your terminal:
    ```bash
    python verification.py
    ```
* **What it Does:** This is the TA-provided script. It imports `create_best_model.py` to build the empty model architecture and loads the saved weights from `saved_models/best_model.npz` to ensure they match.
* **Expected Result:**
    ```
    --- Verification Summary ---
    PASS Required files found.
    PASS Model architecture initialized.
    PASS Autograd model loaded (check warnings!).
    
    PASS Verification script finished successfully.
    ```
* **Analysis:** This `PASS` message confirms the final submission is valid and loadable.

---

## 5. Best Model Description

The model selected as the "best performing model" based on the validation set accuracy is the **Multi-Layer Perceptron (MLP)**.

* **Best Validation Accuracy:** **89.27%**

This model significantly outperformed all other candidates. Its success is due to the `ReLU` activation functions, which allow the model to learn complex, non-linear patterns in the pixel data that the linear models (OvR, Softmax) cannot.

### Model Architecture

The model is an instance of the `MyMLP` class defined in `create_best_model.py`, which is built using modules from the `my_ml_lib.nn` library. The architecture is a `Sequential` 3-layer feed-forward network:

* **Input Layer:** `Linear(in_features=784, out_features=256)`
* **Activation 1:** `ReLU()`
* **Hidden Layer 1:** `Linear(in_features=256, out_features=128)`
* **Activation 2:** `ReLU()`
* **Output Layer:** `Linear(in_features=128, out_features=10)` (producing raw logits for the 10 classes)

### Training & Hyperparameters

The model was trained on the 50,000-sample scaled training set using the following configuration:

* **Loss Function:** `CrossEntropyLoss` (from `my_ml_lib.nn.losses`)
* **Optimizer:** `SGD` (from `my_ml_lib.nn.optim`)
* **Learning Rate:** `0.05`
* **Batch Size:** `128`
* **Epochs:** `20` (The final saved model, `best_model.npz`, contains the weights from the epoch with the highest validation accuracy.)