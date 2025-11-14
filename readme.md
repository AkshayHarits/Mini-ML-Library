# FOML Assignment 3: The "Mini-ML" Library and Autograd Engine

**Name:** Akshay S Harits 
**Roll No:** ES23BTECH11002

This project involves building a foundational machine learning library, `my_ml_lib`, from scratch using NumPy. It includes modules for data processing, linear models, Naive Bayes, model selection, and a complete autograd engine for neural networks, as required by the assignment.

The library is then used to conduct experiments in spam classification (Problem 3) and to build and evaluate several models for the Fashion-MNIST dataset (Problem 5).

---

## 1. Setup and Installation

These steps detail how to set up the environment from scratch, including the specific steps for VS Code.

### Step 1.1: Environment Setup

Open a terminal (e.g., PowerShell) and navigate to the project root (`Final_BoilerPlate/Final_BoilerPlate/`).

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```powershell
# On PowerShell
.\venv\Scripts\Activate.ps1
```

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 1.2: Install Graphviz (for Problem 4)

The `visualize.py` script requires the Graphviz system library to render graphs.

1. Go to https://graphviz.org/download/
2. Download and run the installer for your OS
3. **CRITICAL:** During installation, you must select the option to "Add Graphviz to the system PATH"
4. After installation, you must completely restart your terminal and VS Code for the new PATH to be recognized

### Step 1.3: Configure VS Code Kernel (for Problem 5)

This project uses a Jupyter Notebook (`.ipynb`). To run this in VS Code using your venv:

1. Open the `Final_BoilerPlate` folder in VS Code
2. Open the `capstone_showdown.ipynb` file
3. Click the "Select Kernel" button in the top-right corner
4. Select your venv (e.g., `venv (Python 3.1x.x)`)
5. VS Code will prompt you to install the `ipykernel` package. Click Yes or run `pip install ipykernel` in your terminal. This installs the necessary bridge between VS Code and your environment.

---

## 2. Data Population

To run the scripts, you must populate it with the required datasets.

### Spambase (for Problem 3):

1.  Download the file from: [https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data)
2.  Place it in the `data/` folder, ensuring the final path is `data/spambase.data`.

### Fashion-MNIST (for Problem 5):

1. Go to the Kaggle dataset page: https://www.kaggle.com/datasets/zalando-research/fashionmnist
2. Download only these two files:
   - `fashion-mnist_train.csv`
   - `fashion-mnist_test.csv`
3. Place both files inside the `data/` folder

---

## 3. my_ml_lib Library Overview

This is a breakdown of the custom library and where each component is used.

| Module | Contents | Where It's Used |
|:-------|:---------|:----------------|
| **preprocessing** | StandardScaler, GaussianBasisFeatures, PolynomialFeatures | StandardScaler is used in `run_spam_experiment.py` and `capstone_showdown.ipynb`. GaussianBasisFeatures is used in `capstone_showdown.ipynb` for Model 3. |
| **datasets** | load_spambase, load_fashion_mnist, _synthetic | load_spambase is used in `run_spam_experiment.py`. load_fashion_mnist is used in `capstone_showdown.ipynb`. |
| **model_selection** | train_test_split, KFold | KFold and train_test_split are used in `run_spam_experiment.py`. train_test_split is also used in `capstone_showdown.ipynb`. |
| **linear_models** | LogisticRegression (with SGD), RidgeRegression, LDA, Perceptron | The SGD-based LogisticRegression is the core of `run_spam_experiment.py` and Model 1 in `capstone_showdown.ipynb`. |
| **naive_bayes** | GaussianNaiveBayes, BernoulliNaiveBayes | (These models are built but were not required for the final P3/P5 experiments.) |
| **nn (Autograd)** | Value (autograd node), Module (base class), Sequential (container), Linear (layer), ReLU (activation), CrossEntropyLoss (loss), SGD (optimizer) | This is the core of P4/P5. `visualize.py` tests the graph. `capstone_showdown.ipynb` uses all components to build, train, and save Models 2, 3, and 4. `create_best_model.py` uses the layers to define the final architecture. |

---

## 4. How to Run & Analyze (Step-by-Step)

Follow these steps in order to reproduce the project results.

### Step 4.1: Run Problem 3 (Spam Experiment)

* **How to Run:** In your activated terminal, run:
    ```bash
    python run_spam_experiment.py
    ```
* **What it Gives:** The script will run 5-fold CV to find the best `alpha` for raw and standardized data. It will print a final summary table to the console and save a plot named `p3_alpha_vs_accuracy.png`.
* **How to Analyze:**
    * You will receive a table like this:
        ```
        --- Summary Results ---
        Preprocessing  | Best Alpha | Train Error | Test Error
        :---------------|:-----------|:------------|:-----------
        Raw            | 0.1        | 0.3280      | 0.3138
        Standardized   | 0.001      | 0.0766      | 0.0771
        ```
    * **Analysis:** Note the **dramatic** difference in performance. The **Standardized** data achieves a low **7.71%** test error, while the **Raw** data fails, with a **31.38%** test error.
    * This is because the raw data has features on vastly different scales, which causes the SGD optimizer to become numerically unstable (as seen by the `RuntimeWarning: overflow` messages during the run). `StandardScaler` fixes this by scaling all features, allowing the optimizer to converge to a good solution.

### Step 4.2: Run Problem 4 (Visualization)

**How to Run:** In your terminal, run:

```bash
python visualize.py
```

**What it Gives:** This creates a new file named `computational_graph.png`

**Analysis:** The resulting image shows the full computation graph for a tiny MLP (Linear(2,3) -> ReLU). It visualizes how the `x_input` Value connects to the weight and bias Values through `@` (matmul) and `+` operations, which then flow into the ReLU node. This confirms the autograd engine and nn modules are building the graph correctly.

### Step 4.3: Run Problem 5 (Capstone Showdown)

**How to Run:**

1. Open `capstone_showdown.ipynb` in VS Code
2. Ensure your venv kernel is selected
3. Click the "Run All" (▶️▶️) button

**What it Gives:** This script will take several minutes.
- It will train all four models (OvR, Softmax, Softmax+RBF, MLP)
- It will print training progress, plots of the autograd models' loss/accuracy, and a final summary table
- Crucially, it will identify the winning model, re-train it, and create the `saved_models/best_model.npz` file

**How to Analyze:**

Observe the plots. You will see the MLP loss decreasing steadily while its validation accuracy climbs, surpassing the other models.

The final output table confirms the winner:

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

**Analysis:** The MLP is the clear winner. Its ReLU non-linearities allow it to learn complex patterns in the pixel data that the linear models (OvR, Softmax) cannot. The Softmax (RBF) model failed completely, indicating that 200 random centers are not sufficient to represent the 784-dimensional data.

### Step 4.4: Verify Final Submission

**How to Run:** After running the notebook (Step 4.3), run this in your terminal:

```bash
python verification.py
```

**What it Gives:** This script mimics the autograder. It imports `create_best_model.py` (which contains the MyMLP architecture) and tries to load the weights from `saved_models/best_model.npz` into it.

**How to Analyze:**

You will see a final summary:

```
--- Verification Summary ---
PASS Required files found.
PASS Model architecture initialized.
PASS Autograd model loaded (check warnings!).

PASS Verification script finished successfully.
```

**Analysis:** This PASS message confirms your submission is valid. The "check warnings" message is a generic reminder; since no "Missing keys" or "Unexpected keys" warnings appeared, the file is perfect.

---

## 5. Best Model Description

The model selected as the "best performing model" based on the validation set accuracy is the **Multi-Layer Perceptron (MLP)**.

* **Best Validation Accuracy:** **89.27%**

This model significantly outperformed all other candidates, including the One-vs-Rest (OvR) Logistic Regression (84.11%) and the linear Softmax Regression (85.83%). Its success is due to the `ReLU` activation functions, which allow the model to learn complex, non-linear patterns in the pixel data that the linear models cannot.

### Model Architecture

The model is an instance of the `MyMLP` class defined in `create_best_model.py`, which is built using modules from the `my_ml_lib.nn` library.

The architecture is a `Sequential` 3-layer feed-forward network:

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