import numpy as np
from .modules.base import Module
from .autograd import Value

class BinaryCrossEntropyLoss(Module):
    """Computes the Binary Cross Entropy loss between logits and targets (0 or 1).
    Assumes the input is a single logit (pre-sigmoid score) per sample.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, logits: Value, targets: np.ndarray) -> Value:
        """Calculates the Binary Cross Entropy loss."""
        
        # --- Step 1 Apply Sigmoid ---
        # This gives the predicted probabilities 'p'.
        # We'll use the .sigmoid() method we added to Value
        probs = logits.sigmoid()

        # --- Step 2: Numerical Stability (Clipping) ---
        # Our Value.log() already handles stability, so this is covered.
        probs_stable = probs # Use probs directly

        # --- Step 3 Wrap Targets ---
        # Ensure targets are float64 and have the correct shape
        targets_np = targets.astype(np.float64).reshape(logits.data.shape)
        targets_val = Value(targets_np)

        # --- Step 4 Calculate BCE Formula ---
        # loss = -[ y * log(p) + (1-y) * log(1-p) ]
        term1 = targets_val * probs_stable.log()
        term2 = (1.0 - targets_val) * (1.0 - probs_stable).log()
        loss_elements = -(term1 + term2)

        # --- Step 5 Apply Reduction ---
        if self.reduction == 'mean':
            loss = loss_elements.mean()
        elif self.reduction == 'sum':
            loss = loss_elements.sum()
        else: # 'none'
            loss = loss_elements
            
        return loss

    def __repr__(self):
        return f"BinaryCrossEntropyLoss(reduction='{self.reduction}')"

class CrossEntropyLoss(Module):
    """Computes the cross-entropy loss between input logits and target class indices.
    Combines LogSoftmax and NLLLoss.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, input_logits: Value, target: np.ndarray) -> Value:
        """Computes the cross-entropy loss using only Value operations."""
        batch_size, n_classes = input_logits.data.shape

        # --- Step 1 Calculate LogSoftmax ---
        # a) Find max logit per sample (as a numpy array)
        max_logits_np = np.max(input_logits.data, axis=1, keepdims=True)
        # b) Subtract max logit (as a constant)
        stable_logits = input_logits - max_logits_np # Broadcasting works
        # c) Exponentiate
        exp_logits = stable_logits.exp()
        # d) Sum over class dimension
        sum_exp_logits = exp_logits.sum(axis=1, keepdims=True)
        # e) Take the logarithm
        log_sum_exp = sum_exp_logits.log()
        # f) Subtract log_sum_exp from stable_logits
        log_probs = stable_logits - log_sum_exp # shape (batch_size, n_classes)

        # --- Step 2 Create One-Hot Targets (Numpy) ---
        y_one_hot_np = np.zeros((batch_size, n_classes), dtype=np.float64)
        y_one_hot_np[np.arange(batch_size), target] = 1.0

        # --- Step 3 Wrap One-Hot Targets (Value) ---
        y_one_hot_val = Value(y_one_hot_np)

        # --- Step 4 Calculate NLL ---
        # NLL = -sum(y_one_hot * log_probs) over classes
        # a) Multiply element-wise
        element_wise_loss = y_one_hot_val * log_probs
        # b) Negate (done at the end)
        # c) Sum over class dimension (axis=1)
        nll_per_sample = -element_wise_loss.sum(axis=1, keepdims=False) # shape (batch_size,)

        # --- Step 5 Apply Reduction ---
        if self.reduction == 'mean':
            loss = nll_per_sample.mean()
        elif self.reduction == 'sum':
            loss = nll_per_sample.sum()
        else: # 'none'
            loss = nll_per_sample

        return loss

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"