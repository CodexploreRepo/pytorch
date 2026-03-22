# PyTorch Training Loop

## 1. Standard PyTorch Training Loop

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # This resets gradients.
        optimizer.zero_grad()
        # Forward pass
        logits = model(X_batch)
        # Compute loss: average loss of the batch
        loss = criterion(logits, y_batch)
        # Backward pass: compute gradients. This uses automatic differentiation (autograd).
        loss.backward()

        # Update weights
        # new_weight = old_weight - learning_rate * gradient
        optimizer.step()

        # Loss calculation
        batch_size = X_batch.size(0)

        # Total loss of the batch: average loss of the batch * batch_size
        # Since the loss returned was mean, so to compute dataset average correctly we do:
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples
```

### Conceptual Pipeline

    Input batch
        ↓
    Forward pass (model)
        ↓
    Prediction (logits)
        ↓
    Loss calculation
        ↓
    Backward pass (gradients)
        ↓
    Optimizer update: new_weight = old_weight - learning_rate * gradient

---

# 2. Loss Calculation in Training

For binary classification we usually use:

```python
criterion = torch.nn.BCEWithLogitsLoss()
```

### Why logits?

Model outputs **logits** (raw scores):

    (-∞ , +∞)

Loss internally applies **sigmoid + BCE** in a numerically stable way.

---

## Binary Cross Entropy

Loss per sample:

L = -\[ y log(p) + (1-y) log(1-p) \]

Where:

    p = sigmoid(logit)

### Batch Loss

By default:

    reduction = "mean"

So final loss is:

    batch_loss = average(sample_losses)

```shell
# suppose batch_size = 4
sample1 = 0.20
sample2 = 0.80
sample3 = 0.40
sample4 = 0.60

# With default "mean":
loss = (0.20 + 0.80 + 0.40 + 0.60) / 4
     = 0.50
```

- PyTorch allows three options:
  - `nn.BCEWithLogitsLoss(reduction="mean")` (most common) return a scalar value `0.5`
  - `nn.BCEWithLogitsLoss(reduction="sum")` returns `loss = 0.20 + 0.80 + 0.40 + 0.60 = 2.0`
  - `nn.BCEWithLogitsLoss(reduction="none")` return **per-sample** loss `tensor([0.20, 0.80, 0.40, 0.60])` instead of `0.5` with "mean" option
    - Useful for: custom weighting, focal loss, hard example mining

```Python
# Example: example of focal loss using reduction="none"

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = targets * probs + (1 - targets) * (1 - probs)

        loss = ((1 - pt) ** self.gamma) * bce

        return loss.mean()
```
