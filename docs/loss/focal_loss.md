# Focal Loss — Quick Recall Note

## Purpose

Focal Loss helps models learn from **hard examples** in **imbalanced datasets** by **down-weighting easy samples**.

Typical use cases:

- Fraud detection
- Credit default prediction
- Object detection
- Any rare-event classification

---

## Problem with Standard BCE

Binary Cross Entropy:

L = -[ y log(p) + (1-y) log(1-p) ]

Issue in imbalanced datasets:

Many **easy negatives** dominate training.

Example dataset:

- negatives: 1,000,000
- positives: 10,000

Most negatives quickly become:

y = 0, p ≈ 0.01

Loss is small but **millions of them still dominate gradients**.

The model wastes time improving already-correct examples.

---

## Key Idea of Focal Loss

Reduce importance of **easy examples**.

Focal Loss:

FL = (1 - p_t)^γ \* BCE

Where:

p_t = model probability of the correct class

γ (gamma) = focusing parameter (commonly 2)

---

## Meaning of p_t

Binary case:

p_t = p if y = 1  
p_t = 1 - p if y = 0

p_t represents **model confidence for the correct class**.

Examples:

| Case        | p_t  | Difficulty |
| ----------- | ---- | ---------- |
| y=1, p=0.95 | 0.95 | easy       |
| y=1, p=0.55 | 0.55 | medium     |
| y=1, p=0.10 | 0.10 | very hard  |

---

## Effect of the Focal Weight

Weight = (1 - p_t)^γ

If γ = 2:

| p_t  | Weight | Meaning        |
| ---- | ------ | -------------- |
| 0.95 | 0.0025 | almost ignored |
| 0.55 | 0.20   | moderate       |
| 0.10 | 0.81   | strong penalty |

So:

- easy examples → almost ignored
- hard examples → emphasized

---

## Intuition

Normal BCE:

easy example → small loss  
hard example → big loss

Focal Loss:

easy example → **almost zero weight**  
hard example → **large contribution**

Training focuses on **decision boundary**.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Numerically stable focal loss for binary classification.

    Uses BCEWithLogits internally for stability.

    Args:
        alpha (float): weight for positive class
        gamma (float): focusing parameter
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        targets = targets.float()

        # stable BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # predicted probability
        probs = torch.sigmoid(logits)

        # pt = probability of the true class
        pt = torch.where(targets == 1, probs, 1 - probs)

        # alpha weighting
        alpha_factor = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha)
        )

        # focal weight
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        else:
            return loss
```

---

## Why Kaggle Uses Focal Loss

Advantages:

- focuses learning on hard samples
- reduces dominance of easy negatives
- improves **PR-AUC / ROC-AUC**
- works well in highly imbalanced datasets

---

## Key Takeaway

Focal Loss = **BCE × difficulty weight**

Easy samples → ignored  
Hard samples → emphasized

Goal:

**Focus training on difficult examples.**
