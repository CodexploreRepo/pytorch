# Handling Imbalanced Data

Example dataset:

Class Percentage

---

Negative 92%
Positive 8%

Problem:

Model sees many more negatives → gradients dominated by negatives.

Common solutions:

1.  pos_weight
2.  Stratified batch sampling
3.  Focal loss

---

## `pos_weight` (BCEWithLogitsLoss)

```Python
def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    pos_weight = n_neg / max(n_pos, 1)
    return torch.tensor(pos_weight, dtype=torch.float32)
pos_weight = compute_pos_weight(y_train).to(CFG.device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Modified loss

L = -\[ pos_weight \* y log(p) + (1-y) log(1-p) \]

Meaning:

Sample Type Weight

---

Positive scaled
Negative unchanged

### Effect

- increases penalty for positive mistakes
- improves recall
- **distorts predicted probabilities**

Example:

True probability = 0.08\
Model prediction with pos_weight ≈ 0.47

Ranking may still be correct but probability calibration is lost.

---

## Stratified Batch Sampling

Instead of modifying the loss, change **how batches are sampled**.

Original dataset:

    92% negative
    8% positive

Balanced batch example:

    256 negatives
    256 positives

### Key property

Loss function stays the same:

    L = -[ y log(p) + (1-y) log(1-p) ]

So the model still estimates the **true probability**.

Benefits:

- better gradient signal
- preserves probability interpretation

---

## Focal Loss

Designed for extreme class imbalance.

Formula:

FL = -(1-p)\^γ y log(p) - p\^γ (1-y) log(1-p)

Typical parameter:

    γ = 2

### Intuition

Sample Effect

---

Easy example smaller gradient
Hard example larger gradient

The model focuses on **hard samples near the decision boundary**.

Example implementation:

```python
class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1-targets)*(1-probs)

        loss = (1-pt)**self.gamma * bce

        return loss.mean()
```

---

## Comparison

Method What changes Probability calibration

---

BCE nothing correct
pos_weight loss function distorted
stratified sampling sampling frequency preserved
focal loss gradient emphasis mostly preserved

---

## Practical Guidance

For learning / baseline:

    BCEWithLogitsLoss + pos_weight

For competitive tabular models (Kaggle):

    Stratified sampling + focal loss

For production risk models:

    BCE + pos_weight + probability calibration

---
