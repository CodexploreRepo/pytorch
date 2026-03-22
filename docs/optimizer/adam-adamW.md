# Adam vs AdamW Optimizer (PyTorch)

- **Adam** = fast, robust baseline — great for small models and tabular data
- **AdamW** = better generalization — default choice for modern deep learning
- **Adaptive scaling** = the reason Adam works so well across diverse architectures
- **Always use `AdamW`** when regularization matters — `Adam`'s weight decay is broken by design

## 1. Overview

Adam and AdamW are popular optimizers used in deep learning.

- **Adam**: Adaptive optimizer with L2 regularization (coupled weight decay)
- **AdamW**: Decoupled weight decay (more correct formulation)

---

## 2. Adam Optimizer

Adam combines **Momentum** + **RMSProp** — it gets the best of both worlds.

| Method         | Idea                                            | Problem                    |
| -------------- | ----------------------------------------------- | -------------------------- |
| SGD + Momentum | Accumulate past gradients to smooth updates     | Same LR for all parameters |
| RMSProp        | Adapt LR per-parameter using gradient magnitude | No momentum                |
| **Adam**       | **Both**                                        | —                          |

### Algorithm

Adam tracks two running statistics of the gradient at each step `t`:

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t       # 1st moment: mean  (momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²      # 2nd moment: variance (RMSProp)
```

**Bias correction** — at early steps `m` and `v` are near zero (initialized at 0), so we correct for that:

```
m̂_t = m_t / (1 - β₁ᵗ)     # unbiased mean estimate
v̂_t = v_t / (1 - β₂ᵗ)     # unbiased variance estimate
```

**Parameter update:**

```
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

### Hyperparameters

| Parameter      | Default | Meaning                                                   |
| -------------- | ------- | --------------------------------------------------------- |
| `lr` (α)       | `1e-3`  | Global learning rate                                      |
| `β₁`           | `0.9`   | Momentum decay — "remember 90% of past direction"         |
| `β₂`           | `0.999` | RMSProp decay — "remember 99.9% of past magnitude"        |
| `ε`            | `1e-8`  | Numerical stability constant                              |
| `weight_decay` | `0`     | L2 regularization (distorted in Adam — use AdamW instead) |

---

## 3. Adaptive Scaling — The Core of Adam

### The Core Idea

> **"If a parameter's gradient is consistently large → slow it down. If small → speed it up."**

Adam does this **automatically, per parameter**, without manually tuning per-layer learning rates.

### Analogy: Driving on a Highway

```
Bumpy road (large gradients) → slow down → small steps
Smooth road (small gradients) → speed up  → large steps
```

Adam does exactly this — but independently for **every single weight** in your network.

### Concrete Example

Say your model has 2 parameters: `w1` and `w2`

```
Step 1:  grad of w1 = 10.0,   grad of w2 = 0.01
Step 2:  grad of w1 = 9.0,    grad of w2 = 0.01
Step 3:  grad of w1 = 11.0,   grad of w2 = 0.01
```

`v_t` accumulates the **squared gradients** over time:

```python
v_w1 ≈ 100    # 10² = 100  → large
v_w2 ≈ 0.0001 # 0.01² → tiny
```

Now the update step:

```
update = α * gradient / (√v + ε)

w1 update = 0.001 * 10.0  / √100    = 0.001 * 10.0 / 10   = 0.001  ← shrunk!
w2 update = 0.001 * 0.01  / √0.0001 = 0.001 * 0.01 / 0.01 = 0.001  ← boosted!
```

Both parameters move by the same effective amount — Adam normalized them. **This is adaptive scaling.**

### Without Adaptive Scaling (plain SGD)

```
w1 update = 0.001 * 10.0  = 0.010    ← 1000x larger than w2!
w2 update = 0.001 * 0.01  = 0.00001  ← barely moves
```

`w1` overshoots wildly while `w2` barely learns.

### The Formula, Simply

```
effective_lr = α / (√v̂ + ε)
                        ↑
              grows large when gradients are large
              → dividing makes the step SMALLER
```

| Gradient history   | `√v̂`   | Effective LR | Behavior               |
| ------------------ | ------ | ------------ | ---------------------- |
| Large & consistent | Large  | Small        | Slow down, be careful  |
| Small & sparse     | Small  | Large        | Speed up, explore more |
| Mixed / noisy      | Medium | Medium       | Balanced               |

### Visual Summary

```
Parameter w1 (large gradients):
  gradient: ████████████ 10.0
  √v̂      : ████████████ 10.0
  step     : █            1.0   ← divided down

Parameter w2 (small gradients):
  gradient: █ 0.01
  √v̂      : █ 0.01
  step     : █ 1.0             ← same effective step!
```

Adam keeps every parameter moving at a **comparable, controlled pace** — that's the power of adaptive scaling.

---

## 4. Weight Decay: Adam vs AdamW

### Adam — Weight Decay Pollutes the Gradient

In Adam, weight decay is implemented as L2 regularization — it is added **into the gradient** before adaptive scaling:

```
g_t = ∇L + λθ          # weight decay added to gradient
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**The problem:** The adaptive scaling `1/(√v̂_t + ε)` also rescales the weight decay term — so the effective regularization strength is distorted per-parameter. This is NOT true weight decay.

### AdamW — Weight Decay Applied Separately

AdamW **decouples** weight decay from the gradient update:

```
g_t = ∇L                                                   # clean gradient
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε) - α * λ * θ_{t-1}  # decay applied AFTER
```

The weight decay `λ` is applied directly to weights after the adaptive step — consistent across all parameters.

---

## 5. Numerical Walkthrough

### Setup

```
w = 2.0    # current weight (large value, should be regularized)
g = 0.1    # gradient from loss
α = 0.01   # learning rate
λ = 0.1    # weight decay
v = 0.04   # accumulated squared gradient (√v = 0.2)
m = 0.1    # accumulated momentum
```

### Adam Step

```python
# Step 1: Add weight decay INTO the gradient
g_modified = g + λ * w = 0.1 + 0.1 * 2.0 = 0.3   # gradient is now 3x larger!

# Step 2: Update momentum using modified gradient
m_new = 0.9 * 0.1 + 0.1 * 0.3 = 0.12

# Step 3: Adaptive update
w_new = 2.0 - 0.01 * 0.12 / (0.2 + ε)
w_new = 1.994
```

### AdamW Step

```python
# Step 1: Gradient stays CLEAN
g_clean = 0.1   # unchanged

# Step 2: Update momentum using clean gradient
m_new = 0.9 * 0.1 + 0.1 * 0.1 = 0.10

# Step 3: Adaptive update (with clean gradient)
adaptive_step = 0.01 * 0.10 / (0.2 + ε) = 0.005

# Step 4: Apply weight decay DIRECTLY to weight
decay_step = α * λ * w = 0.01 * 0.1 * 2.0 = 0.002

w_new = 2.0 - 0.005 - 0.002 = 1.993
```

### Why the Mechanism Matters

The real problem shows up when gradients vary across parameters:

```python
# Parameter A: large gradient (√v is large → adaptive scaling kicks in hard)
# Adam:   decay contribution = λ*w / √v = 0.1*2.0 / 5.0 = 0.04  ← shrunk 5x!
# AdamW:  decay contribution = α*λ*w   = 0.01*0.1*2.0   = 0.002 ← always consistent

# Parameter B: small gradient (√v is tiny → adaptive scaling barely applies)
# Adam:   decay contribution = λ*w / √v = 0.1*2.0 / 0.01 = 20.0 ← explodes!
# AdamW:  decay contribution = α*λ*w   = 0.01*0.1*2.0    = 0.002 ← same as A!
```

Adam's effective weight decay is **completely unpredictable**. AdamW keeps it **uniform**.

---

## 6. Comparison Table

| Feature                    | Adam                          | AdamW                                   |
| -------------------------- | ----------------------------- | --------------------------------------- |
| Weight decay mechanism     | L2 reg (added to gradient)    | Decoupled (applied to weights directly) |
| Effective regularization   | Distorted by adaptive scaling | Uniform, predictable                    |
| Generalization             | Weaker                        | Stronger                                |
| Default in transformers    | Rarely                        | Yes (BERT, GPT, ViT all use AdamW)      |
| Recommended `weight_decay` | Small (`1e-5`)                | Larger (`1e-2`) safe                    |
| PyTorch class              | `torch.optim.Adam`            | `torch.optim.AdamW`                     |

---

## 7. Mental Model to Remember

```
Adam:    [gradient + λw] → adaptive scaling → update
                ↑
         decay gets DISTORTED by adaptive scaling ❌

AdamW:   [gradient]      → adaptive scaling → update
                                                  ↓
                                         then subtract α * λ * w
                                                  ↑
                                  decay is CLEAN, always proportional to w ✅
```

Another way to think about it:

```
Adam   = momentum + RMSProp + broken weight decay
AdamW  = momentum + RMSProp + correct weight decay
```

---

## 8. PyTorch Code

```python
import torch.optim as optim

# Adam — weight_decay is L2 reg mixed into gradient (distorted)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW — weight_decay is true decoupled decay (use larger value safely)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

Per-layer learning rates (e.g. fine-tuning):

```python
optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},  # frozen-ish
    {"params": model.head.parameters(),     "lr": 1e-3},  # train fast
], weight_decay=1e-2)
```

---

## 9. Practical Rule

| Use case                   | Optimizer         |
| -------------------------- | ----------------- |
| Transformers / LLMs        | AdamW             |
| Large CNNs / vision models | AdamW             |
| Small MLP / tabular data   | Adam (often fine) |
| Need strong regularization | AdamW             |

---

## 10. Common Pitfalls

1. **Don't use `Adam` with `weight_decay` for real regularization** — use `AdamW` instead
2. **LR is still the most important hyperparameter** — try `1e-4` or `3e-4` if `1e-3` diverges
3. **Adam can overfit faster** than SGD — compensate with dropout or switch to AdamW
4. **`ε` matters for stability** — increase to `1e-6` if you see NaN losses with fp16 / mixed precision
5. **AdamW tolerates larger `weight_decay`** (e.g. `0.01–0.1`) — don't treat it the same as Adam's
