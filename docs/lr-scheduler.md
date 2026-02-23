# LEARNING RATE SCHEDULERS

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Part 1: Why LR Scheduling Matters](#part-1-why-lr-scheduling-matters)
  - [1.1 The Problem with a Fixed LR](#11-the-problem-with-a-fixed-lr)
  - [1.2 What a Scheduler Does](#12-what-a-scheduler-does)
  - [1.3 Key Concepts](#13-key-concepts)
- [Part 2: Scheduler Types](#part-2-scheduler-types)
  - [2.1 ReduceLROnPlateau — Reactive / Metric-Driven](#21-reducelronplateau--reactive--metric-driven)
  - [2.2 CosineAnnealingLR — Smooth Fixed Decay](#22-cosineannealinglr--smooth-fixed-decay)
  - [2.3 StepLR — Simple Step-Wise Decay](#23-steplr--simple-step-wise-decay)
  - [2.4 OneCycleLR — Warmup + Peak + Decay in One](#24-onecyclelr--warmup--peak--decay-in-one)
  - [2.5 CosineAnnealingWarmRestarts — Periodic Restarts](#25-cosineannealingwarmrestarts--periodic-restarts)
  - [2.6 LinearLR / Warmup — Early Training Stabilization](#26-linearlr--warmup--early-training-stabilization)
  - [2.7 SequentialLR — Chaining Schedulers](#27-sequentiallr--chaining-schedulers)
- [Part 3: Best Practices](#part-3-best-practices)
  - [3.0 Start Without a Scheduler — Establish a Baseline First](#30-start-without-a-scheduler--establish-a-baseline-first)
  - [3.1 Always Call scheduler.step() After optimizer.step()](#31-always-call-schedulerstep-after-optimizerstep)
  - [3.2 Per-Scheduler Call Frequency](#32-per-scheduler-call-frequency)
  - [3.3 Set min_lr to Avoid LR Decaying to Zero](#33-set-min_lr-to-avoid-lr-decaying-to-zero)
  - [3.4 Use Warmup for Embedding-Heavy or Transformer Models](#34-use-warmup-for-embedding-heavy-or-transformer-models)
  - [3.5 Separate Weight Decay from LR Scheduling](#35-separate-weight-decay-from-lr-scheduling)
  - [3.6 Always Log the Current LR](#36-always-log-the-current-lr)
  - [3.7 Combining with Early Stopping](#37-combining-with-early-stopping)
- [Part 4: Decision Guide](#part-4-decision-guide)
  - [4.1 Flowchart: Which Scheduler to Pick](#41-flowchart-which-scheduler-to-pick)
  - [4.2 Comparison Table](#42-comparison-table)
  - [4.3 Recommendations by Model Type](#43-recommendations-by-model-type)
- [Part 5: Common Mistakes](#part-5-common-mistakes)
  - [5.1 Calling step() in the Wrong Order](#51-calling-step-in-the-wrong-order)
  - [5.2 Not Setting min_lr (LR Goes to Zero)](#52-not-setting-min_lr-lr-goes-to-zero)
  - [5.3 ReduceLROnPlateau Patience Conflicts with Early Stopping](#53-reducelronplateau-patience-conflicts-with-early-stopping)
  - [5.4 Forgetting OneCycleLR Steps Per Batch](#54-forgetting-onecyclelr-steps-per-batch)
- [Part 6: Summary Table](#part-6-summary-table)

---

# Part 1: Why LR Scheduling Matters

## 1.1 The Problem with a Fixed LR

Training a neural network with a **constant learning rate** is almost always suboptimal:

```
Too HIGH throughout:
  ✗ Early training:  overshoots minima → loss spikes, divergence
  ✗ Late training:   oscillates around optimum, never converges cleanly

Too LOW throughout:
  ✗ Early training:  painfully slow progress
  ✗ Late training:   technically fine, but wastes the early epochs

Fixed LR = a single compromise that is wrong for most of training
```

**The trade-off in practice:**

| Phase         | Ideal LR   | Reason                                               |
| ------------- | ---------- | ---------------------------------------------------- |
| Warmup        | Very small | Avoid bad initial updates before gradients stabilize |
| Early–mid     | Large      | Explore loss landscape quickly                       |
| Late training | Small      | Converge precisely to a sharp minimum                |

## 1.2 What a Scheduler Does

A **learning rate scheduler** automatically adjusts the optimizer's learning rate according to a pre-defined or adaptive policy.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(epochs):
    train_one_epoch(model, optimizer)
    scheduler.step()   # LR updated after each epoch
```

The scheduler does **not** change the model or the optimizer's weight decay — it only modifies the `lr` parameter stored in each `param_group`.

[(Back to top)](#table-of-contents)

## 1.3 Key Concepts

| Concept         | Definition                                                                          |
| --------------- | ----------------------------------------------------------------------------------- |
| **LR Range**    | The interval `[min_lr, initial_lr]` over which the scheduler operates               |
| **Warmup**      | A brief phase at the start where LR ramps up from a small value to the target       |
| **Decay**       | Systematic reduction of LR over time (step, cosine, linear, exponential)            |
| **Restarts**    | Periodically resetting LR to its peak to escape local minima (SGDR / warm restarts) |
| **Patience**    | In reactive schedulers: number of non-improving epochs before LR is reduced         |
| **T_max / T_0** | Period length for cosine-based schedulers (measured in epochs or steps)             |

[(Back to top)](#table-of-contents)

---

# Part 2: Scheduler Types

## 2.1 ReduceLROnPlateau — Reactive / Metric-Driven

**When to use:** When you want the scheduler to react to validation loss or another metric, rather than following a fixed curve. Good default when you don't know how many epochs you need.

**How it works:**

```
Monitor a metric (e.g., val_loss) every epoch.
If no improvement for `patience` epochs → multiply LR by `factor`.
Repeat until min_lr is reached.
```

**Code example:**

```python
import torch
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 'min' for loss, 'max' for metrics like Recall
    factor=0.5,       # Multiply LR by this when triggered
    patience=3,       # Epochs with no improvement before reducing
    min_lr=1e-6,      # Floor; LR never goes below this
    verbose=True
)

for epoch in range(epochs):
    train_loss = train_one_epoch(model, optimizer)
    val_loss   = evaluate(model, val_loader)
    scheduler.step(val_loss)   # Pass the metric here, not epoch
```

**Pros and Cons:**

| Pros                                         | Cons                                              |
| -------------------------------------------- | ------------------------------------------------- |
| ✓ Adapts automatically to training dynamics  | ✗ Requires a validation metric every epoch        |
| ✓ Robust; works across many model types      | ✗ Can conflict with early stopping patience       |
| ✓ No need to choose a fixed number of epochs | ✗ Irreversible — LR only goes down, never back up |

[(Back to top)](#table-of-contents)

## 2.2 CosineAnnealingLR — Smooth Fixed Decay

**When to use:** When you know (approximately) how many epochs you will train and want a smooth, principled decay curve. A strong default for most deep learning workloads.

**How it works:**

```
LR follows one half of a cosine curve from initial_lr down to eta_min:

LR(t) = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(π * t / T_max))

where t  = current epoch
      T_max = total number of epochs (or half-period)
```

**Code example:**

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,       # Total epochs (or half-period for restarts)
    eta_min=1e-6    # Minimum LR at the end of the cycle
)

for epoch in range(50):
    train_one_epoch(model, optimizer)
    scheduler.step()   # Called once per epoch
```

**Pros and Cons:**

| Pros                                     | Cons                                          |
| ---------------------------------------- | --------------------------------------------- |
| ✓ Smooth — avoids sharp LR drops         | ✗ Requires knowing the total number of epochs |
| ✓ Reaches a low LR gracefully at the end | ✗ Does not adapt to validation performance    |
| ✓ Pairs well with warmup (SequentialLR)  | ✗ Fixed curve regardless of loss trajectory   |

[(Back to top)](#table-of-contents)

## 2.3 StepLR — Simple Step-Wise Decay

**When to use:** When you want explicit, interpretable control over when LR drops. Often used in computer vision pipelines with well-known training recipes (e.g., ResNet: drop at epoch 30 and 60 of 90).

**How it works:**

```
Every `step_size` epochs, multiply LR by `gamma`:

LR(t) = initial_lr * gamma^(floor(t / step_size))
```

**Code example:**

```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,   # Drop LR every 10 epochs
    gamma=0.1       # Multiply LR by 0.1 at each step
)

for epoch in range(30):
    train_one_epoch(model, optimizer)
    scheduler.step()
```

**Pros and Cons:**

| Pros                             | Cons                                            |
| -------------------------------- | ----------------------------------------------- |
| ✓ Extremely simple to understand | ✗ Requires manual tuning of step_size and gamma |
| ✓ Predictable, easy to reproduce | ✗ Abrupt drops can destabilize training         |
| ✓ Works well with SGD + momentum | ✗ Not adaptive; ignores validation signal       |

[(Back to top)](#table-of-contents)

## 2.4 OneCycleLR — Warmup + Peak + Decay in One

**When to use:** When you want fast convergence in a fixed number of steps, especially with SGD. Commonly used in training recipes that prioritize speed (e.g., super-convergence).

**How it works:**

```
Phase 1 (warmup):  LR ramps from initial_lr/div_factor → max_lr
Phase 2 (decay):   LR decays from max_lr → initial_lr/final_div_factor

Both phases use cosine annealing internally.
```

**Code example:**

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)

steps_per_epoch = len(train_loader)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,                            # Peak LR
    steps_per_epoch=steps_per_epoch,       # Steps in one epoch
    epochs=30,                             # Total training epochs
    pct_start=0.3,                         # 30% of steps for warmup
    anneal_strategy='cos',                 # 'cos' or 'linear'
    div_factor=25,                         # initial_lr = max_lr / 25
    final_div_factor=1e4                   # final_lr = max_lr / 1e4
)

for epoch in range(30):
    for batch in train_loader:
        loss = compute_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()   # ← called per BATCH, not per epoch
```

**Pros and Cons:**

| Pros                                               | Cons                                                 |
| -------------------------------------------------- | ---------------------------------------------------- |
| ✓ Combines warmup and decay in a single object     | ✗ Must call step() per batch — common source of bugs |
| ✓ Can achieve fast convergence (super-convergence) | ✗ Requires knowing total_steps upfront               |
| ✓ Single tunable knob: max_lr                      | ✗ Less intuitive than epoch-based schedulers         |

[(Back to top)](#table-of-contents)

## 2.5 CosineAnnealingWarmRestarts — Periodic Restarts

**When to use:** When training for many epochs and you want the model to escape local minima by periodically resetting to a higher LR (SGDR — Stochastic Gradient Descent with Warm Restarts).

**How it works:**

```
LR follows cosine decay from initial_lr → eta_min over T_0 epochs,
then resets to initial_lr and repeats. Each cycle can be longer than the last:

  T_i = T_0 * T_mult^i    (cycle length grows by T_mult each restart)
```

**Code example:**

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,       # Length of the first cycle (epochs)
    T_mult=2,     # Each restart doubles the cycle length
    eta_min=1e-6  # Minimum LR within each cycle
)

for epoch in range(70):   # Covers cycles: 10, 20, 40 epochs
    train_one_epoch(model, optimizer)
    scheduler.step()
```

**Pros and Cons:**

| Pros                                             | Cons                                         |
| ------------------------------------------------ | -------------------------------------------- |
| ✓ Can escape local minima via periodic LR spikes | ✗ Snapshots at restart peaks can be unstable |
| ✓ Good for long training runs                    | ✗ Requires tuning T_0 and T_mult             |
| ✓ Works well with snapshot ensembling            | ✗ Overkill for short training runs           |

[(Back to top)](#table-of-contents)

## 2.6 LinearLR / Warmup — Early Training Stabilization

**When to use:** At the start of training (especially with transformers or embedding models) to ramp LR gradually from near-zero to the target. Prevents destructive updates in the first few steps when gradients are large and unreliable.

**How it works:**

```
LR grows linearly from start_factor * initial_lr to end_factor * initial_lr
over `total_iters` steps:

LR(t) = initial_lr * (start_factor + (end_factor - start_factor) * t / total_iters)
```

**Code example:**

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Warm up over the first 5 epochs
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,    # Start at 10% of initial_lr
    end_factor=1.0,      # End at 100% of initial_lr
    total_iters=5        # Number of epochs for warmup
)

for epoch in range(5):
    train_one_epoch(model, optimizer)
    warmup_scheduler.step()
```

**Pros and Cons:**

| Pros                                              | Cons                                                |
| ------------------------------------------------- | --------------------------------------------------- |
| ✓ Simple and transparent                          | ✗ Typically used as a component within SequentialLR |
| ✓ Stabilizes early training                       | ✗ Does not handle decay; needs to be chained        |
| ✓ Essential for transformers and large embeddings | ✗ Adds one more hyperparameter (warmup duration)    |

[(Back to top)](#table-of-contents)

## 2.7 SequentialLR — Chaining Schedulers

**When to use:** When you want distinct phases (e.g., warmup followed by cosine decay) in a single clean abstraction rather than manually switching schedulers mid-training.

**How it works:**

```
Execute scheduler_1 for `milestone` steps, then switch to scheduler_2.

Epoch 0–4:   LinearLR  (warmup)
Epoch 5–54:  CosineAnnealingLR  (decay)
```

**Code example:**

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

warmup = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=5
)

cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=45,       # Remaining epochs after warmup
    eta_min=1e-6
)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[5]   # Switch at epoch 5
)

for epoch in range(50):
    train_one_epoch(model, optimizer)
    scheduler.step()   # SequentialLR handles the hand-off automatically
```

**Pros and Cons:**

| Pros                                              | Cons                                                    |
| ------------------------------------------------- | ------------------------------------------------------- |
| ✓ Clean single interface for multi-phase training | ✗ Milestone must align with sub-scheduler total_iters   |
| ✓ Reuses existing schedulers as components        | ✗ Slightly more verbose setup                           |
| ✓ Best practice for warmup + cosine pattern       | ✗ Sub-schedulers share the optimizer — be careful about |
|                                                   | resetting internal state between phases                 |

[(Back to top)](#table-of-contents)

---

# Part 3: Best Practices

## 3.0 Start Without a Scheduler — Establish a Baseline First

Before adding any scheduler, train the model with a **fixed LR** and record the results. This gives you a concrete baseline to measure whether the scheduler actually helps.

```python
# Baseline: no scheduler, fixed LR
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    train_loss = train_one_epoch(model, optimizer)
    val_loss   = evaluate(model, val_loader)
    print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} "
          f"| lr={optimizer.param_groups[0]['lr']:.2e}")
```

Run this and note:

```
Baseline (no scheduler):
  Best val_loss  = 0.3142  (epoch 18)
  Recall@10      = 0.1210
  Training time  = 42s
```

Only then add a scheduler and compare:

```
With CosineAnnealingLR:
  Best val_loss  = 0.2987  (epoch 24)   ← improved
  Recall@10      = 0.1340               ← improved
  Training time  = 45s
```

**Why this matters:**

- Schedulers add hyperparameters (`T_max`, `patience`, `min_lr`, warmup duration). Without a baseline you cannot tell whether an improvement comes from the scheduler or from running more epochs.
- A scheduler that hurts performance relative to the baseline is a sign the fixed LR or the schedule shape is wrong — not that schedulers are useless.
- On small datasets (e.g., MovieLens 100K) a well-tuned fixed LR often matches or beats a poorly tuned scheduler.

**Rule of thumb:** if the fixed-LR training curve is still declining at epoch N, a scheduler probably won't help much — training simply needs more epochs. If the curve has plateaued, that's when a scheduler (or a manual LR drop) can unlock further improvement.

[(Back to top)](#table-of-contents)

## 3.1 Always Call scheduler.step() After optimizer.step()

**Rule:** `optimizer.step()` → then `scheduler.step()`. Never the other way around.

```python
# ✓ Correct order
optimizer.zero_grad()
loss.backward()
optimizer.step()     # Apply gradient update first
scheduler.step()     # Then update the LR for the next step

# ✗ Wrong order (deprecated and may produce incorrect LR values)
scheduler.step()
optimizer.step()
```

**Why it matters:** Calling `scheduler.step()` first updates the LR before the optimizer uses it, meaning the first update uses the _modified_ LR rather than the intended initial one. PyTorch will emit a deprecation warning if you call them in the wrong order.

[(Back to top)](#table-of-contents)

## 3.2 Per-Scheduler Call Frequency

Different schedulers expect `step()` at different granularities:

| Scheduler                     | Call Frequency | What gets passed to step()    |
| ----------------------------- | -------------- | ----------------------------- |
| `StepLR`                      | Per epoch      | Nothing                       |
| `CosineAnnealingLR`           | Per epoch      | Nothing                       |
| `CosineAnnealingWarmRestarts` | Per epoch      | Nothing (or epoch as float)   |
| `ReduceLROnPlateau`           | Per epoch      | Metric value (e.g., val_loss) |
| `LinearLR`                    | Per epoch      | Nothing                       |
| `SequentialLR`                | Per epoch      | Nothing                       |
| **`OneCycleLR`**              | **Per batch**  | Nothing                       |

```python
# ReduceLROnPlateau: always pass the metric
scheduler.step(val_loss)

# OneCycleLR: step inside the batch loop, not the epoch loop
for batch in train_loader:
    ...
    optimizer.step()
    scheduler.step()   # ← inside batch loop

# All others: step at the end of each epoch
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()   # ← outside batch loop
```

[(Back to top)](#table-of-contents)

## 3.3 Set min_lr to Avoid LR Decaying to Zero

Always specify a `min_lr` (or `eta_min`) floor. Without it, the LR can decay to effectively zero, freezing learning entirely while training still appears to run.

```python
# ✓ With floor — LR stays usable
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6    # ← never goes below 1e-6
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3, min_lr=1e-6   # ← same idea
)

# ✗ Without floor — LR can reach ~0 after many reductions
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3    # LR: 1e-3 → 5e-4 → 2.5e-4 → ... → 0
)
```

**Typical min_lr values:**

| Model type                | Suggested min_lr |
| ------------------------- | ---------------- |
| MLP / simple feedforward  | `1e-5`           |
| Embedding models (DeepFM) | `1e-6`           |
| Transformers (BERT4Rec)   | `1e-7`           |

[(Back to top)](#table-of-contents)

## 3.4 Use Warmup for Embedding-Heavy or Transformer Models

Large embedding tables and transformer attention layers have **many parameters initialized near zero**. Without warmup, the first few batches produce large, inconsistent gradients that can corrupt the embedding space before it has a chance to organize.

```python
# Recommended pattern for NeuMF, DeepFM, BERT4Rec, Two-Tower models
optimizer = optim.Adam(model.parameters(), lr=1e-3)

warmup  = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=5)
cosine  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[5]
)
```

**General rule:** Use warmup whenever:

- The model has > 1 M parameters
- The model contains embedding layers with large vocabulary
- You observe loss spikes in the first 5–10 epochs

[(Back to top)](#table-of-contents)

## 3.5 Separate Weight Decay from LR Scheduling

Use `AdamW` (not `Adam`) and set `weight_decay` there. Do **not** implement L2 regularization by scaling the LR, and do not tie weight decay to the scheduler.

```python
# ✓ Correct: weight decay is a separate optimizer concern
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2    # L2 penalty applied independently of LR
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# ✗ Wrong: conflates two separate hyperparameters
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# ... and manually adding L2 in the training loop mixed with scheduler
```

**Why AdamW?** In vanilla `Adam`, weight decay is absorbed into the adaptive gradient scaling and effectively becomes weaker for parameters with large gradients. `AdamW` applies it directly to weights, making it independent of the LR schedule.

[(Back to top)](#table-of-contents)

## 3.6 Always Log the Current LR

Log the learning rate at every epoch so you can reconstruct what happened if training diverges.

```python
for epoch in range(epochs):
    train_loss = train_one_epoch(model, optimizer)
    val_loss   = evaluate(model, val_loader)
    scheduler.step(val_loss)  # or scheduler.step()

    # Log current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | lr={current_lr:.2e}")
```

**With TensorBoard / WandB:**

```python
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Train/LR', current_lr, epoch)

# Weights & Biases
import wandb
wandb.log({"lr": current_lr, "train_loss": train_loss}, step=epoch)
```

[(Back to top)](#table-of-contents)

## 3.7 Combining with Early Stopping

When using `ReduceLROnPlateau` and early stopping together, their `patience` values must be set deliberately:

```
Bad setup:
  ReduceLROnPlateau patience = 5
  EarlyStopping patience     = 5

  Both fire at the same epoch → no time for the reduced LR to help

Good setup:
  ReduceLROnPlateau patience = 3   ← fires sooner
  EarlyStopping patience     = 10  ← gives the reduced LR time to work
```

**Practical guideline:**

```python
# Rule of thumb: early_stopping_patience ≥ 2 × scheduler_patience
scheduler    = ReduceLROnPlateau(..., patience=3)
early_stop   = EarlyStopping(patience=8)   # enough room for 2 reductions
```

[(Back to top)](#table-of-contents)

---

# Part 4: Decision Guide

## 4.1 Flowchart: Which Scheduler to Pick

```
START
  │
  ├─ Do you know the total number of training epochs?
  │     │
  │     ├─ NO  → ReduceLROnPlateau  (reactive, metric-driven)
  │     │
  │     └─ YES
  │           │
  │           ├─ Is this a Transformer or embedding-heavy model?
  │           │     │
  │           │     └─ YES → SequentialLR (LinearLR warmup + CosineAnnealingLR)
  │           │
  │           ├─ Do you want to escape local minima with periodic resets?
  │           │     │
  │           │     └─ YES → CosineAnnealingWarmRestarts
  │           │
  │           ├─ Are you using SGD and want super-convergence in one run?
  │           │     │
  │           │     └─ YES → OneCycleLR (step per batch)
  │           │
  │           ├─ Do you have a known step schedule (e.g., drop at epoch 30, 60)?
  │           │     │
  │           │     └─ YES → StepLR
  │           │
  │           └─ Default for most cases → CosineAnnealingLR
```

[(Back to top)](#table-of-contents)

## 4.2 Comparison Table

| Scheduler                     | Trigger       | Call Frequency | Adaptive | Warmup | min_lr | Typical Use Case                     |
| ----------------------------- | ------------- | -------------- | -------- | ------ | ------ | ------------------------------------ |
| `ReduceLROnPlateau`           | Metric-driven | Per epoch      | ✓        | ✗      | ✓      | Unknown training length; general use |
| `CosineAnnealingLR`           | Fixed curve   | Per epoch      | ✗        | ✗      | ✓      | Known epochs; smooth decay           |
| `StepLR`                      | Fixed steps   | Per epoch      | ✗        | ✗      | ✗      | SGD + known drop schedule            |
| `OneCycleLR`                  | Fixed steps   | Per batch      | ✗        | ✓      | ✓      | Fast convergence with SGD            |
| `CosineAnnealingWarmRestarts` | Fixed cycle   | Per epoch      | ✗        | ✗      | ✓      | Long runs; escaping local minima     |
| `LinearLR`                    | Fixed ramp    | Per epoch      | ✗        | ✓      | ✗      | Warmup phase; used inside Sequential |
| `SequentialLR`                | Milestone     | Per epoch      | ✗        | ✓      | ✓      | Warmup + cosine; transformers        |

[(Back to top)](#table-of-contents)

## 4.3 Recommendations by Model Type

### MLP / Feedforward Networks

```python
# Simple, reliable choice
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
)
```

**Why:** MLPs converge quickly and don't require warmup. `ReduceLROnPlateau` is forgiving when you're unsure how many epochs are needed.

### Embedding Models (DeepFM, NeuMF, Two-Tower)

```python
# Warmup + cosine is the standard recipe
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.05, end_factor=1.0, total_iters=5
)
cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=45, eta_min=1e-6
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[5]
)
```

**Why:** Embedding layers initialize near zero. Large gradient updates in the first few batches can corrupt the embedding space before it has a chance to form meaningful structure. Warmup prevents this.

### Sequential / Transformer-Based Models (BERT4Rec)

```python
# Longer warmup, lower min_lr, AdamW required
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=10
)
cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=90, eta_min=1e-7
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[10]
)
```

**Why:** Transformers are highly sensitive to early LR spikes. A longer warmup (10 epochs instead of 5) and a lower initial LR (`1e-4` vs `1e-3`) are standard practice from BERT fine-tuning recipes. `AdamW` is mandatory to correctly decouple weight decay from gradient scaling.

[(Back to top)](#table-of-contents)

---

# Part 5: Common Mistakes

## 5.1 Calling step() in the Wrong Order

```python
# ✗ Wrong — updates LR before the gradient step
for batch in train_loader:
    loss = compute_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    scheduler.step()   # ← LR changed before optimizer uses it
    optimizer.step()

# ✓ Correct
for batch in train_loader:
    loss = compute_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   # ← gradient applied at current LR
    scheduler.step()   # ← LR updated for the next step
```

**Symptom:** PyTorch emits a `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`. Training may still progress, but LR values are off by one step.

[(Back to top)](#table-of-contents)

## 5.2 Not Setting min_lr (LR Goes to Zero)

```python
# ✗ Dangerous — after enough epochs or reductions, LR → ~0
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# After 100 epochs: LR ≈ 0.0 — model parameters no longer update

# ✓ Safe — LR bottoms out at a useful value
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
```

**How to diagnose:** Always log `optimizer.param_groups[0]['lr']` every epoch. If you see values like `1e-12` or `0.0`, training has effectively stopped.

[(Back to top)](#table-of-contents)

## 5.3 ReduceLROnPlateau Patience Conflicts with Early Stopping

```python
# ✗ Both fire at epoch 5 → no recovery time
scheduler  = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
early_stop = EarlyStopping(patience=5)

# ✓ Scheduler fires first, early stopping gives it room
scheduler  = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
early_stop = EarlyStopping(patience=10)
```

**What happens in the bad case:** When both triggers fire simultaneously, early stopping halts training at the exact moment the lower LR might have helped. You lose potential recovery without ever testing the reduced LR.

[(Back to top)](#table-of-contents)

## 5.4 Forgetting OneCycleLR Steps Per Batch

```python
# ✗ Wrong — called once per epoch (too infrequent)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=30
)

for epoch in range(30):
    for batch in train_loader:
        ...
        optimizer.step()
    scheduler.step()   # ← only 30 steps instead of 30 × N

# ✓ Correct — called inside the batch loop
for epoch in range(30):
    for batch in train_loader:
        ...
        optimizer.step()
        scheduler.step()   # ← one step per batch, total = 30 × N steps
```

**Symptom:** The LR never reaches its intended shape. The warmup and decay are spread over 30 epochs instead of 30 × N steps, meaning the peak LR is hit on the very last epoch instead of at 30% of training.

[(Back to top)](#table-of-contents)

---

# Part 6: Summary Table

| Scheduler                     | Best For                           | Call Per  | Needs metric? | Warmup built-in?     | Key param                   |
| ----------------------------- | ---------------------------------- | --------- | ------------- | -------------------- | --------------------------- |
| `ReduceLROnPlateau`           | Unknown run length; general models | Epoch     | ✓ Yes         | ✗ No                 | `patience`                  |
| `CosineAnnealingLR`           | Known epochs; smooth decay         | Epoch     | ✗ No          | ✗ No                 | `T_max, eta_min`            |
| `StepLR`                      | Fixed drop schedule (e.g., ResNet) | Epoch     | ✗ No          | ✗ No                 | `step_size, gamma`          |
| `OneCycleLR`                  | Fast convergence with SGD          | **Batch** | ✗ No          | ✓ Yes                | `max_lr`                    |
| `CosineAnnealingWarmRestarts` | Long runs; escaping local minima   | Epoch     | ✗ No          | ✗ No                 | `T_0, T_mult`               |
| `LinearLR`                    | Warmup phase only                  | Epoch     | ✗ No          | ✓ Yes                | `start_factor, total_iters` |
| `SequentialLR`                | Warmup + cosine (transformers)     | Epoch     | ✗ No          | ✓ Yes (via LinearLR) | `milestones`                |

**Key takeaways:**

1. **Always call `optimizer.step()` before `scheduler.step()`** — wrong order produces off-by-one LR values.
2. **Always set `min_lr` / `eta_min`** — without a floor, LR can decay to zero and silently freeze training.
3. **Use warmup** for any model with large embedding tables or transformer layers.
4. **`OneCycleLR` calls `step()` per batch**, not per epoch — this is the single most common mistake with this scheduler.
5. **Separate weight decay from LR** by using `AdamW` with an explicit `weight_decay` argument.
6. **If using `ReduceLROnPlateau` with early stopping**, set early stopping patience at least 2× the scheduler patience.
7. **The safest defaults for most recommender models:** `AdamW` + `SequentialLR(LinearLR warmup → CosineAnnealingLR)`.

[(Back to top)](#table-of-contents)
