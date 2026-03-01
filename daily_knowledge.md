# Daily Knowledge

## Day 7

### Vectorize the Dataset - The Biggest Win

**Problem**: The original `IntentIterableDataset.__iter__` processed rows one at a time:

```python
# SLOW: ~10s per step
for idx in range(len(df)):
    row = df.iloc[idx:idx+1]                          # pandas slice per row
    num = self.num_preprocessor.transform(row, ...)    # sklearn on 1 row
    cat = self.cat_preprocessor.transform(row, ...)    # .apply(lambda) per cell
    yield torch.tensor(...)                           # individual tensor creation

```

For every sample: a 1-row DataFrame slice, a `StandardScaler.transform()` call on 1 row, a `series.apply(lambda x: encoder.transform([x]))` loop, and individual tensor creation. The DataLoader then re-collated these back into batches.

**Fix**: Process entire PyArrow chunks at once and yield pre-batched tensors:

```python
# FAST: ~55ms per step
df = batch.to_pandas()                                # chunk of ~4K rows
num = self.num_preprocessor.transform(df, ...)        # vectorized numpy
cat = self.cat_preprocessor.transform(df, ...)        # vectorized series.map
all_features = np.concatenate([...], axis=1)          # single C call
yield torch.from_numpy(all_features[start:end])       # zero-copy

```

**Why it matters**: Vectorized numpy/pandas operations run in compiled C/Fortran code. One call processing 4000 rows is orders of magnitude faster than 4000 Python-level calls processing 1 row each. SIMD instructions, cache-friendly memory access, and zero Python interpreter overhead.

**Impact**: ~100x speedup on data loading.

### Vectorize Categorical Encoding

**Problem**: `CategoricalPreprocessor.transform` used `.apply(lambda)`:

```python
def transform(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
    if not self.fitted:
        raise ValueError("Preprocessor not fitted. Call fit() first.")

    result = []
    for col in columns:
        series = data[col].fillna(self.impute_value).astype(str)
        encoder = self.encoders[col]
        # SLOW: Python function called per row, sklearn called per value
        encoded = series.apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else ...
        )

```

For 4000 rows x 15 categorical features = 60,000 Python function calls, each invoking sklearn.

**Fix**: Dict-based `series.map()`:

- Reference: [preprocessor.py](./code/topics/dataset/preprocessor.py)

```python
# FAST: Cython loop with hash table lookups
class_to_idx = {cls: idx for idx, cls in enumerate(encoder.classes_)}
encoded = series.map(class_to_idx).fillna(fallback).astype(int).values

```

`series.map(dict)` runs in Cython inside pandas — hash lookups in C, not Python.

**Impact**: ~400x speedup on categorical encoding alone.

## Day 6

### `pin_memory`

`pin_memory=True` in `DataLoader` allocates CPU tensors in **pinned (page-locked) memory**, enabling faster host-to-GPU transfers.

#### How it works

Normally, CPU RAM is **pageable** — the OS can swap it to disk. GPU DMA engines can't read pageable memory directly, so PyTorch must:

1. Copy data to a temporary pinned buffer
2. DMA transfer from pinned buffer → GPU

With `pin_memory=True` the tensor is already pinned, so step 1 is skipped.

```
Without pin_memory:               With pin_memory:

Pageable RAM                      Pinned RAM
┌─────────────┐                   ┌─────────────┐
│  Batch data │                   │  Batch data │◄── OS cannot page this out
│  (movable)  │                   │  (locked)   │
└──────┬──────┘                   └──────┬──────┘
       │ CPU must first                  │ DMA engine reads directly
       ▼ copy to temp buffer             ▼
┌─────────────┐                   ┌─────────────┐
│Pinned buffer│                   │  GPU Memory │
└──────┬──────┘                   └─────────────┘
       │ then DMA
       ▼
┌─────────────┐
│  GPU Memory │
└─────────────┘
```

### Async transfers (`non_blocking=True`)

**Pinned memory** enables **non-blocking** GPU transfers via a separate CUDA DMA stream:

```python
# Without pin_memory — transfer blocks the CPU
batch = batch.to(device)

# With pin_memory — transfer runs asynchronously on a separate CUDA stream
batch = batch.to(device, non_blocking=True)
```

**Without pin_memory (synchronous):**

```
CPU: [Load B1][Copy to pinned][====idle====][Load B2][Copy to pinned][====idle====]
GPU:                          [Transfer B1][===Compute B1===]        [Transfer B2][===Compute B2===]
```

**With pin_memory + non_blocking (async):**

```
CPU:        [Load B1][====Prepare B2====][====Prepare B3====]
DMA Stream:          [--Transfer B1--]  [--Transfer B2--]   [--Transfer B3--]
GPU Compute:                            [===Compute B1===]  [===Compute B2===]
```

CPU loading, DMA transfers, and GPU compute all run **concurrently**.

#### Pin Memory with Manual (Dataset) Batching (i.e. `batch_size=None`)

Since `batch_size=None` bypasses the DataLoader's built-in pinning, we pin in the Trainer, instead of specifying in the Dataloader

```python
if self._use_pinned:
    features = features.pin_memory().to(device, non_blocking=True)
```

#### When it helps / hurts

| Situation                             | Effect                                       |
| ------------------------------------- | -------------------------------------------- |
| GPU training with `num_workers > 0`   | Significant speedup                          |
| Small datasets that fit in GPU memory | Minimal benefit                              |
| CPU-only training                     | Overhead — avoid it                          |
| MPS (Apple Silicon)                   | No benefit — pinned memory is a CUDA concept |

> **Note:** Pinned memory is a limited OS resource. Don't over-allocate — keep `num_workers` modest (2–4).

## Day 5 - IterableDataset

### Shuffle Strategy for Streaming Datasets

**Problem**: `IterableDataset` reads parquet partitions in the same order every epoch. The model sees identical batch sequences, which can hurt generalization.

**Fix**: Two-level approximate shuffle (true global shuffle of 60M rows is infeasible in streaming):

1. **Fragment-level shuffle**: Randomize the order of parquet files across all date partitions. This interleaves dates (June -> August -> July) rather than reading sequentially.
2. **Intra-chunk row shuffle**: `np.random.permutation` on the final numpy arrays after preprocessing.

**Key detail**: Shuffle the post-preprocessing numpy arrays, not the DataFrame. `df.sample(frac=1.0)` copies all 250+ columns. `np.random.permutation` on 3 small contiguous arrays is much cheaper (~14ms/step saved).

```python
# SLOW: copies entire wide DataFrame
df = df.sample(frac=1.0).reset_index(drop=True)

# FAST: permute index on final arrays only
idx = np.random.permutation(len(all_features))
all_features = all_features[idx]
all_targets = all_targets[idx]

```

### IterableDataset: Batching Strategies Summary

When using a [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html) with an `IterableDataset`, you have two primary architectural choices for how data is batched.

| Feature                | **Approach 1: Automatic Batching**   | **Approach 2: Manual (Dataset) Batching** |
| :--------------------- | :----------------------------------- | :---------------------------------------- |
| **DataLoader Config**  | `batch_size=INT` (e.g., 32)          | `batch_size=None`                         |
| **Dataset `__iter__`** | Yields **one sample** at a time      | Yields a **full batch** at a time         |
| **Collate Function**   | Active (stacks samples into tensors) | Bypassed (returns data as-is)             |
| **Best For**           | Standard data with uniform shapes    | Dynamic sizes, streaming, or "bulk" reads |
| **Complexity**         | Simple implementation                | Requires manual grouping logic            |

- Example 1: Automatic Batching

```Python
from torch.utils.data import IterableDataset, DataLoader

class MySingleExampleDataset(IterableDataset):
    def __iter__(self):
        # Yield only ONE example at a time
        for i in range(100):
            yield {"data": i, "label": i % 2}

dataset = MySingleExampleDataset()

# The DataLoader handles the batching logic for you
loader = DataLoader(dataset, batch_size=16)

for batch in loader:
    # 'batch' is now a dictionary of tensors with size 16
    print(batch["data"].shape) # torch.Size([16])
```

- Example 2: [Manual (Dataset) Batching](./code/topics/data/dataset.py)

### `num_workers > 0` - Not Worth It Here

`num_workers > 0` with `IterableDataset` requires manual sharding (each worker gets a full copy of the iterator by default). For this pipeline, the vectorized batch processing keeps CPU utilization high enough that multi-process overhead isn't justified. The bottleneck is GPU compute + I/O, not CPU preprocessing.

### The sharding problem with `IterableDataset` & `num_worker > 0`

With a normal `Dataset`, the DataLoader automatically splits indices across workers:

```
Map-style Dataset — DataLoader handles sharding automatically:

Worker 0 → indices [0, 4, 8, ...]
Worker 1 → indices [1, 5, 9, ...]   ✓ no duplicates
Worker 2 → indices [2, 6, 10, ...]
```

`IterableDataset` has no indices — each worker runs `__iter__` from the beginning:

```
IterableDataset — no sharding:

Worker 0 → streams ALL data [A, B, C, D, E, F, G, H]
Worker 1 → streams ALL data [A, B, C, D, E, F, G, H]   ✗ full duplication
Worker 2 → streams ALL data [A, B, C, D, E, F, G, H]
```

#### What happens without sharding

The DataLoader pulls from all workers interleaved:

```
Dataset: [A, B, C, D, E, F, G, H]   num_workers=2, batch_size=4

Batch 1: [A, B, A, B]   ← 2 (A,B) from Worker0, 2 (A,B) from Worker1
Batch 2: [C, D, C, D]
Batch 3: [E, F, E, F]
Batch 4: [G, H, G, H]
```

Every sample appears `num_workers` times per epoch. **No error or warning is raised** — training appears normal while the model trains on duplicated data. Hence increasing the training time as the result.

#### Sharding

##### What sharding means

Sharding = **splitting data into non-overlapping slices**, one per worker. Each worker is responsible for only its own slice.

```
Without sharding (num_workers=2)       With sharding (num_workers=2)

Dataset  [A B C D E F G H]            Dataset  [A B C D E F G H]
            ↓       ↓                              ↓       ↓
         W0      W1                            W0       W1
        [A-H]   [A-H]                         [A,C,    [B,D,
                                               E,G]     F,H]
            ↓       ↓                              ↓       ↓
        Queue: A,A,B,B,C,C...               Queue: A,B,C,D,E,F,G,H
               ↑ duplicates                        ↑ unique, complete
```

#### Code implementation in `IterableDataset` when num_worker > 0

- **Option 1**: using `get_worker_info()` when called in a worker process, returns information about the worker.
  - It can be used in either the dataset’s `__iter__()`

```Python

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

# Multi-process loading with two worker processes
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
```

- **Option 2**: `DataLoader` ‘s `worker_init_fn` option to modify each copy’s behavior.

```Python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
# Directly doing multi-process loading yields duplicate data
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

# Mult-process loading with the custom `worker_init_fn`
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
```

## Day 4

### Learning Rate Scheduler

- It is generally best to first train without a scheduler to establish a baseline.
- **Always set `min_lr` / `eta_min`** — without a floor, LR can decay to zero and silently freeze training.
- **Use warmup** for any model with large embedding tables or transformer layers.
- When using `ReduceLROnPlateau` and early stopping together, their `patience` values must be set deliberately:

```
Bad setup:
  ReduceLROnPlateau patience = 5
  EarlyStopping patience     = 5

  Both fire at the same epoch → no time for the reduced LR to help

Good setup:
  ReduceLROnPlateau patience = 3   ← fires sooner
  EarlyStopping patience     = 10  ← gives the reduced LR time to work
```

- Always Log the Current LR at every epoch so you can reconstruct what happened if training diverges.

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

### Cross-Entropy

#### Formulas

- **BCE (Binary Negative Log-Likelihood)** (binary, `y ∈ {0,1}`): `-(1/N) Σ [y·log(p) + (1-y)·log(1-p)]`
<p align="center"><img src="./assets/img/binary-negative-log-likelyhood.gif" width=500></p>

- **NLL (Negative Log-Likelihood)** (multi-class): `-log(p_c)` where `c` is the true class
<p align="center"><img src="./assets/img/multi-class-nll.gif" width=500></p>

- **CrossEntropyLoss** fuses LogSoftmax + NLLLoss internally, so it takes raw logits. Numerically stable for the same reason as `BCEWithLogitsLoss`.
  - Relationship: `nn.CrossEntropyLoss(logits) ≡ nn.NLLLoss(log_softmax(logits))`

#### Binary Classification `nn.BCELoss` vs `nn.BCEWithLogitsLoss`

- **During Training**: Use `BCEWithLogitsLoss` because it is more stable and robust.
- **During Inference**: If you need the actual probability (e.g., to show a confidence score or apply a custom threshold), you must manually apply `torch.sigmoid()` to the model's output.
- Why `BCEWithLogitsLoss` ? `nn.BCEWithLogitsLoss()(logits, target)` internally uses the log-sum-exp trick:
- `loss = max(x, 0) - x*y + log(1 + exp(-|x|))` this avoids computing sigmoid then log separately, which is numerically safe for large or small logit values (e.g. `sigmoid(100) ≈ 1.0` in float32, making `log(1 - 1.0) = -inf`) causing **NaN gradients**.

```Python
# logits + BCEWithLogitsLoss (Recommended)
model = nn.Sequential(nn.Linear(10, 1)) # No Sigmoid here

logits = model(input)
loss = nn.BCEWithLogitsLoss(logits, target)

# Sigmoid + BCELoss (Avoid)
model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
probs = model(inputs)
loss = nn.BCELoss(probs, target)
```

#### Multi-Class Classification `nn.NLLLoss` vs `nn.CrossEntropyLoss`

- `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss` fused into one numerically stable operation:

```
CE = -x_c + log(Σ exp(x_j))    # x = raw logits, c = true class
```

This avoids computing softmax then log separately, preventing overflow/underflow for large logits.

**Best Practice:** pass raw logits — do NOT apply softmax in `forward`:

```Python
import torch
import torch.nn as nn

# Suppose 3 classes

# Option 1: CrossEntropyLoss (Recommended)
criterion_ce = nn.CrossEntropyLoss()
loss_ce = criterion_ce(logits, target)

# Option 2: LogSoftmax + NLLLoss (Avoid)
log_softmax = nn.LogSoftmax(dim=1)
criterion_nll = nn.NLLLoss()
loss_nll = criterion_nll(log_softmax(logits), target)

# loss_ce and loss_nll are equal
```

## Day 3

### Torch Basics

#### Tensor Concatenation

- Create a torch tensor from numpy

```Python
arr = torch.from_numpy(np.array([[1,2,3],[4,5,6]]))
print(arr)
print(f"Shape: {arr.size()}")

# tensor([[1, 2, 3],
#         [4, 5, 6]])
# Shape: torch.Size([2, 3])
```

- Concat the array by `axis=0` (default option)

```Python
torch.cat((arr, arr), 0)
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [1, 2, 3],
#         [4, 5, 6]])
```

- Concat the array by `axis=1`

```Python
torch.cat((arr, arr), 1)
# tensor([[1, 2, 3, 1, 2, 3],
#         [4, 5, 6, 4, 5, 6]])
```

### GPU on Mac

```Python
has_gpu = torch.backends.mps.is_available()
device = torch.device("mps" if has_gpu else "cpu")
```

## Day 2

- Initialise the weights of the network

```Python
# Define the CNN architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define the network here
        self.conv1 = torch.nn.Conv2d(3, 6, 3)
        self.batch_norm1 = torch.nn.BatchNorm2d(6)

        # initialize weights
        self.initialize_weights()

    def forward(self, x):
        # define forward function here


    def initialize_weights(self):
        # initialize network weights
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

# Usage:
net = Net().to(device)
# train on some paramters
# re-train on another set of parameters
net.initialize_weights()
```

- Plot a batch of images using `torchvision.utils.make_grid`

```Python
def imshow(img, title):

    plt.figure(figsize=(batch_size * 4, 4))
    plt.axis('off')
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))

    img = torchvision.utils.make_grid(images)
    imshow(img, title=[str(x.item()) for x in labels])

    return images, labels
show_batch_images(train_dataloader)
```

![image](https://user-images.githubusercontent.com/64508435/224746587-115d1a79-5a4d-4af3-bf3a-4f2e8738d8d8.png)

## Day 1

### Torch Basics

- `torch.flatten(input, start_dim=0, end_dim=- 1)` flattens input by reshaping it into a one-dimensional tensor

```Python
torch.flatten(x, 1) # Before: [64, 16, 61, 61] -> After: [64, 16*61*61=59536]
```

- `.permute()`: returns a view of the original tensor input with its dimensions permuted.
  - `np.transpose()` also can be used for converting the torch tensor

```Python
print(img.size())   # (3, 256, 256) - (C, H, W)

# Method 1:
img.permute(1,2,0)  # torch tensor (256, 256, 3) - (H, W, C)

# Method 2:
np.transpose(img, (1,2,0))  # torch tensor (256, 256, 3) - (H, W, C)
```

- `.squeeze() vs unsqueeze()`
  - `.squeeze()` returns a tensor with all the dimensions of **input of size 1 removed**.
  - `.unsqueeze(input, dim)` Returns a new tensor with a dimension of size one inserted at the specified position.
    - dim (int) – the index at which to insert the singleton dimension

```Python
# ---- squeeze examples -----
print(img.size()) # (1, 28,28)
img.squeeze()     # (28,28)
# ---- unsqueeze examples -----
# unsqueeze(0) at new dimension at dim0 & unsqueeze(1) at new dimension at dim1 examples
# unsqueeze(0): (3, 256, 256) -> (1, 3, 256, 256)
# unsqueeze(1): (3, 256, 256) -> (3, 1, 256, 256)
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)

tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])

```

- `torch.view()` to reshape the tensor

```Python
x = torch.randn(2, 3, 4)
# > torch.Size([2, 3, 4])
x = x.view(-1)
# > torch.Size([24]) # 2*3*4 = 24

# (x.size(0)) keep the first dimension, (-1) flatten the rest
x = x.view(x.size(0), -1)
```
