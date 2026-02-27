# DataLoader: pin_memory, num_workers & Sharding

## Table of Contents

- [pin_memory](#pin_memory)
  - [How it works](#how-it-works)
  - [Async transfers](#async-transfers)
  - [When it helps / hurts](#when-it-helps--hurts)
- [num_workers](#num_workers)
  - [Without num_workers](#without-num_workers)
  - [With num_workers](#with-num_workers)
  - [How they work together](#how-they-work-together)
  - [Choosing the right value](#choosing-the-right-value)
- [IterableDataset](#iterabledataset)
  - [The sharding problem](#the-sharding-problem)
  - [What happens without sharding](#what-happens-without-sharding)
- [Sharding](#sharding)
  - [What sharding means](#what-sharding-means)
  - [Three sharding strategies](#three-sharding-strategies)
  - [Correct sharding rule](#correct-sharding-rule)
- [Complete Code Pattern](#complete-code-pattern)

---

## pin_memory

`pin_memory=True` in `DataLoader` allocates CPU tensors in **pinned (page-locked) memory**, enabling faster host-to-GPU transfers.

### How it works

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

### Async transfers

Pinned memory enables **non-blocking** GPU transfers via a separate CUDA DMA stream:

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

### When it helps / hurts

| Situation                             | Effect                                       |
| ------------------------------------- | -------------------------------------------- |
| GPU training with `num_workers > 0`   | Significant speedup                          |
| Small datasets that fit in GPU memory | Minimal benefit                              |
| CPU-only training                     | Overhead — avoid it                          |
| MPS (Apple Silicon)                   | No benefit — pinned memory is a CUDA concept |

> **Note:** Pinned memory is a limited OS resource. Don't over-allocate — keep `num_workers` modest (2–4).

---

## num_workers

`num_workers` controls how many **subprocess workers** load and preprocess data in parallel, preventing the CPU from bottlenecking the GPU.

### Without num_workers

Everything runs in one process — load, preprocess, and train happen sequentially:

```
Main Process:
  [Load B1][Preprocess B1][Transfer B1][GPU Compute B1][Load B2][Preprocess B2]...
GPU:
                                        [====Compute B1====]      ↑ GPU idle here
```

### With num_workers

Workers **prefetch** batches in the background while the GPU computes the current one:

```
Worker 1:  [Load B1]──[Preprocess B1]──▶ Queue
Worker 2:  [Load B2]──[Preprocess B2]──▶ Queue
Worker 3:  [Load B3]──[Preprocess B3]──▶ Queue
Worker 4:  [Load B4]──[Preprocess B4]──▶ Queue
                                            │
                                            ▼
Main:                               [Batch Queue]──[Transfer]──[GPU Compute]
```

By the time the GPU finishes batch 1, batch 2 is already queued.

### How they work together

`num_workers` + `pin_memory` form a three-stage pipeline:

```
Stage 1 — CPU Workers (parallel subprocesses)
┌──────────────────────────────────────────┐
│ Worker 1: read → decode → transform      │──┐
│ Worker 2: read → decode → transform      │──┤──► Pinned Memory Queue
│ Worker 3: read → decode → transform      │──┘
└──────────────────────────────────────────┘         │ non_blocking=True
                                                      ▼
Stage 2 — DMA Transfer (async, separate CUDA stream)
                              ┌────────────────────────────┐
                              │  DMA: Pinned RAM → GPU VRAM │
                              └────────────────────────────┘
                                             │
Stage 3 — GPU Compute                        ▼
                              ┌────────────────────────────┐
                              │   CUDA Kernels / forward   │
                              └────────────────────────────┘
```

All three stages run **concurrently**.

### Choosing the right value

```
num_workers=0   → single process, no parallelism
num_workers=1   → one background worker, light parallelism
num_workers=2–4 → sweet spot for most workloads
num_workers=8+  → only if disk I/O is the bottleneck (many large images)
```

**Rule of thumb:** start with `num_workers = num_cpu_cores / 2`. Benchmark to find the optimal value:

```python
import time

for nw in [0, 1, 2, 4, 8]:
    loader = DataLoader(dataset, batch_size=64, num_workers=nw, pin_memory=True)
    start = time.time()
    for batch in loader:
        pass
    print(f"num_workers={nw}: {time.time() - start:.2f}s")
```

---

## IterableDataset

### The sharding problem

With a map-style `Dataset`, the DataLoader automatically splits indices across workers:

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

### What happens without sharding

The DataLoader pulls from all workers interleaved:

```
Dataset: [A, B, C, D, E, F, G, H]   num_workers=2, batch_size=4

Batch 1: [A, B, A, B]   ← 2 from Worker0, 2 from Worker1
Batch 2: [C, D, C, D]
Batch 3: [E, F, E, F]
Batch 4: [G, H, G, H]
```

Every sample appears `num_workers` times per epoch. **No error or warning is raised** — training appears normal while the model trains on duplicated data.

Detect it by inspecting batch contents:

```python
seen = []
for batch in DataLoader(dataset, batch_size=4, num_workers=2):
    seen.extend(batch.tolist())

print(len(seen))           # 16 — looks right
print(len(set(seen)))      # 8  — only 8 unique values, duplication confirmed
```

---

## Sharding

### What sharding means

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

### Three sharding strategies

**1. By remainder (interleaved)**

```python
yield from (x for i, x in enumerate(data) if i % num_workers == worker_id)

Worker 0: [A, C, E, G]   (items 0, 2, 4, 6)
Worker 1: [B, D, F, H]   (items 1, 3, 5, 7)
```

**2. By contiguous block**

```python
chunk = len(data) // num_workers
start = worker_id * chunk
end   = start + chunk
yield from data[start:end]

Worker 0: [A, B]
Worker 1: [C, D]
Worker 2: [E, F]
Worker 3: [G, H]
```

**3. By file (for file-based streaming)**

```
files = [file1, file2, file3, file4]

Worker 0: reads file1
Worker 1: reads file2
Worker 2: reads file3
Worker 3: reads file4
```

### Correct sharding rule

```
Union of all shards     = full dataset   (no gaps)
Intersection of shards  = empty          (no overlaps)

Worker 0: {A, C, E, G}
Worker 1: {B, D, F, H}

Union:        {A, B, C, D, E, F, G, H}  ✓
Intersection: {}                         ✓
```

---

## Complete Code Pattern

```python
from torch.utils.data import IterableDataset, DataLoader
import torch

class StreamingDataset(IterableDataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # num_workers=0 — yield everything
            start, end = 0, len(self.data)
        else:
            # num_workers > 0 — assign a contiguous block to each worker
            total      = len(self.data)
            n_workers  = worker_info.num_workers
            wid        = worker_info.id
            chunk      = total // n_workers
            remainder  = total % n_workers

            # First `remainder` workers get one extra sample
            if wid < remainder:
                start = wid * (chunk + 1)
                end   = start + chunk + 1
            else:
                start = wid * chunk + remainder
                end   = start + chunk

        for idx in range(start, end):
            yield self.data[idx]


# DataLoader
loader = DataLoader(
    StreamingDataset(data),
    batch_size=32,
    num_workers=2,
    pin_memory=True,           # lock tensors in RAM for fast DMA transfer
    persistent_workers=True,   # keep workers alive between epochs
)

# Training loop
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for features, labels in loader:
    features = features.to(device, non_blocking=True)   # async transfer
    labels   = labels.to(device, non_blocking=True)
    # forward / backward ...
```

### Key rules summary

| Rule                      | Detail                                                        |
| ------------------------- | ------------------------------------------------------------- |
| `pin_memory=True`         | Pairs with `non_blocking=True` for async GPU transfers        |
| `num_workers > 0`         | Always shard `IterableDataset` using `get_worker_info()`      |
| `persistent_workers=True` | Avoids worker respawn cost between epochs                     |
| MPS (Apple Silicon)       | `pin_memory` is safe but has no effect — skip for clarity     |
| Too many workers          | Competes for CPU cache and memory bandwidth — benchmark first |

> **Full code example:** `code/tutorials/14_iterable_dataset_sharding.ipynb`
