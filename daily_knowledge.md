# Daily Knowledge

## Day 4

### Cross-Entropy, Negative Log-Likelihood

#### Formulas

- **BCE (Binary Negative Log-Likelihood)** (binary, `y ∈ {0,1}`): `-(1/N) Σ [y·log(p) + (1-y)·log(1-p)]`
<p align="center"><img src="./assets/img/binary-negative-log-likelyhood.gif" width=500></p>

- **NLL (Negative Log-Likelihood)** (multi-class): `-log(p_c)` where `c` is the true class
<p align="center"><img src="./assets/img/multi-class-nll.gif" width=500></p>

- **CrossEntropyLoss** fuses LogSoftmax + NLLLoss internally, so it takes raw logits. Numerically stable for the same reason as `BCEWithLogitsLoss`.
  - Relationship: `nn.CrossEntropyLoss(logits) ≡ nn.NLLLoss(log_softmax(logits))`

#### Common Mistake vs Best Practice

- **Binary Classification**:
  - **During Training**: Use `BCEWithLogitsLoss` because it is more stable and robust.
  - **During Inference**: If you need the actual probability (e.g., to show a confidence score or apply a custom threshold), you must manually apply `torch.sigmoid()` to the model's output.
  - Why `BCEWithLogitsLoss` ? `nn.BCEWithLogitsLoss()(logits, target)` internally uses the log-sum-exp trick:
    - `loss = max(x, 0) - x*y + log(1 + exp(-|x|))` this avoids computing sigmoid then log separately, which is numerically safe for large or small logit values (e.g. `sigmoid(100) ≈ 1.0` in float32, making `log(1 - 1.0) = -inf`).

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

- **Multi-Class Classification**: `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss` fused into one numerically stable operation:

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
