# Daily Knowledge

# Day 1
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
- `.squeeze()`: returns a tensor with all the dimensions of **input of size 1 removed**.
```Python
print(img.size()) # (1, 28,28)
img.squeeze()     # (28,28)
```
