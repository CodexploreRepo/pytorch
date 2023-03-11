# Daily Knowledge

# Day 1
- `.permute()`: returns a view of the original tensor input with its dimensions permuted.
```Python
print(img.size())   # (3, 256, 256) - (C, H, W)
img.permute(1,2,0)  # (256, 256, 3) - (H, W, C)
```
- `.squeeze()`: returns a tensor with all the dimensions of **input of size 1 removed**.
```Python
print(img.size()) # (1, 28,28)
img.squeeze()     # (28,28)
```
