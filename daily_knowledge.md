# Daily Knowledge

# Day 2
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
- `torch.view()` to reshape the tensor
```Python
x = torch.randn(2, 3, 4)
# > torch.Size([2, 3, 4])
x = x.view(-1)
# > torch.Size([24]) # 2*3*4 = 24

# keep the first dimension, (-1) flatten the rest
x = x.view(x.size(0), -1) 

```
