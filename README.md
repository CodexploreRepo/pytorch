# Pytorch

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Basics](#1-basics)
- [2. Backpropagation](#2-backpropagation)
- [3. Training Pipeline: Model, Loss, and Optimizer](#3-training-pipeline)


# 1. Basics

# 2. Backpropagation
1. **Forward Pass** &#8594; Compute Loss
  <p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/159221417-891dd988-942c-4330-8f60-c15280a0342d.png" width="400" />
  </p>

2. Compute Local Gradients: `dLoss/ds` &#8594; `ds/dy_hat` &#8594; `dy_hat/dw` to bring to *Step 3*
3. **Backward Pass**: compute `dLoss/dWeigths` using Chain Rules of Local Gradients
  <p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/159222006-b80f63ab-a002-436b-adf2-1f25878764d2.png" width="400" />
  </p>

- Note: only interested to compute  `dLoss/dw` to update the weights, but not `dLoss/x` or `dLoss/y`
```Python
x = torch.tensor(1.0)
y = torch.tensor(2.0)

#requires_grad=True > as only interest to compute dLoss/dw
w = torch.tensor(1.0, requires_grad=True) 

#Forward Pass and compute Loss
y_hat = w*x
loss = (y_hat - y)**2
print(f"Loss: {loss}") #1.0

#Backward Pass: to compute gradient dLoss/dw
# call .backward() and have all the gradients computed automatically.
loss.backward()

# The gradient for this tensor will be accumulated into .grad attribute.
# It is the partial derivate of the function w.r.t. the tensor
print(w.grad) # dLoss/dw = -2

# Update weights: this operation should not be part of the computational graph
with torch.no_grad():
    w -= 0.01 * w.grad
# don't forget to zero the gradients
w.grad.zero_()

### Next Forward and backward iterations
```
# 3. Training Pipeline



