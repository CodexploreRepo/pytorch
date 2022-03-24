# Pytorch

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Basics](#1-basics)
- [2. Backpropagation](#2-backpropagation)
- [3. Training Pipeline: Model, Loss, and Optimizer](#3-training-pipeline)
  - [3.1. Example of Training Pipeline](#31-example-of-training-pipeline) 
- [4. Dataset & DataLoader](#4-dataset-and-dataloader)
  - [4.1. Dataset](#41-dataset)
  - [4.2. Dataloader](#42-dataloader) 
- [Resources](#resources)

# 1. Basics

[(Back to top)](#table-of-contents)

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

[(Back to top)](#table-of-contents)

# 3. Training Pipeline
- Step 1: Design model 
  - Input: ROW = Number of Training instances; COL = Number of Features
  - Output
  - Forward pass with different layers
- Step 2: Construct `loss` and `optimizer` (init weigths or model parameters)
- Step 3: Training loop
  - Forward = compute prediction and loss
  - Backward = compute gradients
  - Update weights

## 3.1. Example of Training Pipeline
#### Step 0: Load Data & Convert to Py Torch format
```Python
#0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
#print(n_samples, n_features) #569 samples with 30 features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

#Convert to Torch
X_train = torch.from_numpy(X_train.astype(np.float32)) #need to convert to float 
y_train = torch.from_numpy(y_train.astype(np.float32)) #need to convert to float
X_test = torch.from_numpy(X_test.astype(np.float32))   #need to convert to float
y_test = torch.from_numpy(y_test.astype(np.float32))   #need to convert to float

#Convert y into from [0,1,2,3] ==> [[0], [1], [1]]
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
```
#### Step 1: Design Model
```Python
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features, output_size):
        #super(Model, self).__init__() = super().__init__(self)
        #this allows to init the model without any input paras: model = LogisticRegression()
        super(LogisticRegression, self).__init__()
        
        #First f = wx + b, then apply sigmoid at the end
        self.linear = nn.Linear(n_input_features, output_size)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

input_size = n_features
output_size = 1

model = LogisticRegression(n_features, output_size)
```
#### Step 2: Design Model
```Python
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss() #binary cross-entropy
optimmizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
#### Step 3: Training Loop
```Python

for epoch in range(num_epochs):
    #forward pass & loss
    y_predicted = model.forward(X_train)
    loss = criterion(y_predicted, y_train)
    #backward pass
    loss.backward()

    #weight update
    optimmizer.step()
    optimmizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3,}: Loss: {loss.item():.3f}")
```

#### Step 4:  Evaludation
```Python
with torch.no_grad(): #to ensure this is not in the computation graph
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() #Since sigmoid return btw 0~1
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f"Accuracy = {acc:.4f}")
```

[(Back to top)](#table-of-contents)

# 4. Dataset and DataLoader
## 4.1. Dataset
- Import Dataset Class from Pytorch Utils: `from torch.utils.data import Dataset`
```Python
class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1) #skiprows=1 skip header row
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # [0]  : (n_samples, 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

```

## 4.2. DataLoader
- Instead of updating the entire dataset per epoch, we can split the dataset into small batches.
- Say, `1 epoch` with `batch_size = 4`, we can have ` math.ceil(total_samples/batch_size)` update iterations to compute the loss, gradient and update the weights.
```Python
batch_size = 4
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2,
                        )
                        
num_epochs = 2
total_samples = len(dataset)
#n of update iterations per epoch
n_iterations = math.ceil(total_samples/batch_size) 


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iterations}, inputs: {inputs.shape[0]} samples")

#epoch 1/2, step 5/45, inputs: 4 samples
#epoch 1/2, step 10/45, inputs: 4 samples
#epoch 1/2, step 15/45, inputs: 4 samples
```

[(Back to top)](#table-of-contents)


# Resources
## Todo List
- [Hyperparameter Tuning of Neural Networks with Optuna and PyTorch](https://towardsdatascience.com/hyperparameter-tuning-of-neural-networks-with-optuna-and-pytorch-22e179efc837) 
- 
[(Back to top)](#table-of-contents)
