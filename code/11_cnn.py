import torch
import torch.nn as nn
import torch.nn.functional as F #for activation funciton
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
#Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#Transform: dataset has PIL Image images of range [0,1].
# we transform them to Tensors of normalized range [-1,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
])

#Download dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
#Data Loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
#How to determin the output Size of Conv layers
#get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next() #get the first batch of images

conv1 = nn.Conv2d(3,6,5) #Channel (depth) In - Channel (depth) Out, Kernel Size
pool = nn.MaxPool2d(2,2) #kernel Size - Stride
conv2 = nn.Conv2d(6,16,5)
print(f"Original Image \t: {images.shape}")
x = conv1(images)
print(f"Conv1 Layer \t: {x.shape}")
x = pool(x)
print(f"Pool Layer \t: {x.shape}")
x = conv2(x)
print(f"Conv2 Layer \t: {x.shape}") 
x = pool(x)
print(f"Pool Layer \t: {x.shape}") #Final Output from ConvNet:  [16, 5, 5] => Flatten = 16*5*5
"""

#Create ConvNet Class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) #Channel (depth) In - Channel (depth) Out, Kernel Size
        self.pool = nn.MaxPool2d(2,2) #kernel Size - Stride
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(16*5*5, 120) #Input = (Flatten from Conv 16*5*5), Output: 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))
        
    def forward(self, x):
        # ->  n,3,32,32
        x = self.pool(F.relu(self.conv1(x))) # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x))) # -> n, 16, 5, 5
        x = x.view(-1, 16*5*5)               # -> n, 16*5*5 = 400
        x = F.relu(self.fc1(x))              # -> n, 120
        x = F.relu(self.fc2(x))              # -> n, 84
        x = self.fc3(x)                      # -> n, 10
        
        return x
    

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
#         # input_layer: 3 input channels, 6 output channels, 5 kernel size
#         images = images.to(device)
#         labels = labels.to(device)
        
#         #Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         #Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1)%2000 == 0:
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/ {n_total_steps}], Loss: {loss.item():.4f}")
    
# print('Finished Training')
FILE = './model/cnn.pth'
# torch.save(model.state_dict(), FILE)


model = ConvNet().to(device)
model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        #Move pred and labels to CPU
        #detach(): remove requires_grad from x
        outputs = outputs.detach().to('cpu').numpy() 
        labels = labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(outputs)
        true_labels.append(labels)
        
    print('Eval DONE.')


# Combine the results across all batches. 
# predictions: list of predictions per batch
flat_predictions = np.concatenate(predictions, axis = 0) 
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)


print(classification_report(flat_true_labels, flat_predictions, target_names=classes))