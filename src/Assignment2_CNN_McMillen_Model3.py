# Add Batch Normalization before ReLU 
# + 
# Add Dropout before fully connected layer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import time

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)   	# 3x32x32 -> 16x32x32
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)                           		# -> 16x16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)  	# -> 32x16x16
        self.bn2   = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1   = nn.Linear(32 * 8 * 8, 10)                    		# after 2nd pool: 32x8x8

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   	# [B, 16, 32, 32]
        x = self.pool(x)               		# [B, 16, 16, 16]
        x = self.relu(self.bn2(self.conv2(x)))   	# [B, 32, 16, 16]
        x = self.pool(x)               		# [B, 32,  8,  8]
        x = x.view(x.size(0), -1)      	# [B, 32*8*8]
        x = self.dropout(x)
        x = self.fc1(x)                		# [B, 10] (logits)
        return x


# Transform
transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))])

# Batch Size
batch_size = 64

# Train and Test data set and loaders
train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, test_loader, num_epochs=10, lr=1e-3):
  start_time = time.perf_counter()
  criterion = nn.CrossEntropyLoss()
  import torch.optim as optim
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(num_epochs):
      model.train()
      for images, labels in train_loader:
          # forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          # backpropagation
          loss.backward() # calculates loss
          optimizer.step() # apply adjustments to weights and bias
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

  # evaluate model and perform forward pass with the test loader, record accuracy
  model.eval()
  all_preds, all_labels = [], []
  with torch.no_grad():
      for images, labels in test_loader:
          outputs = model(images)
          _, predicted = torch.max(outputs, dim=1)
          all_preds.extend(predicted.tolist())
          all_labels.extend(labels.tolist())

  acc = accuracy_score(all_labels, all_preds)
  print(f"Test Accuracy: {acc * 100:.2f}%")
  end_time = time.perf_counter()
  elapsed_time = (end_time - start_time)/60
  print(f"Elapsed time: {elapsed_time:.4f} minutes")

model = CNN()
train_model(model, train_loader, test_loader)