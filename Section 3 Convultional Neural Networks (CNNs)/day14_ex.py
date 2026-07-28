## Import Pytorch - provides tools for building neural networks
import torch
## torchvision provides access to popular image datasets, transforms contains utilities to preprocess and transform image data
from torchvision import datasets, transforms
## DataLoader is used to load data in batches
from torch.utils.data import DataLoader
## torch.nn is used for building neural networks
import torch.nn as nn
## torch.nn.functional is used for applying activation functions
import torch.nn.functional as F
## torch.optim is used for optimizing the model
import torch.optim as optim

# Define transformation with data augmentation
## This transformation trains the model by applying random transformations to the training data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),                          # Random horizontal flip
    transforms.RandomCrop(32, padding=4),                       # Random crop with padding
    transforms.ToTensor(),                                      # Convert to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      # Normalize the image - subtract mean and divide by standard deviation
])

# Define transformation without data augmentation
## This transformation evaluates the model on the test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

# Create data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print dataset sizes
print(f"Training Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")

# Define a CNN Architecture
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Define a batch normalization layer
        self.bn1 = nn.BatchNorm2d(6)
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define a batch normalization layer
        self.bn2 = nn.BatchNorm2d(16)
        # Define a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Define a dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of the output from the convolutional layers dynamically
        self._calculate_conv_output()
        
        # Define the fully connected layers
        ## first fully connected layer
        self.fc1 = nn.Linear(self.conv_output_size, 120)
        ## second layer - converts the input size to the output size. We're transforming 120 neurons and 84 classes
        self.fc2 = nn.Linear(120, 84)
        ## third layer - converts the input size to the output size. We're transforming 84 neurons and 10 classes
        self.fc3 = nn.Linear(84, 10)
    
    def _calculate_conv_output(self):
        # Dummy input tensor with the same size as the input images
        dummy_input = torch.zeros(1, 3, 32, 32)     # Size, channels and height and width
        with torch.no_grad():
            output = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy_input)))))))
        self.conv_output_size = output.numel()

    # Define the forward pass
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))                 # Apply ReLU activation
        x = self.pool(F.relu(self.bn2(self.conv2(x))))      # Apply max pooling
        x = x.view(x.size(0), -1)                           # Flattening the tensor dynamically
        x = F.relu(self.fc1(x))                             # Fully connected layer
        x = self.dropout(x)                                 # Apply dropout
        x = F.relu(self.fc2(x))                             # Fully connected layer
        x = self.fc3(x)                                     # output layer
        return x

# Create the model
model = EnhancedCNN()
# Print the model
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training_loss = []

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
# Evaluate the model
evaluate_model(model, test_loader)

import matplotlib.pyplot as plt

# Plot the loss curve
plt.plot(training_loss, label="Training Loss")
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()