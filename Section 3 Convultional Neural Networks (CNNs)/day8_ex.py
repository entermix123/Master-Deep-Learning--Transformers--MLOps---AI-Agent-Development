# install libraries
## pip install torchvision matplotlib numpy tensorflow

# import libraries
## used for creating visualizations such as charts and plots
import matplotlib.pyplot as plt
## torchvision provides access to popular image datasets 
## Data sets provides access to the popular datasets like CIFAR ten, which we are going to use, or mNIST and others
## Transforms contains utilities to preprocess and transform image data such as converting images to tensors
from torchvision import datasets, transforms
## matrix operation and mathematical function
import numpy as np

# # Load Dataset
# ## Defines a transformation to convert images from the data set into PyTorch tensors. This is necessary for using the data in PyTorch models.
# transform = transforms.ToTensor()
# ## Load the CIFAR10 dataset
# ## CIFAR10 classifies all the images into ten different classes from 0 to 9. zero is for airplane, one is for automobile, two is for bird, then cat, then deer, dog, frog, horse, ship and truck.
# ## root='./data' - specifies the directory where the data set will be stored
# ## train=True - loads the training split of the dataset
# ## transform=transform - applies the defined transformation to the images
# ## download=True - Download the data set if it's not already present in the specified directory
# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

# # Visualize sample images
# ## creates a figure with one row and five columns of subplots for displaying images.
# ## fixed size defines the size of the figure as 12in wide and 3 inches in height.
# fig, axes = plt.subplots(1, 5, figsize=(12,3))
# ## iterates over the first five images in the dataset
# for i in range(5):
#     ## fetches the image and label at index i from the dataset
#     image, label = train_dataset[i]
#     ## displays the image using the imshow function
#     ## inside that function we are calling the permute method which reorders the dimensions from channels, height and width to height and width and channels
#     ## So usually it's 0, 1, 2. But we are saying 1, 2, 0 because we want to move the channels to the end height, We are going to move to the first and then width we want to move to second. So this moves them around and accordingly gets us the data.
#     axes[i].imshow(image.permute(1, 2, 0))
#     ## hide the axis for a cleaner display of the image. Don't show the axis in the graph.
#     axes[i].axis('off')
#     ## set the title for each subplot
#     axes[i].set_title(f"Label: {label}")
# ## show the plot
# plt.show()

# # Display pexel values for the first image
# image, label = train_dataset[0]
# print(f"Label: {label}")
# print(f"Image Shape: {image.shape}")
# print("Pixel Values:")
# print(image)

## Tensorflow - building and training deep learning models
import tensorflow as tf

# Define a simple CNN (convolutional neural network) model
## This is a sequential model, which means each layer is run one at a time in sequential order
model = tf.keras.Sequential([
    ## Conv2D is a convolutional layer with 32 filters, a 3 by 3 kernel size and a ReLU activation function for the image we have
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    ## This is a max pooling 2D layer with two by two pooling window to reduce the spatial dimensions of the previous input
    tf.keras.layers.MaxPooling2D((2, 2)),
    ## This function flattens the 2D feature maps into a 1D vector for fully connected layers
    tf.keras.layers.Flatten(),
    ## Fully connected layer with 128 neurons and ReLU activation function
    tf.keras.layers.Dense(128, activation="relu"),
    ## Output layer with 10 neurons (ten different classification units we have in CIFAR ten) and softmax activation function
    ## activation="softmax" - activation for classification
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
## optimizer='adam' - optimization algorithm for training the model
## loss='sparse_categorical_crossentropy' - specifies the loss function for multi-class classification
## metrics=['accuracy'] - metric for evaluating the model - tracks accuracy during the training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## print message when the model is ready
print("Tensorflow CNN Model is ready")


## Pytorch - provides tools for building neural networks
import torch.nn as nn

# Define a simple CNN model
class SimpleCNN(nn.Module):
    ## initializer for the class
    def __init__(self):
        ## call the parent class constructor
        super(SimpleCNN, self).__init__()
        ## create a convolutional layer with 3 input channels, 32 filters and a 3x3 kernel size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, activation='relu')
        ## create a max pooling layer with a 2 by 2 pooling window
        self.pool = nn.MaxPool2d(2, 2)
        ## create first fully connected layer, transforming it from 32 * 15 * 15 into 128
        self.fc1 = nn.Linear(32 * 15 * 15, 128)
        ## second layer - converts the input size to the output size. We're transforming 128 neurons and ten classes.
        self.fc2 = nn.Linear(128, 10)
        
    ## This defines the forward pass
    def forward(self, x):
        ## This applies convolution conv one and ReLU activation
        x = F.relu(self.conv1(x))
        ## Apply max pooling
        x = self.pool(x)
        ## Flatten - converts the 2D feature maps into a 1D vector
        x = x.view(-1, 32 * 15 * 15)
        ## next two lines passes the data through fully connected layers of FC1 and FC2
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

# print message when the model is ready
print("PyTorch CNN model ready")
