# install libraries
## pip install matplotlib numpy scipy torch tensorflow

## essential for numerical computations and handling arrays
import numpy as np
## used for visualizing the feature maps
import matplotlib.pyplot as plt
## import maximum_filter for max pooling and uniform_filter for average pooling
from scipy.ndimage import maximum_filter, uniform_filter

# Create a sample feature map
## It defines 4x4 2D array feature map as a sample feature set
feature_map = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])

# Max pooling (2X2)
## We are performing max pooling with a 2x2 kernel size, and each region in the feature map is replaced with its maximum value
## mode='constant' to pad with zeros
## size=2 is the size of the pooling window
max_pooled = maximum_filter(feature_map, size=2, mode='constant')

# Average pooling (2X2)
## We perform average pooling with a 2x2 kernel size and each region is replaced with the average of its values
avg_pooled = uniform_filter(feature_map, size=2, mode='constant')

# # # Plot
# # ## creates a figure with 1 row and 3 columns for side by side visualization of the images and fixed size 12x12 four sets the size of the figure to 12in wide and 4in tall
# # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# # ## displays the original image on the first subplot with grayscale color map. So it takes the image and shows it in the first axis
# # axes[0].imshow(feature_map, cmap='viridis')
# # axes[0].set_title("Original Feature Map")
# # ## display the edge detected image on the second subplot
# # axes[1].imshow(max_pooled, cmap='viridis')
# # axes[1].set_title("Max Pooled")
# # ## display the blurred image on the third subplot
# # axes[2].imshow(avg_pooled, cmap='viridis')
# # axes[2].set_title("Average Pooled")
# # plt.show()


## Tensorflow - building and training deep learning models
import tensorflow as tf

# Create a sample input tensor (1X4x4X1 for batch size, height, width, channels)
## We've converted the feature map into a 4D tensor with dimensions of batch size, height, width and channels
input_tensor = tf.constant(feature_map.reshape(1, 4, 4, 1), dtype=tf.float32)

# Max Pooling
## max pool variable - Defining a 2x2 max pooling layer with stride of 2
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
## apply the pooling to the input tensor
max_pooled_tensor = max_pool(input_tensor)

# Avg Pooling
## avg pool variable - Defined a 2x2 average pooling layer with strides of 2
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
## apply average pooling to the input tensor
avg_pooled_tensor = avg_pool(input_tensor)

# Print results
print(f"Max Pooled Tensor:\n{tf.squeeze(max_pooled_tensor).numpy()}")
print(f"Average Pooled Tensor:\n{tf.squeeze(avg_pooled_tensor).numpy()}")
print("\n\n\n")

## import Pytorch
import torch
## Pytorch - provides tools for building neural networks
import torch.nn as nn

# Create a sample input tensor (batch_size, channels, height, width)
## We are converting the feature map to a 4D tensor with dimensions of batch size, channels, height and width and then we have to unsqueeze it twice.
input_tensor = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Max Pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
max_pooled_tensor = max_pool(input_tensor)

# Average Pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled_tensor = avg_pool(input_tensor)

# Print results
print(f"Max Pooled Tensor:\n{max_pooled_tensor.squeeze().numpy()}")
print(f"Average Pooled Tensor:\n{avg_pooled_tensor.squeeze().numpy()}")

# TensorFlow Example
## This is a sequential model, which means each layer is run one at a time in sequential order
model_tf = tf.keras.Sequential([
    ## set the shape first so no warnings are thrown
    tf.keras.Input(shape=(32, 32, 3)),
    ## Conv2D is a convolutional layer with 32 filters, a 3x3 kernel size and a ReLU activation function
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    ## Conv2D is a second convolutional layer with 64 filters, a 3x3 kernel size and a ReLU activation function
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D((2, 2))
])

# Pytorch example
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        ## call the parent class constructor
        super(SimpleCNN, self).__init__()
        ## create a convolutional layer with 3 input channels, 32 filters and a 3x3 kernel size
        self.conv1 = nn.Conv2d(3, 32, kernal_size=3)
        ## create a max pooling layer with a 2x2 max pooling window
        self.pool1 = nn.MaxPool2d(2, 2)
        ## create a second convolutional layer with 32 input channels, 64 filters and a 3x3 kernel size
        self.conv2 = nn.Conv2d(32, 64, kernal_size=3)
        ## create a average pooling layer with a 2x2 avg pooling window
        self.pool2 = nn.AvgPool2d(2, 2)
        
    ## This defines the forward pass
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        return x




