# # insall libraries
# ## pip install matplotlib numpy scipy torch tensorflow

# # import libraries
# ## used for creating visualizations such as charts, plots and images
# import matplotlib.pyplot as plt
# ## essential for numerical computations and handling arrays
# import numpy as np
# ## convolve is used to perform convolution operations on images
# from scipy.ndimage import convolve

# # Load a sample grayscale image
# ## generates a ten by ten random grayscale image values between 0 and 1 to simulate an example of an image
# image = np.random.rand(10, 10)

# # Visualize the effects of convolution on an image
# #print(image)


# # Define convolution kernels(filters)
# ## this edge detection kernel is a 3 by 3 kernel filter used for detecting edges in the image by emphasizing high contrast area
# edge_detection_kernel = np.array([
#     [-1, -1, -1],
#     [-1, 8, -1],
#     [-1, -1, -1],
# ])

# ## Create a blur kernel
# ## This is a 3x3 kernel used for blurring the image by Averaging the pixel values in a neighborhood
# blur_kernel = np.array([
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1]
# ]) / 9                  # normalizing for averaging so the sum of all the elements equals one, ensuring the brightness remains consistent

# # Apply Convolution
# ## apply the convolution operation on the image using the specified kernel
# ## contains the result of an edge detection kernel
# edge_detected_image = convolve(image, edge_detection_kernel)
# ## contains the result of a blur kernel
# blurred_image = convolve(image, blur_kernel)

# # Visualize original and filtered image
# ## creates a figure with 1 row and 3 columns for side by side visualization of the images and fixed size 12x12 four sets the size of the figure to 12in wide and 4in tall
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# ## displays the original image on the first subplot with grayscale color map. So it takes the image and shows it in the first axis
# axes[0].imshow(image, cmap="gray")
# axes[0].set_title("Original Image")
# ## display the edge detected image on the second subplot
# axes[1].imshow(edge_detected_image, cmap="gray")
# axes[1].set_title("Edge Detected")
# ## display the blurred image on the third subplot
# axes[2].imshow(blurred_image, cmap="gray")
# axes[2].set_title("Blurred")
# ## show the plot
# plt.show()


## Tensorflow - building and training deep learning models
import tensorflow as tf

# Create a sample input tesnor (batch_size, height, width, channels)
## This is a random ten by ten grayscale image
image_tensor = tf.random.normal([1, 10, 10, 1])

# Define a convolutional layer
## Calling this 2D which defines a 2D convolutional layer in TensorFlow
conv_layer = tf.keras.layers.Conv2D(
    filters=1,                          # specifies the number of output channels or filters
    kernel_size=(3,3),                  # defines the size of the convolutional kernel
    strides=(1, 1),                     # specifies the step size of the convolution
    padding='same'                      # ensuring the output size matches the input size by padding the borders
)

# Applying convolution
## Applies the convolution layer to the image tensor
output_tensor = conv_layer(image_tensor)

## Print results
print(f"Original Shape: {image_tensor.shape}")
print(f"Ouput Shape: {output_tensor.shape}")

## import Pytorch
import torch
## Torch's main library
import torch.nn as nn

# Create a sample input tensor (batch_size, channels, height, width)
## This is q random 10x10 grayscale image
image_tensor_pt = torch.randn(1, 1, 10, 10)

# Define a convolutional layer
## Calling this 2D which defines a 2D convolutional layer in Pytorch
conv_layer_pt = nn.Conv2d(
    in_channels=1,              # number of input channels
    out_channels=1,             # number of output channels
    kernel_size=3,              # size of the convolutional kernel
    stride=1,                   # stride of the convolution
    padding=1                   # padding of the convolution
)

# APply Convolution
## Applies the convolution layer to the image tensor pytorch
output_tensor_pt = conv_layer_pt(image_tensor_pt)

## Print results
print(f"Original Shape: {image_tensor_pt.shape}")
print(f"Oytput Shape: {output_tensor_pt.shape}")

## We can experiment with different kernel sizes, strides, and padding to see how they affect the output shape

# TensorFlow Example
conv_layer_large_kernel = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), padding="same")
output_large_kernel = conv_layer_large_kernel(image_tensor)
## Print results for TensorFlow
print(f"Large Kernel Output Shape: {output_large_kernel.shape}")

# Pytorch Example
conv_layer_stride_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
output_stride_2 = conv_layer_stride_2(image_tensor_pt)
## Print results for Pytorch
print(f"Stride Output Shape: {output_stride_2.shape}")
