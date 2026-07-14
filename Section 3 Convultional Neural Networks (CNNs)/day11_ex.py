# Install Libraries
## pip install tensorflow matplotlib

# import libraries
## Import the cipher ten data set, which consists of 60,032 by 32 color images in ten classes where we will be dividing it, 50,000 for training and 10,000 for testing samples.
from tensorflow.keras.datasets import cifar10
## This imports a utility to convert integer labels to one hot encoded labels
from tensorflow.keras.utils import to_categorical
## Tensorflow - building and training deep learning models
import tensorflow as tf

## Tensorflow - building and training deep learning models
from tensorflow.keras.models import Sequential
## Conv2D is a convolutional layer with 32 filters, a 3 by 3 kernel size and a ReLU activation function
## MaxPooling2D is a max pooling 2D layer with two by two pooling window
## Flatten is a function that flattens the 2D feature maps into a 1D vector for fully connected layers
## Dense is a fully connected layer with 128 neurons and ReLU activation function
## Dropout is a dropout layer to prevent overfitting
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load CIFAR-10 dataset
## This loads the data set and splits it into training which is X_train and y_train and testing into X_test and y_test sets 
## X_train and X_test contain image data, while y_train and y_test contain the corresponding labels for those images
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data
## This converts the image data to a 32 bit floating point number for compatibility with TensorFlow models 
## divide it by 250 5.0, it normalizes the pixel values from 0 to 255 to 0 to 1 to improve model convergence during the training process
X_train = X_train.astype('float32') / 255.0
## same for test data
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels
## For example a label for 3 will become 0010000 because it will kind of label the third element in the list as one
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Print shapes of the data
print(f"Training Data Shape: {X_train.shape}, Label Shapes: {y_train.shape}")
print(f"Test Data Shape: {X_test.shape}, Label Shapes: {y_test.shape}")

# Build the CNN model
model = Sequential([
    ## Adds a convolution layer with 32 filters, kernel size 3x3 
    ## Adds ReLU activation and input shape of 32 x 32 x 3, which is the image size and RGB channels
    Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)),       # 32 filters, 3x3 kernel size, ReLU activation, input shape
    ## Adds a 2x2 max pooling layer to reduce the spatial dimensions
    MaxPooling2D((2, 2)),                                               # 2x2 pooling window
    ## Adds a second convolution layer with 64 filters, kernel size 3x3 and ReLU activation
    Conv2D(64, (3, 3), activation='relu'),
    ## Adds a 2x2 max pooling layer to reduce the spatial dimensions
    MaxPooling2D((2, 2)),
    ## This flattens the 2D features map into a 1D array for fully connected layers - Flatten the 2D feature maps into a 1D vector
    Flatten(),
    ## This creates that fully connected layer with 128 units and ReLU activation
    Dense(128, activation='relu'),
    ## adds dropout layer - Regularization with 50% dropout to prevent the overfitting part 
    Dropout(0.5),
    ## Create the output layer with 10 units, one for each class and softmax activation for classification
    Dense(10, activation='softmax')
])

# diplay model summary/architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',                   # Efficient optimization for large networks
    loss='categorical_crossentropy',    # Categorical Cross-Entropy Loss
    metrics=['accuracy']                # Accuracy metric for model evaluation
)

# Train the model
history = model.fit(
    X_train, y_train,           # Training data
    epochs=10,                  # Number of training epochs - each epoch is a full pass through the training data
    batch_size=64,              # Batch size - Number of samples per gradient update
    validation_split=0.2        # Validation split - 20% of the training data will be used for validation
)

# Evaluate on the test dataset
## Evaluate the model on the test datasets
test_loss, test_accuracy = model.evaluate(X_test, y_test)
## Print the test accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")

# import matplotlib for visualization of the results
import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot Loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
