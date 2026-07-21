## Tensorflow - building and training deep learning models
import tensorflow as tf
from tensorflow.keras import layers, models
## cifar10 dataset
from tensorflow.keras.datasets import cifar10
## ImageDataGenerator - used for data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
## used for creating visualizations such as charts and plots
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
## This loads the data set and splits it into training which is X_train and y_train and testing into X_test and y_test sets 
## X_train and X_test contain image data, while y_train and y_test contain the corresponding labels for those images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
## This converts the image data to a 32 bit floating point number for compatibility with TensorFlow models 
## divide it by 250 5.0, it normalizes the pixel values from 0 to 255 to 0 to 1 to improve model convergence during the training process
x_train = x_train.astype('float32') / 255.0
## same for test data
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
## For example a label for 3 will become 00010000 because it will kind of label the third element in the list as one
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Apply data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,                  # Rotate images by up to 15 degrees
    width_shift_range=0.1,              # Shift images horizontally by up to 10%
    height_shift_range=0.1,             # Shift images vertically by up to 10%
    horizontal_flip=True                # Flip images horizontally
)

# Fit the generator to training data
datagen.fit(x_train)

# Create the model architecture
def create_model():
    # Initializes a sequential model which allows layers to be stacked linearly
    model = models.Sequential()
    
    # Convolutional Layer 1
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model
    
# Create the model
model = create_model()

# Compile the model
## optimizer='adam' - optimization algorithm for training the model
## loss='categorical_crossentropy' - specifies the loss function for multi-class classification
## metrics=['accuracy'] - metric for evaluating the model - tracks accuracy during the training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data generator
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),          # Use the data generator to generate batches of augmented data
    epochs=20,                                              # Number of training epochs
    validation_data=(x_test, y_test),                       # Use the test data for validation
    steps_per_epoch=x_train.shape[0] // 64                  # Number of batches per epoch
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot accuracy and loss over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Plot loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()