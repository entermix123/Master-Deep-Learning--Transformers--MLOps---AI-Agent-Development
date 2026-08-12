# Import Libraries
# import ready dataset - a standard data set for sentiment analysis, where each review is represented as a sequence of integers
from tensorflow.keras.datasets import imdb
# pad_sequences is used to ensure all sequences which are reviews are of the same length by padding or truncating them
from tensorflow.keras.preprocessing.sequence import pad_sequences
# TensorFlow for deep learning operations
import tensorflow as tf
# Sequential class used for building a sequential which is layer by layer model
from tensorflow.keras.models import Sequential
# These are these import specific layers
# Embedding converts word indices to dense vectors. SimpleRNN implements a basic RNN layer and Dense is fully connected layer for output
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Hyperparameters
## vocabulary size - the top 10,000 most frequent words will be used
vocab_size = 10000
## This is the maximum sequence length. Reviews will be truncated or padded to 200 words, so every review is 200 words.
max_len = 200

# Load the dataset and set training and testing data
## X_train, y_train, X_test, y_test are lists of integers representing word indices and binary label zero for negative and one for positive for the y I items.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Preprocess the data
## Ensuring that all the reviews have the same length, which is max_len=200 and pad sequences with zeros at the end, which that's why we mentioned "post".
## Padding happens at the end and truncates reviews longer than 200 words if it's more than 200 words.
X_train = pad_sequences(X_train, maxlen = max_len, padding="post")
## Same thing for our test data set
X_test = pad_sequences(X_test, maxlen = max_len, padding="post")

# Print shapes of the datasets
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# Build the model
## Using sequential, which is a model built for stacking layers sequentially
model = Sequential([
    ## setting input dim equal to vocab size where input vocabulary size is 10,000 words and output dim equal to 128,
    ## where each word index is mapped to a 128 dimensional vector
    Embedding(input_dim=vocab_size, output_dim=128),
    # initializing simple RNN with 128 hidden units, 
    ## activation is tanh, 
    ## return_sequences=False - outputs only the last time steps results.
    SimpleRNN(128, activation='tanh', return_sequences=False),
    # creating a fully connected output layer using dense
    ## The 1 is the outputs a single value for binary classification
    ## activation is sigmoid - ensures the output is between 0 and 1 one probability
    Dense(1, activation='sigmoid')
])

# Compile the model
## optimizer='adam' - optimization algorithm for training the model
## loss='binary_crossentropy' - function for calculating loss for binary classification
## metrics=['accuracy'] - tracks the model accuracy during training process
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary including layers, parameters and output shapes
model.summary()

# Train the model
## X_train, y_train - training data and labels
## epochs=5 - number of epochs to train the model - five iterations over data set
## batch_size=32 - number of samples per batch - 32 processes 32 samples at a time.
## validation_split=0.2 - uses 20% of the training data for validation
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
## X_test, y_test - test data and labels
loss, accuracy = model.evaluate(X_test, y_test)

# Print test loss and accuracy
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

