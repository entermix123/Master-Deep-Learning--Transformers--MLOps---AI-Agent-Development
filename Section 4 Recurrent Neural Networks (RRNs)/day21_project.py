# import ready dataset - a standard data set for sentiment analysis, where each review is represented as a sequence of integers
from tensorflow.keras.datasets import imdb
# pad_sequences is used to ensure all sequences which are reviews are of the same length by padding or truncating them
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sequential class used for building a sequential which is layer by layer model
from tensorflow.keras.models import Sequential
# These are these import specific layers
# Embedding converts word indices to dense vectors. SimpleRNN implements a basic RNN layer, Dense is fully connected layer for output 
# LSTM is long short term memory, GRU is gated recurrent unit
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU
# matplotlib is a library for creating static, animated, and interactive visualizations in Python
import matplotlib.pyplot as plt

# Hyperparameters
## vocabulary size - the top 10,000 most frequent words will be used
vocab_size = 10000
## This is the maximum sequence length. Reviews will be truncated or padded to 100 words, so every review is 100 words.
max_len = 100

# Load the dataset and set training and testing data
## X_train, y_train, X_test, y_test are lists of integers representing word indices and binary label zero for negative and one for positive for the y I items.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words= vocab_size)

# Pad Sequences
# Preprocess the data
## Ensuring that all the reviews have the same length, which is max_len=200 and pad sequences with zeros at the end, which that's why we mentioned "post".
## Padding happens at the end and truncates reviews longer than 200 words if it's more than 200 words.
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
## Same thing for our test data set
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# Print shapes of the datasets
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Define the RNN model
## Using sequential, which is a model built for stacking layers sequentially
rnn_model = Sequential([
    ## setting input dim equal to vocab size where input vocabulary size is 10,000 words and output dim equal to 128,
    ## where each word index is mapped to a 128 dimensional vector
    Embedding(input_dim=vocab_size, output_dim=128),
    # initializing simple RNN with 128 hidden units, 
    ## activation is tanh, 
    ## return_sequences=False - outputs only the last time steps results.
    SimpleRNN(128, activation='tanh'),
    # creating a fully connected output layer using dense
    ## The 1 is the outputs a single value for binary classification
    ## activation is sigmoid - ensures the output is between 0 and 1 one probability
    Dense(1, activation='sigmoid')
])

# Compile the model
## optimizer='adam' - optimization algorithm for training the model
## loss='binary_crossentropy' - function for calculating loss for binary classification
## metrics=['accuracy'] - tracks the model accuracy during training process
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary including layers, parameters and output shapes
rnn_model.summary()

# Define the LSTM Model
## Using sequential, which is a model built for stacking layers sequentially
lstm_model = Sequential([
    ## setting input dim equal to vocab size where input vocabulary size is 10,000 words and output dim equal to 128,
    ## where each word index is mapped to a 128 dimensional vector
    Embedding(input_dim=vocab_size, output_dim=128),
    # initializing simple RNN with 128 hidden units, 
    ## activation is tanh, 
    ## return_sequences=False - outputs only the last time steps results.
    LSTM(128, activation='tanh'),
    # creating a fully connected output layer using dense
    ## The 1 is the outputs a single value for binary classification
    ## activation is sigmoid - ensures the output is between 0 and 1 one probability
    Dense(1, activation='sigmoid')
])

# Compile the LSTM model
## optimizer='adam' - optimization algorithm for training the model
## loss='binary_crossentropy' - function for calculating loss for binary classification
## metrics=['accuracy'] - tracks the model accuracy during training process
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary including layers, parameters and output shapes
lstm_model.summary()

# Define the GRU Model
gru_model = Sequential([
    ## setting input dim equal to vocab size where input vocabulary size is 10,000 words and output dim equal to 128,
    ## where each word index is mapped to a 128 dimensional vector
    Embedding(input_dim=vocab_size, output_dim=128),
    # initializing simple RNN with 128 hidden units, 
    ## activation is tanh, 
    ## return_sequences=False - outputs only the last time steps results.
    GRU(128, activation='tanh'),
    # creating a fully connected output layer using dense
    ## The 1 is the outputs a single value for binary classification
    ## activation is sigmoid - ensures the output is between 0 and 1 one probability
    Dense(1, activation='sigmoid')
])

# Compile the GRU model
## optimizer='adam' - optimization algorithm for training the model
## loss='binary_crossentropy' - function for calculating loss for binary classification
## metrics=['accuracy'] - tracks the model accuracy during training process
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary including layers, parameters and output shapes
gru_model.summary()

# Train all models
## X_train, y_train - training data and labels
## epochs=5 - number of epochs to train the model - five iterations over data set
## batch_size=64 - number of samples per batch - 64 processes 64 samples at a time.
## validation_split=0.2 - uses 20% of the training data for validation

# Train RNN model
history_rnn = rnn_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Train LSTM model
history_lstm = lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Train GRU model
history_gru = gru_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Evaluate models
## X_test, y_test - test data and labels
loss_rnn, accuracy_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
loss_gru, accuracy_gru = gru_model.evaluate(X_test, y_test, verbose=0)

print(f"RNN Test Accuracy: {accuracy_rnn:.4f}")
print(f"LSTM Test Accuracy: {accuracy_lstm:.4f}")
print(f"GRU Test Accuracy: {accuracy_gru:.4f}")

# Plot training accuracy
plt.plot(history_rnn.history['accuracy'], label="RNN Training Acuracy")
plt.plot(history_lstm.history['accuracy'], label="LSTM Training Acuracy")
plt.plot(history_gru.history['accuracy'], label="GRU Training Acuracy")
plt.title("Training Accuracy Comparison")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()