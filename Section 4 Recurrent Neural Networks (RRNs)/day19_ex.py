# import ready dataset - a standard data set for sentiment analysis, where each review is represented as a sequence of integers
from tensorflow.keras.datasets import imdb
# Tokenizer is used to convert text to integer sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# pad_sequences is used to ensure all sequences which are reviews are of the same length by padding or truncating them
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sequential class used for building a sequential which is layer by layer model
from tensorflow.keras.models import Sequential
# Embedding converts word indices to dense vectors. LSTM implements a basic LSTM layer and Dense is fully connected layer for output
from tensorflow.keras.layers import Embedding, LSTM, Dense
# numpy is used for numerical operations
import numpy as np
# matplotlib for visualization
import matplotlib.pyplot as plt

# Load Dataset
# Hyperparameters
## vocabulary size - the top 10,000 most frequent words will be used
vocab_size = 10000
## This is the maximum sequence length. Reviews will be truncated or padded to 200 words, so every review is 200 words.
max_len = 200

# Load the dataset and set training and testing data
## X_train, y_train, X_test, y_test are lists of integers representing word indices and binary label zero for negative and one for positive for the y I items.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Decode reviews to text for preprocessing
## Imdb.get_word_index() returns a dictionary where keys are words and values are their corresponding indices
word_index = imdb.get_word_index()
## word_index is a dictionary where keys are words and values are their corresponding indices
reverse_word_index = {value: key for key, value in word_index.items()}
## reverse_word_index is a dictionary where keys are indices and values are their corresponding words
decoded_reviews = [" ".join([reverse_word_index.get(i - 3, "?") for i in review]) for review in X_train[:5]]

# Pad sequences
# Preprocess the data
## Ensuring that all the reviews have the same length, which is max_len=200 and pad sequences with zeros at the end, which that's why we mentioned "post".
## Padding happens at the end and truncates reviews longer than 200 words if it's more than 200 words.
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
## Same thing for our test data set
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# Print shapes of the datasets
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Load the pretrained GloVe embeddings
## The word vectors are stored in a text file where each line represents a word and its corresponding vector
embedding_index = {}
## Open the GloVe text file
glove_file = "data/glove.6B.100d.txt"
## open the GloVe text file and read each line, splitting it into a word and its corresponding vector
with open(glove_file, "r", encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        ## that will give us the coefficients
        coefs = np.asarray(values[1:], dtype="float32")
        ## add the word and its coefficients to the dictionary
        embedding_index[word] = coefs

# Print number of loaded word vectors
print(f"Loaded {len(embedding_index)} word vectors")

# Prepare embedding matrix
## embedding_dim is the dimensionality of the word embeddings
embedding_dim = 100
## vocab_size is the number of words in the vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
## iterate through each word in the vocabulary and get its corresponding embedding vector
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define LSTM model with GloVe embeddings
## Using sequential, which is a model built for stacking layers sequentially
model = Sequential([
    ## input_dim=vocab_size where input vocabulary size is 10,000 words 
    ## output_dim=embedding_dim where embedding dimension is 100
    ## weights=[embedding_matrix] - use pre-trained word embeddings
    ## trainable=False - freeze the embeddings
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
    # initializing LSTM with 128 hidden units, 
    ## activation is tanh, 
    ## return_sequences=False - outputs only the last time steps results
    LSTM(128, activation='tanh', return_sequences=False),
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
# Display the LSTM model summary including layers, parameters and output shapes
model.summary()

# Train the model
## X_train, y_train - training data and labels
## epochs=5 - number of epochs to train the model - five iterations over data set
## batch_size=64 - number of samples per batch - 64 processes 64 samples at a time.
## validation_split=0.2 - uses 20% of the training data for validation
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1
)

# Evaluate the model
## X_test, y_test - test data and labels, verbose=0 - suppress verbose output
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Define LSTM model without GloVe embeddings
lstm_model = Sequential([
    ## setting input dim equal to vocab size where input vocabulary size is 10,000 words and output dim equal to 128,
    ## where each word index is mapped to a 128 dimensional vector
    Embedding(input_dim=vocab_size, output_dim=128),
    # initializing LSTM with 128 hidden units, 
    ## activation is tanh, 
    ## return_sequences=False - outputs only the last time steps results
    LSTM(128, activation='tanh', return_sequences=False),
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
# Display the LSTM model summary including layers, parameters and output shapes
lstm_model.summary()

# Train the LSTM model
## X_train, y_train - training data and labels
## epochs=5 - number of epochs to train the model - five iterations over data set
## batch_size=64 - number of samples per batch - 64 processes 64 samples at a time.
## validation_split=0.2 - uses 20% of the training data for validation
lstm_history = lstm_model.fit(
    X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the LSTM model
## X_test, y_test - test data and labels
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)

# Print test loss and accuracy for LSTM model
print(f"LSTM without GloVe Test Accuracy: {lstm_accuracy:.4f}")
print(f"LSTM model with GloVe Test Accuracy: {accuracy:.4f}")

# Plot accuracy comparison
models = ['LSTM', 'LSTM GloVe']
accuracies = [lstm_accuracy, accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Comparison of LSTM with and without word embeddings')
plt.ylabel('Accuracy')
plt.show()
