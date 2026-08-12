## Pytorch
import torch
## nn is used for building neural networks
import torch.nn as nn
## optim is used for optimizing neural networks
import torch.optim as optim
## utilities for working with data set which includes data loader for batching and tensor data set for packaging data.
## DataLoader is used to load data in batches
## TensorDataset is used to create a dataset from tensors
from torch.utils.data import DataLoader, TensorDataset
## tensorflow.keras provides access to popular image datasets
from tensorflow.keras.datasets import imdb
## pad_sequences - preprocessing - ensure all sequences which are reviews are of the same length by padding or truncating them
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
## vocabulary size - the top 10,000 most frequent words will be used
vocab_size = 10000
## This is the maximum sequence length. Reviews will be truncated or padded to 200 words, so every review is 200 words.
max_len = 200

# Load the dataset and set training and testing data
## X_train, y_train, X_test, y_test are lists of integers representing word indices and binary label zero for negative and one for positive for the y I items.
## X_train and X_test each review is a sequence of integers representing word indices. And y_train and y_test contains the binary sentiment labels of zero and one.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Preprocess the data
## Ensuring that all the reviews have the same length, which is max_len=200 and pad sequences with zeros at the end, which that's why we mentioned "post".
## Padding happens at the end and truncates reviews longer than 200 words if it's more than 200 words.
X_train = pad_sequences(X_train, maxlen = max_len, padding="post")
## Same thing for our test data set.
X_test = pad_sequences(X_test, maxlen = max_len, padding="post")

# Prepare the data for PyTorch
## Converts numpy arrays into PyTorch tensors and creates a data set of paired inputs and labels
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
## This batches the data set into groups of 32 and shuffles the data for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a RNN Model
class RNNModel(nn.Module):
    ## Constructor
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        ## Call the parent class constructor
        super(RNNModel, self).__init__()
        ## Embedding layer to transform the input data into a dense vector of a fixed size
        ## ## nn.Embedding embeds the input word indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        ## nn.RNN - simple recurrent neural network layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        ## nn.Linear - Fully connected layer - Mapping the RNN outputs to the final binary sentiment prediction.
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    # Forward pass function
    def forward(self, x):
        ## Embed the input sequences
        embedded = self.embedding(x)
        ## Pass the embedded sequences to the RNN
        ## hidden, contains the RNNs last hidden state
        output, hidden = self.rnn(embedded)
        ## Applies the fully connected layer to the hidden state and uses the sigmoid activation to output the probabilities
        return torch.sigmoid(self.fc(hidden.squeeze(0)))
    
# Initialize the model
## vocab_size=10000 - the top 10,000 most frequent words will be used
## embedding_dim=128 - the dimensionality of the word embeddings
## hidden_dim=128 - the dimensionality of the hidden state of the RNN
## output_dim=1 - the output dimensionality for binary classification so it will be 0 or 1
model = RNNModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, output_dim=1)

# Define the loss function and optimizer
## criterion is what computes the binary cross entropy loss for the predictions
## nn.BCELoss - Binary Cross Entropy Loss
criterion = nn.BCELoss()
## optimizer optimizes the model using Adam with a learning rate of 0.001
## optim.Adam - Stochastic Gradient Descent
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):                                         # for each epoch
        epoch_loss = 0                                                  # cumulative loss initially to zero
        for X_batch, y_batch in train_loader:                           # for each batch in the train loader
            optimizer.zero_grad()                                       # ptimizer zero grad, which clears the previous gradients that has been calculated
            predictions = model(X_batch).squeeze(1)                # make the predictions
            loss = criterion(predictions, y_batch.float())         # calculate the loss
            # computing the computing the gradients
            loss.backward()                                             # backpropagate the loss
            optimizer.step()                                            # pdates the model weights
            epoch_loss += loss.item()                                   # adding the batch loss to epoch loss 
        # print the epoch loss
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# Train the model
train_rnn(model, train_loader, criterion, optimizer)

# Evaluation function 
def evaluate_rnn(model, X_test, y_test):
    model.eval()                                    # set the model to evaluation mode
    with torch.no_grad():                           # disable gradient computation for eficiency
        predictions = model(torch.tensor(X_test)).squeeze(1)                # make the predictions
        loss = criterion(predictions, torch.tensor(y_test).float())         # calculate the loss
        accuracy = ((predictions > 0) == torch.tensor(y_test).float()).float().mean().item()        # calculate the accuracy
    # print the loss and accuracy
    print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# Evaluate the model
evaluate_rnn(model, X_test, y_test)
