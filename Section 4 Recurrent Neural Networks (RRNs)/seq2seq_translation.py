# =============================================================================
# Sequence-to-Sequence (Seq2Seq) Machine Translation: English → French
# Architecture: Encoder-Decoder with LSTM layers
# =============================================================================

# Import Libraries

## PyTorch core library
import torch
## nn provides building blocks for neural networks (layers, loss functions, etc.)
import torch.nn as nn
## optim provides optimization algorithms (Adam, SGD, etc.)
import torch.optim as optim
## DataLoader handles batching and shuffling data during training
## Dataset is the base class for creating custom datasets
from torch.utils.data import DataLoader, Dataset
## numpy is used for numerical operations and array manipulation
import numpy as np


# =============================================================================
# Data: Example English-to-French sentence pairs
# =============================================================================

english_sentences = ["hello", "how are you", "good morning", "thank you", "good night"]
french_sentences = ["bonjour", "comment ça va", "bon matin", "merci", "bonne nuit"]


# =============================================================================
# Vocabulary Building
# =============================================================================

def build_vocab(sentences):
    # Initialize the vocabulary with four special tokens:
    # <PAD> pads shorter sequences to a uniform length
    # <SOS> marks the Start Of a Sequence
    # <EOS> marks the End Of a Sequence
    # <UNK> represents any Unknown word not seen during training
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for sentence in sentences:
        for word in sentence.split():
            # Only add the word if it hasn't been seen before
            if word not in vocab:
                # Assign the next available integer index to this new word
                vocab[word] = len(vocab)
    return vocab

# Build separate vocabularies for source (English) and target (French) languages
english_vocab = build_vocab(english_sentences)
french_vocab = build_vocab(french_sentences)


# =============================================================================
# Tokenisation and Padding
# =============================================================================

def tokenize(sentences, vocab, max_len):
    tokenized = []
    for sentence in sentences:
        # Convert each word to its integer index; fall back to <UNK> if unseen
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]
        # Wrap the token sequence with <SOS> at the start and <EOS> at the end
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        # Pad with <PAD> tokens so every sequence reaches the same max_len
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        tokenized.append(tokens)
    # Return a 2-D NumPy array of shape (num_sentences, max_len)
    return np.array(tokenized)

# Calculate the maximum sequence length for each language, adding 2 for <SOS> and <EOS>
max_len_eng = max(len(sentence.split()) for sentence in english_sentences) + 2
max_len_fr  = max(len(sentence.split()) for sentence in french_sentences)  + 2

# Tokenise and pad both datasets using their respective vocabularies
english_data = tokenize(english_sentences, english_vocab, max_len_eng)
french_data  = tokenize(french_sentences,  french_vocab,  max_len_fr)


# =============================================================================
# Custom Dataset
# =============================================================================

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        # Store the source (English) and target (French) token arrays
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        # Return the total number of sentence pairs in the dataset
        return len(self.src_data)

    def __getitem__(self, idx):
        # Return a single (source, target) pair as PyTorch Long tensors
        # Long (int64) is required by nn.Embedding, which expects integer indices
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])

# Instantiate the dataset with the tokenised arrays
dataset = TranslationDataset(english_data, french_data)

# Wrap the dataset in a DataLoader:
# batch_size=2 yields 2 sentence pairs per training step
# shuffle=True randomises the order each epoch to improve generalisation
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    # input_dim  – vocabulary size of the source language (number of unique tokens)
    # embed_dim  – size of each word embedding vector
    # hidden_dim – number of hidden units in each LSTM layer
    # num_layers – number of stacked LSTM layers
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        # Initialise the parent nn.Module class
        super(Encoder, self).__init__()

        # Embedding layer: maps each integer token index to a dense vector of size embed_dim.
        # Acts as a lookup table of shape (input_dim, embed_dim).
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # LSTM layer: processes the sequence of embeddings and captures temporal dependencies.
        # batch_first=True means input/output tensors have shape (batch, seq_len, features)
        # rather than the default (seq_len, batch, features).
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # Convert integer token indices into dense embedding vectors
        # Shape: (batch_size, src_seq_len, embed_dim)
        embedded = self.embedding(x)

        # Feed embeddings through the LSTM.
        # outputs – hidden states for every time step, shape (batch, src_seq_len, hidden_dim)
        # hidden  – final hidden state for each layer,  shape (num_layers, batch, hidden_dim)
        # cell    – final cell state for each layer,    shape (num_layers, batch, hidden_dim)
        outputs, (hidden, cell) = self.lstm(embedded)

        # Only the final hidden and cell states are needed; they carry the full source context
        # and will be used to initialise the Decoder.
        return hidden, cell


# =============================================================================
# Decoder
# =============================================================================

class Decoder(nn.Module):
    # output_dim – vocabulary size of the target language (number of unique tokens)
    # embed_dim  – size of each word embedding vector
    # hidden_dim – number of hidden units in each LSTM layer (must match Encoder)
    # num_layers – number of stacked LSTM layers (must match Encoder)
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        # Initialise the parent nn.Module class
        super(Decoder, self).__init__()

        # Embedding layer: maps each target token index to a dense embedding vector
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # LSTM layer: generates the next hidden state given the current token and previous state.
        # batch_first=True keeps shapes consistent with the Encoder.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected (linear) layer: projects the LSTM hidden state to a score
        # over every word in the target vocabulary, shape (hidden_dim → output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x arrives as a 1-D tensor of token indices, shape (batch_size,).
        # unsqueeze(1) adds a sequence-length dimension → shape (batch_size, 1)
        # so the LSTM receives a single time step per call.
        x = x.unsqueeze(1)

        # Embed the single input token → shape (batch_size, 1, embed_dim)
        embedded = self.embedding(x)

        # Run one LSTM step, conditioned on the previous hidden and cell states.
        # outputs shape: (batch_size, 1, hidden_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Remove the sequence-length dimension: (batch_size, 1, hidden_dim) → (batch_size, hidden_dim)
        # then project to vocabulary scores → shape (batch_size, output_dim)
        predictions = self.fc(outputs.squeeze(1))

        # Return raw logits plus the updated states for the next decoding step
        return predictions, hidden, cell


# =============================================================================
# Seq2Seq Model (ties Encoder and Decoder together)
# =============================================================================

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        # Initialise the parent nn.Module class
        super(Seq2Seq, self).__init__()
        # Store the encoder, decoder, and target device as instance attributes
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    # src                  – source (English) token tensor,  shape (batch, src_len)
    # tgt                  – target (French)  token tensor,  shape (batch, tgt_len)
    # teacher_forcing_ratio – probability of using the ground-truth token (vs. the model's
    #                         own prediction) as the next decoder input during training
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # Number of sequences processed in parallel
        batch_size = src.size(0)
        # Length of the target sequence (including <SOS> and <EOS>)
        tgt_len = tgt.size(1)
        # Size of the target-language vocabulary (output dimension of the linear layer)
        tgt_vocab_size = self.decoder.fc.out_features

        # Pre-allocate a tensor to store the decoder's output logits at every time step.
        # Shape: (batch_size, tgt_len, tgt_vocab_size); initialised to zeros.
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encode the full source sequence; returns the final hidden and cell states
        hidden, cell = self.encoder(src)

        # Seed the decoder with the <SOS> token (first column of the target tensor)
        input = tgt[:, 0]

        # Generate one target token at a time, starting from position 1
        # (position 0 is the <SOS> token used to prime the decoder)
        for t in range(1, tgt_len):
            # Run a single decoder step; output shape: (batch_size, tgt_vocab_size)
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Store this step's logits in the pre-allocated outputs tensor
            outputs[:, t, :] = output

            # Greedy prediction: pick the token with the highest score
            top1 = output.argmax(1)

            # Teacher forcing: with probability teacher_forcing_ratio use the real
            # next token from the target sequence; otherwise use the model's own prediction.
            # This helps stabilise early training while gradually forcing the model
            # to rely on its own outputs.
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs


# =============================================================================
# Model Instantiation
# =============================================================================

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary sizes determine the embedding table dimensions
input_dim  = len(english_vocab)
output_dim = len(french_vocab)

# Hyperparameters – can be tuned to trade model capacity against training speed
embed_dim  = 64    # Dimensionality of word embedding vectors
hidden_dim = 128   # Number of units in each LSTM hidden state
num_layers = 2     # Number of stacked LSTM layers in both Encoder and Decoder

# Build sub-modules and compose them into the full Seq2Seq model
encoder = Encoder(input_dim,  embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers)
model   = Seq2Seq(encoder, decoder, device).to(device)

# Adam optimiser: adapts the learning rate per-parameter, generally a strong default
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CrossEntropyLoss combines LogSoftmax + NLLLoss internally.
# ignore_index tells the loss to skip <PAD> positions so they don't affect gradients.
criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])


# =============================================================================
# Training Loop
# =============================================================================

def train(model, dataloader, optimizer, criterion, device, num_epochs=20):
    # Put the model in training mode: enables dropout, batch norm updates, etc.
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0  # Accumulate loss across all batches in this epoch

        for src, tgt in dataloader:
            # Move tensors to the configured device (CPU or GPU)
            src, tgt = src.to(device), tgt.to(device)

            # Clear gradients from the previous batch to prevent accumulation
            optimizer.zero_grad()

            # Forward pass: produce logits for every target time step
            output = model(src, tgt)

            # Reshape for the loss function:
            # Skip the first time step (index 0) because it is always <SOS>
            # and has no corresponding prediction in outputs.
            # output reshaped: (batch * (tgt_len-1), tgt_vocab_size)
            output = output[:, 1:].reshape(-1, output.shape[2])
            # tgt reshaped:   (batch * (tgt_len-1),)
            tgt = tgt[:, 1:].reshape(-1)

            # Compute cross-entropy loss between predictions and ground-truth tokens
            loss = criterion(output, tgt)

            # Backpropagate: compute gradients of the loss w.r.t. all parameters
            loss.backward()

            # Update model parameters using the computed gradients
            optimizer.step()

            # Accumulate the scalar loss value for reporting
            epoch_loss += loss.item()

        # Print the average loss over all batches for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Kick off training
train(model, dataloader, optimizer, criterion, device)


# =============================================================================
# Inference: Translate a single sentence
# =============================================================================

def translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device):
    # Switch to evaluation mode: disables dropout and stops gradient tracking
    model.eval()

    # Tokenise the input sentence using the English vocabulary
    tokens = [english_vocab.get(word, english_vocab["<UNK>"]) for word in sentence.split()]
    # Wrap with <SOS> and <EOS> markers
    tokens = [english_vocab["<SOS>"]] + tokens + [english_vocab["<EOS>"]]

    # Convert to a tensor and add a batch dimension of 1 → shape (1, src_len)
    src = torch.tensor(tokens).unsqueeze(0).to(device)

    # Encode the source sentence; no gradients needed during inference
    with torch.no_grad():
        hidden, cell = model.encoder(src)

    # Build the inverse French vocabulary: index → word (for converting predictions back to text)
    tgt_vocab = {v: k for k, v in french_vocab.items()}

    # Start decoding with the <SOS> token
    tgt_indices = [french_vocab["<SOS>"]]

    for _ in range(max_len_fr):
        # Feed the most recently predicted token index into the decoder
        tgt_tensor = torch.tensor([tgt_indices[-1]]).to(device)
        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)

        # Greedy decoding: select the token with the highest logit score
        pred = output.argmax(1).item()
        tgt_indices.append(pred)

        # Stop generating once the <EOS> token is predicted
        if pred == french_vocab["<EOS>"]:
            break

    # Convert predicted indices back to words, skipping <SOS> (first) and <EOS> (last)
    translated_sentence = [tgt_vocab[i] for i in tgt_indices[1:-1]]
    # Join individual word tokens into a single string
    return " ".join(translated_sentence)


# =============================================================================
# Test Translation
# =============================================================================

sentence   = "good night"
translation = translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device)
print(f"Translated Sentence: {translation}")
