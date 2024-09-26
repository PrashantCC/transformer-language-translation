# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from model import Transformer, load_data, SRC, TRG  # Import SRC and TRG

# Load data
train_iterator, valid_iterator, test_iterator = load_data()

# Initialize model parameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
N_HEADS = 8
N_LAYERS = 3
PF_DIM = 512
DROPOUT = 0.1

# Create model
model = Transformer(INPUT_DIM, OUTPUT_DIM, EMB_DIM, N_HEADS, N_LAYERS, PF_DIM, DROPOUT)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Training function
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg[:-1, :])  # Exclude last token from trg for decoder input
        output_dim = output.shape[-1]

        output = output.view(-1, output_dim)
        trg = trg[1:, :].view(-1)  # Exclude first token for target

        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Main training loop
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}')
