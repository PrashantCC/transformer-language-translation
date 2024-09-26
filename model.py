# model.py
import torch
import torch.nn as nn
import torchtext
import spacy

# Load Spacy tokenizers
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

# Tokenization function
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


SRC = torchtext.data.Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = torchtext.data.Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
# Define the Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, n_heads, n_layers, pf_dim, dropout, max_length=100):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, n_heads, pf_dim, dropout), n_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(emb_dim, n_heads, pf_dim, dropout), n_layers)

        self.src_tok_emb = nn.Embedding(input_dim, emb_dim)
        self.trg_tok_emb = nn.Embedding(output_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, emb_dim))

        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        src_seq_length = src.shape[0]
        trg_seq_length = trg.shape[0]

        src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, -1)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, -1)

        src_emb = self.dropout(self.src_tok_emb(src) + self.positional_encoding[:, :src_seq_length, :])
        trg_emb = self.dropout(self.trg_tok_emb(trg) + self.positional_encoding[:, :trg_seq_length, :])

        enc_src = self.encoder(src_emb)
        output = self.decoder(trg_emb, enc_src)

        return self.fc_out(output)

# Data loading function
def load_data():
    # Load dataset
    train_data, valid_data, test_data = torchtext.datasets.IWSLT.splits(
        exts=('.en', '.de'), fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # Create iterators
    train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=128,
        sort_within_batch=True,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    return train_iterator, valid_iterator, test_iterator
