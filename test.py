# test.py
import torch
import spacy
from model import Transformer, load_data, SRC, TRG  # Import SRC and TRG

# Load Spacy tokenizers
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

# Function to translate sentences
def translate_sentence(sentence, model, src_field, trg_field, device):
    model.eval()  # Set the model to evaluation mode

    tokens = [token.text.lower() for token in spacy_en(sentence)]  # Tokenize input sentence
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]  # Add <sos> and <eos> tokens

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]  # Convert tokens to indexes
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # Shape: [src_len, batch_size]

    with torch.no_grad():  # Disable gradient calculation
        enc_src = model.encoder(model.src_tok_emb(src_tensor))

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]  # Initialize target sequence with <sos> token

    for _ in range(50):  # Limit the length of the output
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)  # Shape: [trg_len, batch_size]

        with torch.no_grad():
            output = model.decoder(model.trg_tok_emb(trg_tensor), enc_src)
        
        # Get the last output token
        output_dim = output.shape[-1]
        pred_token = output.argmax(dim=2)[-1].item()  # Get the last prediction
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:  # Stop if <eos> is predicted
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]  # Convert indexes to tokens
    return trg_tokens[1:]  # Exclude <sos>

# Load data and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
N_HEADS = 8
N_LAYERS = 3
PF_DIM = 512
DROPOUT = 0.1

# Instantiate model
model = Transformer(INPUT_DIM, OUTPUT_DIM, EMB_DIM, N_HEADS, N_LAYERS, PF_DIM, DROPOUT)
model.load_state_dict(torch.load('path/to/your/model.pth'))  # Load the trained model weights
model = model.to(device)

# Example sentences for translation
sentences = [
    "This is a test sentence.",
    "How are you today?",
    "What is the weather like?",
]

# Translate and print results
for sentence in sentences:
    translation = translate_sentence(sentence, model, SRC, TRG, device)
    print(f"Source: {sentence}\nTranslation: {' '.join(translation)}\n")
