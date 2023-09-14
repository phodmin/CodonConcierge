import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math
#from util import *
from util3 import *

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        assert len(src_sequences) == len(tgt_sequences), "Source and target sequences must have the same length."
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, index):
        
        src_sequence = encode_amino_sequence(self.src_sequences[index])
        tgt_sequence = encode_codon_sequence(self.tgt_sequences[index])
        return torch.tensor(src_sequence), torch.tensor(tgt_sequence)

# Amino acids (20 + '*' fors stop + 'X')
src_vocab_size = 22

# Codons (64 + 1 'X' for padded, i.e. unknown codons)
tgt_vocab_size = 65

# Model Configurations
MODEL_CONFIGS = {
    "small": {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 512,
        "dropout": 0.1
    },
    "medium": {
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 1024,
        "dropout": 0.1
    },
    "large": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1
    }
}

# Training Function
def train_model(model, dataloader, tgt_vocab_size, epochs=100, lr=0.0001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is used for padding.
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (src_data, tgt_data) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(src_data, tgt_data[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    max_seq_length = 500
    config = "small"
    
    print("Done setting the constants.")

    #src_sequences, tgt_sequences = load_src_tgt_sequences()
    src_sequences, tgt_sequences = load_src_tgt_sequences(source_file=gencode_source_file_path,max_seq_length=max_seq_length)
    print("Done loading the data.")
    # Filtering can be done based on either src or tgt, or both
    filtered_data = [(src, tgt) for src, tgt in zip(src_sequences, tgt_sequences) if len(src) <= max_seq_length]
    print(f"Filtered from {len(src_sequences)} to {len(filtered_data)} sequences.")
    src_sequences, tgt_sequences = zip(*filtered_data)
    print("Done filtering the data.")

    dataset = SequenceDataset(src_sequences, tgt_sequences)
    print("Done creating the dataset.")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    print("Done creating the dataloader.")


    # Create the transformer model using the chosen configuration
    config_params = MODEL_CONFIGS[config]
    print(f"Done setting - Model Configuration: {config_params}")
    transformer = Transformer(src_vocab_size, tgt_vocab_size, **config_params, max_seq_length=max_seq_length)
    print("Done creating the model. INITIALISE TRAINING.")
    # Train the model
    train_model(transformer, dataloader, tgt_vocab_size=tgt_vocab_size, epochs=10)


    # Train the model
    model = train_model(transformer, dataloader, tgt_vocab_size=tgt_vocab_size, epochs=10)

    # Save the model to a file
    torch.save(model.state_dict(), 'my_model.pt')