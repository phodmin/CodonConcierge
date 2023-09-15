import torch
import torch.nn as nn 
import torch.optim as optim
import pandas as pd
from util import *


# Model definition
class DiffusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Training loop
def train(model, sequences, num_epochs, lr, sigma):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for seq in sequences:
            noised_seq = noise(seq, sigma)
            predicted = model(noised_seq)
            target = seq - noised_seq
            loss = loss_fn(predicted, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
# Sequence processing functions       
def encode(sequences):
    # One-hot encode sequences
    encodings = []
    for seq in sequences:
        encoding = one_hot_encode(seq)
        encodings.append(encoding)
    return encodings

def pad(sequences, max_len):
    # Pad sequences to equal length
    padded = [seq.ljust(max_len) for seq in sequences] 
    return padded

def noise(sequence, sigma):
    # Add noise to sequence
    noised = sequence + sigma * torch.randn_like(sequence)
    return noised

# Generate a sample sequence
def generate_sample(model):
    random_input = torch.randn(input_size) 
    predicted = model(random_input)
    return decode_sequence(predicted)