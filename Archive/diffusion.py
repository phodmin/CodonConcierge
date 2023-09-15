import torch
import torch.nn as nn
import torch.optim as optim
import Archive.csv_writer as csv_writer
import pandas as pd
from util import *

# Paths
source_file_path = '../data/gencode/gencode.v44.pc_transcripts.fa'

parsed_gencode = csv_writer.basic_fasta_parser(source_file_path)
parsed_gencode = pd.DataFrame(parsed_gencode)

# mRNA sequence data,
cds_data = []

for i, row in parsed_gencode.iterrows():
    seq = row['sequence']
    cds_start = int(row['CDS'].split('-')[0])
    cds_end = int(row['CDS'].split('-')[1])
    
    # checks
    if cds_start <= 0 or cds_end > len(seq):
        continue # skip invalid sequence

    cds = seq[int(cds_start)-1:int(cds_end)]
    cds_data.append(cds)

#print(cds_data)

# Pad sequences
max_len = max([len(seq) for seq in cds_data])
padded_seqs = [seq.ljust(max_len) for seq in cds_data]

encoded_data = torch.tensor([one_hot_encode(seq) for seq in cds_data], dtype=torch.float32)

#
# Step Two
# Model Architecture
#

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ScoreNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# For a sequence of length 5 with one-hot encoding, input_dim = 5 * 4
model = ScoreNetwork(5 * 4, 50, 5 * 4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 
# Step Three
# Training Loop
# 

num_epochs = 1000
num_timesteps = 10  # Number of diffusion timesteps
sigma = 0.5  # Noise level for the diffusion process

for epoch in range(num_epochs):
    for seq in encoded_data:
        seq_flat = seq.view(-1)
        
        # Corrupting the sequence using a diffusion process
        noised_seq = seq_flat + sigma * torch.randn_like(seq_flat)
        
        # Predicting the score for the noised sequence
        predicted_score = model(noised_seq)
        
        # The target score is the difference between the original and noised sequence
        target_score = seq_flat - noised_seq
        
        loss = loss_fn(predicted_score, target_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Pad sequences to max length
max_len = max([len(seq) for seq in cds_data])  
padded_seqs = [seq.ljust(max_len) for seq in cds_data]

# Encode padded sequences 
encoded_data = [one_hot_encode(seq) for seq in padded_seqs]

# Train on all sequences 
for epoch in range(num_epochs):

  for seq in encoded_data:
    
    # Rest of training loop...
    
    if epoch % 100 == 0:
         print(f"Epoch {epoch}, Loss: {loss.item()}")

# Generate sequence
random_seq = torch.randn(max_len * 4) 
predicted = model(random_seq)
generated = decode_sequence(predicted)