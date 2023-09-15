# filename: training_ground.py

# Python standard library
from sklearn.model_selection import train_test_split

# PyTorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Custom modules
# from util import amino_acid_to_codon, load_sequences, one_hot_encode, collate_fn
from util import *


def load_and_split_data():
    sequences = load_sequences() 

    # 80-10-10 train-val-test split
    train_seqs, rem = train_test_split(sequences, test_size=0.2) 
    val_seqs, test_seqs = train_test_split(rem, test_size=0.5)
    return train_seqs, val_seqs, test_seqs


class mRNADataset(Dataset):

    def __init__(self, amino_acid_sequences):
        self.amino_acid_sequences = amino_acid_sequences

    def __len__(self):
        return len(self.amino_acid_sequences)

    def __getitem__(self, i):
        amino_acid_seq = self.amino_acid_sequences[i]
        codon_seq = amino_acid_to_codon(amino_acid_seq)
        return one_hot_encode(amino_acid_seq), [codon_to_index[codon_seq[j:j+3]] for j in range(0, len(codon_seq), 3)]

def create_dataloaders(train_seqs, val_seqs, batch_size=64):
    train_loader = DataLoader(mRNADataset(train_seqs), shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(mRNADataset(val_seqs), batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader


class BaseModel(nn.Module):

    def __init__(self, loss_fn):
        super(BaseModel, self).__init__()
        self.loss_fn = loss_fn

    def initialize_optimizer(self, optimizer_cls, lr):
        self.optimizer = optimizer_cls(self.parameters(), lr)

    def forward(self, x):
        # This will be model-specific but should exist in the base class.
        raise NotImplementedError("The forward method should be overridden by subclasses.")

    def train_step(self, x, y):
        predictions = self(x)
        y_label_encoded = torch.argmax(y, dim=-1)
        loss = self.loss_fn(predictions, y_label_encoded)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dataloader):
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                y_label_encoded = torch.argmax(y, dim=-1) # Add this line
                predictions = self(x)
                loss = self.loss_fn(predictions, y_label_encoded) # Modify this line
                total_loss += loss.item()
        return total_loss / len(dataloader)


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def binary_cross_entropy(predictions, targets):
  return F.binary_cross_entropy(predictions, targets)

def mse_loss(predictions, targets):
  return F.mse_loss(predictions, targets)

print("training_ground.py runs successfully.")