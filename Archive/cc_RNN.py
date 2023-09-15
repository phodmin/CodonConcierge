# filename: cc_RNN.py

from Archive.training_ground import *
from util import *
import time

class RNNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob, optimizer_cls, lr, loss_fn):
        # First, call the parent initializer
        super(RNNModel, self).__init__(loss_fn=loss_fn)

        # After that, initialize the layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_prob)

        # Now, initialize the optimizer
        self.initialize_optimizer(optimizer_cls, lr)

    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        lstm_out, _ = self.lstm(x)
        
        # Stack up LSTM outputs using view
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # Return the final output
        return out
    



def generate_sequence(model, amino_acid_seq):
    model.eval()
    with torch.no_grad():
        codon_seq = amino_acid_to_codon(amino_acid_seq)
        input_data = one_hot_encode(codon_seq)
        predictions = model(input_data)
        codon_seq_generated = predictions_to_codon_sequence(predictions)
    return codon_seq_generated


def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch
            print(f"x shape: {x.shape}, y shape: {y.shape}")  # Add this line
            loss = model.train_step(x, y)
            total_loss += loss

        avg_train_loss = total_loss / len(train_loader)
        val_loss = model.validate(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {val_loss}")



def main():
    # Define the RNN model
    model = RNNModel(input_dim=4, hidden_dim=64, output_dim=64, n_layers=2, drop_prob=0.2, optimizer_cls=torch.optim.Adam, lr=0.001, loss_fn=F.cross_entropy)

    # model = RNNModel(input_dim=4, hidden_dim=64, output_dim=len(all_codons), 
    #                 n_layers=2, drop_prob=0.2, 
    #                 optimizer_cls=torch.optim.Adam, lr=0.001, loss_fn=F.cross_entropy)

    # Load and split data
    train_seqs, val_seqs, test_seqs = load_and_split_data()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_seqs, val_seqs)

    # Train the model
    start_time = time.time()
    train_model(model, train_loader, val_loader, num_epochs=10)
    end_time = time.time()

    # Print out the training time
    print(f"\nTraining time for model: {end_time - start_time:.2f} seconds\n")

    # Testing on various amino acid sequences
    test_sequences = {
        "Example": "ARNDCEQGHILKMFPSTWYV",
        "Spike Protein (Default)": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAA",  # Replace with the actual sequence
        "Spike Protein (Moderna)": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAALERLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAA",  # Replace with the actual sequence
        "Spike Protein (Pfizer)": "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAALERLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAA"  # Replace with the actual sequence
    }

    for name, aa_seq in test_sequences.items():
        generated_codon_seq = generate_sequence(model, aa_seq)
        print(f"Generated codon sequence for {name}: {generated_codon_seq}\n")
        print("-" * 80)  # Separator

if __name__ == '__main__':
    main()

# # List of hyperparameters for models of different sizes
# hyperparameters = [
#     {"hidden_dim": 32, "n_layers": 1, "drop_prob": 0.1, "lr": 0.001},
#     {"hidden_dim": 64, "n_layers": 2, "drop_prob": 0.2, "lr": 0.001},
#     {"hidden_dim": 128, "n_layers": 2, "drop_prob": 0.2, "lr": 0.001},
#     # ... Add more combinations as you see fit
# ]

# # For each hyperparameter combination, initialize a model, train it, and evaluate its performance
# for params in hyperparameters:
#     print(f"Training model with hyperparameters: {params}")
    
#     # Initialize the model
#     model = RNNModel(
#         input_dim=4,
#         hidden_dim=params["hidden_dim"],
#         output_dim=4,
#         n_layers=params["n_layers"],
#         drop_prob=params["drop_prob"],
#         optimizer_cls=torch.optim.Adam,
#         lr=params["lr"],
#         loss_fn=F.cross_entropy
#     )
    
#     # Train the model
#     start_time = time.time()
#     train_model(model, train_loader, val_loader, num_epochs=10)
#     end_time = time.time()
    
#     # Print out the training time
#     print(f"Training time for model: {end_time - start_time} seconds\n")

#     # For evaluation: you can also generate some sequences to visually inspect them
#     test_amino_acid_seq = "ARNDCEQGHILKMFPSTWYV"  # Example amino acid sequence
#     generated_codon_seq = generate_sequence(model, test_amino_acid_seq)
#     print(f"Generated codon sequence for test amino acid sequence: {generated_codon_seq}\n")
#     print("-" * 80)  # Separator



# # Define the RNN model with some arbitrary parameters as an example
# example_rnn = RNNModel(input_dim=4, hidden_dim=64, output_dim=4, n_layers=2, drop_prob=0.2, optimizer_cls=torch.optim.Adam, lr=0.001, loss_fn=F.cross_entropy)

# example_rnn
