from Bio import SeqIO
import pandas as pd

def fasta_to_dataframe(fasta_file):
    # Parse the FASTA file
    records = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Extract ids and sequences
    ids = [record.id for record in records]
    sequences = [str(record.seq) for record in records]
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'id': ids,
        'sequence': sequences
    })
    
    return df

# Load data
path = "../data/gencode/gencode.v44.pc_translations.fa"
df = fasta_to_dataframe(path)

# Count number of sequences
count = df.shape[0]
print(f"Total number of translation sequences: {count}")

# If you want to see the first few rows of the dataframe
print(df.head())
