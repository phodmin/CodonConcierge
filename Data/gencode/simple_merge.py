from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd

protein_path = '../data/gencode/gencode.v44.pc_translations.fa'
mrna_path = '../data/gencode/gencode.v44.pc_transcripts.fa'

def fasta_to_dataframe(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    data = {
        "id": [rec.id for rec in records],
        "sequence": [str(rec.seq) for rec in records]
    }
    return pd.DataFrame(data)

def merge_gencode_fastaframes(protein_df, mrna_df):
    # Extract Ensembl transcript IDs for proteins
    protein_df['transcript_id'] = protein_df['id'].str.split('|').str[1]

    # Extract Ensembl transcript IDs for mRNAs (notice the change to index 0)
    mrna_df['transcript_id'] = mrna_df['id'].str.split('|').str[0]

    # Merge the two DataFrames based on transcript_id
    merged_df = protein_df.merge(mrna_df, on='transcript_id', suffixes=('_protein', '_mrna'))
    return merged_df

#if main == "__main__":
# write if main functionality below
if __name__ == "__main__":
    print("Running simple_merge.py as main")
    protein_df = fasta_to_dataframe(protein_path)
    mrna_df = fasta_to_dataframe(mrna_path)

    # Merge the two DataFrames based on transcript_id
    merged_df = merge_gencode_fastaframes(protein_df, mrna_df)

    # Basic Properties
    print("\nMerged DataFrame Properties:")
    print(f"Shape: {merged_df.shape}")
    print(f"Columns: {merged_df.columns}")
    print(f"Index: {merged_df.index}")
    print(f"Data types:\n{merged_df.dtypes}")

    # Viewing Data
    print("\nMerged DataFrame First 5 Rows:")
    print(merged_df.head())
    print("\nMerged DataFrame Last 5 Rows:")
    print(merged_df.tail())
    print("\nMerged DataFrame Random 5 Rows:")
    print(merged_df.sample(5))

    # Creating a super simple DataFrame
    simplified_df = merged_df[['sequence_protein', 'sequence_mrna']]
    print(simplified_df.head())

    simplified_df['prot_length'] = simplified_df['sequence_protein'].apply(len)
    simplified_df['mrna_length'] = simplified_df['sequence_mrna'].apply(len)

    print(simplified_df.head())

    # Print a distrubtion of protein lengths
    print(simplified_df['prot_length'].describe())

    #Print a distrubtion of mRNA lengths
    print(simplified_df['mrna_length'].describe())







