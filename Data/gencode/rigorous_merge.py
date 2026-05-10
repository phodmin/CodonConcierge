from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import sys
sys.path.append('/Users/misko/Documents/_Code/CodonConcierge')
from simple_merge import fasta_to_dataframe
from util3 import extract_cds_columns
#from pandasgui import show
from IPython.display import display

protein_path = '../data/gencode/gencode.v44.pc_translations.fa'
mrna_path = '../data/gencode/gencode.v44.pc_transcripts.fa'

def extract_cds_columns(df):
    """Extract CDS start/end columns"""
    
    # Split the 'CDS' column once
    cds_splits = df['CDS'].str.split('-')
    
    # Check if all rows have exactly two parts after splitting
    valid_format = cds_splits.apply(lambda x: len(x) == 2 if x else False)

    # For rows with the valid 'start-end' format
    df.loc[valid_format, 'cds_start'] = cds_splits[valid_format].str[0].astype(int)
    df.loc[valid_format, 'cds_end'] = cds_splits[valid_format].str[1].astype(int)

    # For rows without the valid 'start-end' format or if 'CDS' is not found
    default_indices = ~valid_format | df['CDS'].isna()
    df.loc[default_indices, 'cds_start'] = 1
    df.loc[default_indices, 'cds_end'] = df.loc[default_indices, 'sequence'].str.len()

    # Ensure 'cds_start' and 'cds_end' are integers
    df['cds_start'] = df['cds_start'].astype(int)
    df['cds_end'] = df['cds_end'].astype(int)

    valid_rows = (df['cds_start'] > 0) & (df['cds_end'] <= df['sequence'].str.len())
    valid_df = df[valid_rows]
    return valid_df

def CDS_cleaning_fasta_to_dataframe(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Extract sequence ID, sequence, and CDS info (assuming CDS info is after a space in the 'id' field)
    data = {
        "id": [rec.id.split(' ')[0] for rec in records],
        "CDS": [rec.id.split(' ')[1] if ' ' in rec.id else None for rec in records],  # Handle cases without CDS info
        "sequence": [str(rec.seq) for rec in records]
    }
    df = pd.DataFrame(data)
    
    # Extract CDS start and end columns
    df = extract_cds_columns(df)
    
    # Trim the sequence based on CDS start and end
    df['sequence'] = df.apply(lambda row: row['sequence'][row['cds_start']-1:row['cds_end']], axis=1)
    
    return df[['id', 'sequence']]


protein_df = fasta_to_dataframe(protein_path)
mrna_df = CDS_cleaning_fasta_to_dataframe(mrna_path)

# Extract Ensembl transcript IDs for proteins
protein_df['transcript_id'] = protein_df['id'].str.split('|').str[1]

# Extract Ensembl transcript IDs for mRNAs (notice the change to index 0)
mrna_df['transcript_id'] = mrna_df['id'].str.split('|').str[0]

# Merge the two DataFrames based on transcript_id
merged_df = protein_df.merge(mrna_df, on='transcript_id', suffixes=('_protein', '_mrna'))

# --------------------------------  --------------------------------  --------------------------------
#                           Adding Nucleotides - Amino Acids ratio (ideally should be ±3)
# --------------------------------  --------------------------------  --------------------------------

# Calculate protein and mRNA lengths
merged_df['prot_length'] = merged_df['sequence_protein'].str.len() 
merged_df['mrna_length'] = merged_df['sequence_mrna'].str.len()

# Add N-AA ratio column
merged_df['N-AA-ratio'] = merged_df['mrna_length'] / merged_df['prot_length'] 

# Filter for valid ratios 
valid_ratios = (merged_df['N-AA-ratio'] > 2.5) & (merged_df['N-AA-ratio'] < 3.5)

# Store original length
orig_len = len(merged_df)

# Filter DataFrame
merged_df = merged_df[valid_ratios]

# Report number removed  
num_removed = orig_len - len(merged_df)
print(f"Removed {num_removed} rows with invalid ratios")

# Calculate statistics on filtered data
avg_ratio = merged_df['N-AA-ratio'].mean()
std_deviation = merged_df['N-AA-ratio'].std()

# Print statistics
print(f"\nAverage N-AA ratio: {avg_ratio}")
print(f"Standard Deviation N-AA ratio: {std_deviation}")


# --------------------------------  --------------------------------  --------------------------------
#                           Printing funky info
# --------------------------------  --------------------------------  --------------------------------



# print(merged_df.columns)

# # Display the first few rows to visually inspect the added columns
# print(merged_df[['transcript_id', 'prot_length', 'mrna_length']].head())

# # Basic statistics
# print("\n=== Basic Statistics ===")
# print(merged_df[['prot_length', 'mrna_length']].describe())

# # Count of unique protein lengths
# unique_prot_lengths = merged_df['prot_length'].nunique()
# print(f"\nNumber of unique protein lengths: {unique_prot_lengths}")

# # Count of unique mRNA lengths
# unique_mrna_lengths = merged_df['mrna_length'].nunique()
# print(f"Number of unique mRNA lengths: {unique_mrna_lengths}")

# # Top 5 most common protein lengths
# print("\nTop 5 most common protein lengths:")
# print(merged_df['prot_length'].value_counts().head(5))

# # Top 5 most common mRNA lengths
# print("\nTop 5 most common mRNA lengths:")
# print(merged_df['mrna_length'].value_counts().head(5))




# # --------------------------------  --------------------------------  --------------------------------
# #                               Saving & Viewing Data
# # --------------------------------  --------------------------------  --------------------------------
# merged_df.to_csv('merged_data.csv', index=False)