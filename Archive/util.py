# filename: util.py

from Bio import SeqIO
import pandas as pd
import os
import random
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_sequences, tgt_sequences = zip(*batch)
    # Padding sequences
    src_sequences = pad_sequence(src_sequences, batch_first=True)
    tgt_sequences = pad_sequence(tgt_sequences, batch_first=True)
    return src_sequences, tgt_sequences


# Define amino acids and their integer representations
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'  # Standard 20 amino acids
AMINO_TO_INT = {amino: idx for idx, amino in enumerate(AMINO_ACIDS)}

# Define codons and their integer representations
CODONS = [
    'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 
    'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
    'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT',
    'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
    'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT',
    'TTA', 'TTC', 'TTG', 'TTT'
]
CODON_TO_INT = {codon: idx for idx, codon in enumerate(CODONS)}

def encode_amino_sequence(sequence):
    return [AMINO_TO_INT[amino] for amino in sequence]

def encode_codon_sequence(sequence):
    return [CODON_TO_INT[sequence[i:i+3]] for i in range(0, len(sequence), 3)]

# Paths
gencode_source_file_path = '../data/gencode/gencode.v44.pc_transcripts.fa'
simple_output_path = '../data/1_simple_parsed_records.csv'
elimination_output_path = '../data/1_elimination_parsed_records.csv'
filled_output_path = '../data/1_filled_parsed_records.csv'

# Codon table data as a map
amino_acid_to_codon_mapping = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],  # Alanine
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
    'N': ['AAC', 'AAT'],  # Asparagine
    'D': ['GAT', 'GAC'],  # Aspartic acid
    'C': ['TGC', 'TGT'],  # Cysteine
    'E': ['GAA', 'GAG'],  # Glutamic acid
    'Q': ['CAA', 'CAG'],  # Glutamine
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],  # Glycine
    'H': ['CAT', 'CAC'],  # Histidine
    'I': ['ATT', 'ATC', 'ATA'],  # Isoleucine
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],  # Leucine
    'K': ['AAA', 'AAG'],  # Lysine
    'M': ['ATG'],  # Methionine (start codon)
    'F': ['TTT', 'TTC'],  # Phenylalanine
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],  # Proline
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'SCT', 'SEC'],  # Serine; Note: 'SCU' and 'SEC' don't exist, you might want to verify these
    'T': ['TAT', 'TAC'],  # Threonine
    'W': ['TGG'],  # Tryptophan
    'Y': ['TAT', 'TAC'],  # Tyrosine; Note: This is the same as Threonine, you might want to verify this
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],  # Valine
    'Stop': ['TAA', 'TAG', 'TGA']  # Stop codons
    
    # 'A': ['GCU', 'GCC', 'GCA', 'GCG'],  # Alanine
    # 'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
    # 'N': ['AAC', 'AAT'],  # Asparagine
    # 'D': ['GAU', 'GAT'],  # Aspartic acid
    # 'C': ['UGU', 'UGC'],  # Cysteine
    # 'E': ['GAA', 'GAG'],  # Glutamic acid
    # 'Q': ['CAA', 'CAG'],  # Glutamine
    # 'G': ['GGU', 'GGC', 'GGA', 'GGG'],  # Glycine
    # 'H': ['CAU', 'CAC'],  # Histidine
    # 'I': ['AUU', 'AUC', 'AUA'],  # Isoleucine
    # 'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],  # Leucine
    # 'K': ['AAA', 'AAG'],  # Lysine
    # 'M': ['AUG'],  # Methionine (start codon)
    # 'F': ['UUU', 'UUC'],  # Phenylalanine
    # 'P': ['CCU', 'CCC', 'CCA', 'CCG'],  # Proline
    # 'S': ['UCU', 'UCC', 'UCA', 'UCG', 'SCU', 'SEC'],  # Serine
    # 'T': ['UAU', 'UAC'],  # Threonine
    # 'W': ['UGG'],  # Tryptophan
    # 'Y': ['UAU', 'UAC'],  # Tyrosine
    # 'V': ['GUU', 'GUC', 'GUA', 'GUG'],  # Valine
    # 'Stop': ['UAA', 'UAG', 'UGA']  # Stop codons
    
}

# Create a list of all codons
all_codons = [codon for codons in amino_acid_to_codon_mapping.values() for codon in codons]

# Create mappings
index_to_codon = {index: codon for index, codon in enumerate(all_codons)}
codon_to_index = {codon: index for index, codon in enumerate(all_codons)}


# Helper functions for amino acid to codon conversion and vice versa
def amino_acid_to_codon(amino_acid_seq):
    """Convert an amino acid sequence to its corresponding codon sequence."""
    return ''.join(random.choice(amino_acid_to_codon_mapping[aa]) for aa in amino_acid_seq)

def codon_to_amino_acid(codon_seq):
    """Convert a codon sequence to its corresponding amino acid sequence."""
    codon_to_aa = {codon: aa for aa, codons in amino_acid_to_codon_mapping.items() for codon in codons}
    return ''.join(codon_to_aa[codon_seq[i:i+3]] for i in range(0, len(codon_seq), 3))

def predictions_to_codon_sequence(predictions):
    # Convert the one-hot encoded tensor to indices of max values
    _, indices = torch.max(predictions, dim=-1)
    # Convert indices to codons and then concatenate them to get the sequence
    codon_seq = ''.join([index_to_codon[idx.item()] for idx in indices.numpy()])
    return codon_seq

# Load sequences
def load_sequences(source_file='../data/gencode/gencode.v44.pc_transcripts.fa'):

    # Check if file exists
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} not found.")   

    parsed_gencode = basic_fasta_parser(source_file)
    df = pd.DataFrame(parsed_gencode)
    
    # Check for required columns
    if 'sequence' not in df.columns or 'CDS' not in df.columns:
        raise ValueError("Expected columns 'sequence' and 'CDS' not found in the DataFrame.")
    
    # Extract start and end positions and handle potential format errors
    try:
        df['cds_start'] = df['CDS'].str.split('-').str[0].astype(int)
        df['cds_end'] = df['CDS'].str.split('-').str[1].astype(int)
    except:
        raise ValueError("Error in parsing 'CDS' column. Ensure it has the format 'start-end'.")
    
    # Filter out invalid rows using vectorized operations
    valid_rows = (df['cds_start'] > 0) & (df['cds_end'] <= df['sequence'].str.len())
    valid_df = df[valid_rows]
    
    # Extract sequences using vectorized operations
    # sequences = (valid_df['sequence'].str.slice(start=valid_df['cds_start']-1, stop=valid_df['cds_end'])).tolist()
    sequences = valid_df.apply(lambda row: row['sequence'][row['cds_start']-1:row['cds_end']], axis=1).tolist()


    return sequences

def load_src_tgt_sequences(source_file='../data/gencode/gencode.v44.pc_transcripts.fa'):
    """
    Loads source (amino acid) and target (codon) sequences from the specified file.

    Args:
    - source_file (str): Path to the input fasta file.

    Returns:
    - tuple: A tuple containing two lists - source sequences (amino acids) and target sequences (codons).
    """
    
    # Check if file exists
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} not found.")
    
    # Parse the fasta file and convert it to a dataframe
    parsed_gencode = basic_fasta_parser(source_file)
    df = pd.DataFrame(parsed_gencode)
    
    # Validate required columns in dataframe
    if 'sequence' not in df.columns or 'CDS' not in df.columns:
        raise ValueError("Expected columns 'sequence' and 'CDS' not found in the DataFrame.")
    
    # Extract start and end positions for CDS
    try:
        df['cds_start'] = df['CDS'].str.split('-').str[0].astype(int)
        df['cds_end'] = df['CDS'].str.split('-').str[1].astype(int)
    except:
        raise ValueError("Error in parsing 'CDS' column. Ensure it has the format 'start-end'.")
    
    # Filter out invalid rows
    valid_rows = (df['cds_start'] > 0) & (df['cds_end'] <= df['sequence'].str.len())
    valid_df = df[valid_rows]
    
    # Extract codons using the start and end indices
    tgt_sequences = [seq[cds_start-1:cds_end] for seq, cds_start, cds_end in zip(valid_df['sequence'], valid_df['cds_start'], valid_df['cds_end'])]
    
    # Convert codons to amino acids (assuming you have a function for that)
    src_sequences = [codon_to_amino_acid(seq) for seq in tgt_sequences]
    
    # Convert amino acid sequences to their integer representations
    src_sequences_encoded = [encode_amino_sequence(seq) for seq in src_sequences]

    # Convert codon sequences to their integer representations
    tgt_sequences_encoded = [encode_codon_sequence(seq) for seq in tgt_sequences]

    # Return the sequences
    return src_sequences_encoded, tgt_sequences_encoded

def collate_fn(batch):
    src_sequences, tgt_sequences = zip(*batch)
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

# def collate_fn(batch):
#     # Sort the batch by the length of the sequences in descending order
#     sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
#     # Separate the sequences and their corresponding labels
#     sequences, labels = zip(*sorted_batch)
    
#     # Pad the sequences
#     sequences_padded = pad_sequence([torch.Tensor(seq) for seq in sequences], batch_first=True)
#     labels_padded = pad_sequence([torch.Tensor(label) for label in labels], batch_first=True)
    
#     # Return the padded sequences and their corresponding labels
#     return sequences_padded, labels_padded




# One-hot encoding for the sequences
def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return [mapping[base] for base in sequence]

def countCodons(sequence):
    codon_count = {}
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        if codon in codon_count:
            codon_count[codon] += 1
        else:
            codon_count[codon] = 1
    return codon_count

def countNonUniqueSequences(filename):

    # Parse the fasta file and convert it to a list
    sequences = list(SeqIO.parse(filename, 'fasta'))

    # Count the number of sequences (including duplicates)
    return len(sequences)

def countUniqueSequences(filename):
    
    # Parse the fasta file
    sequences = SeqIO.parse(filename,'fasta')
    # Create a set to store unique sequences
    unique_sequences = set()

    for record in sequences:
        # Add the sequence to the set
        unique_sequences.add(str(record.seq))

    # Count the number of unique sequences
    return len(unique_sequences)

def find_coding_sequences(sequence_record):
    # Define start and stop codons
    start_codon = 'ATG'
    stop_codons = ['TAA', 'TAG', 'TGA']

    coding_sequences = []

    # Find all possible ORFs
    for frame in [0, 1, 2]:
        for i in range(frame, len(sequence_record.seq) - 2, 3):
            codon = sequence_record.seq[i:i+3]
            if codon == start_codon:
                # Start codon found, search for stop codon
                coding_sequence = start_codon
                j = i + 3
                while j < len(sequence_record.seq) - 2:
                    codon = sequence_record.seq[j:j+3]
                    if codon in stop_codons:
                        coding_sequence += codon
                        break
                    coding_sequence += codon
                    j += 3
                if len(coding_sequence) > 3:  # Ignore sequences less than 1 codon
                    coding_sequences.append(coding_sequence)

    return coding_sequences

def extract_coding_sequences_with_defline(sequence_record):
    defline_info = sequence_record.description.split()  # Split defline by spaces
    cds_start = int(defline_info[defline_info.index('CDS_start=X') + 1])  # Replace X with the actual key
    cds_end = int(defline_info[defline_info.index('CDS_end=Y') + 1])  # Replace Y with the actual key

    coding_sequence = sequence_record.seq[cds_start - 1:cds_end]  # Adjust for 0-based indexing

    return coding_sequence

# Previously in GenCode
def extract_first_100_records(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        count = 0
        for line in infile:
            if line.startswith(">"):
                if count == 100:
                    break
                count += 1
            outfile.write(line)

# Identifies start & stop codons, returns indices
def identify_codons(sequence):
    """Identifies the start and stop codons for a given sequence."""
    start_codon = 'ATG'
    stop_codons = ['TAA', 'TAG', 'TGA']

    start_position = sequence.find(start_codon)
    if start_position == -1:
        return None, None

    for i in range(start_position + 3, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in stop_codons:
            return start_position, i
    return start_position, None

def basic_fasta_parser(file_path):
    """
    Parses the content of a FASTA file without using external libraries like Biopython.

    Parameters:
    - file_path (str): Path to the FASTA file to be parsed.

    Returns:
    - list[dict]: A list of dictionaries, each representing a record (sequence entry) in the FASTA file. 
                  Each dictionary contains the following key-value pairs:
                  - "transcript_id": ID of the transcript (str).
                  - "gene_id": ID of the gene (str).
                  - "manual_gene_id": Manual ID of the gene (str).
                  - "manual_transcript_id": Manual ID of the transcript, or None if not provided (str or None).
                  - "gene_symbol_variant": Symbolic variant of the gene, or None if not provided (str or None).
                  - "gene_name": Name of the gene, or None if not provided (str or None).
                  - "sequence_length": Length of the nucleotide sequence (int).
                  - "sequence": Nucleotide sequence itself (str).
                  Dynamic fields based on prefixes in the header (like UTR5, CDS, UTR3) are also included as key-value pairs.

    Example output for a single record in the list:
    {
        "transcript_id": "ENST00000530893",
        "gene_id": "ENSG00000183888",
        "manual_gene_id": "OTTHUMG00000021207",
        "manual_transcript_id": "OTTHUMT00000057564",
        "gene_symbol_variant": "LINC00115-201",
        "gene_name": "LINC00115",
        "sequence_length": 1500,
        "sequence": "AGGTCCAGGCGTAGCATGTTTGAGCTGGTCTCAAACTCCTGACCTCGTGATCCACCCGCCTTGGCCTCCCAAAGTGCTGGGATTACAGGCATGAGCCACCGTGCCCAGCCTGGGTAACAGGCGTGAGCCACCGTGCCCAGCCT",
        "UTR5": "1..49",
        "CDS": "50..100",
        "UTR3": "101..1500"
    }

    Notes:
    - The function assumes that each sequence in the FASTA file has a unique header.
    - Dynamic fields are added to the output based on their presence in the FASTA header and their recognized prefixes.
    """
    records = []
    with open(file_path, "r") as handle:
        sequence = ""
        header = None
        for line in handle:
            line = line.strip()
            if line.startswith(">"):  # Header line
                # Save the previous record
                if header:
                    records.append({
                        "header": header,
                        "sequence": sequence
                    })
                # Start a new record
                header = line[1:]  # Remove '>'
                sequence = ""
            else:
                sequence += line
        # Save the last record
        if header:
            records.append({
                "header": header,
                "sequence": sequence
            })
    
    # Parsing header details as per the provided function logic
    parsed_records = []
    for record in records:
        header_parts = record["header"].split("|")

        # Initialize a dictionary with the fixed positions
        transcript_info = {
            "transcript_id": header_parts[0],
            "gene_id": header_parts[1],
            "manual_gene_id": header_parts[2],
            "manual_transcript_id": header_parts[3] if len(header_parts) > 3 else None,
            "gene_symbol_variant": header_parts[4] if len(header_parts) > 4 else None,
            "gene_name": header_parts[5] if len(header_parts) > 5 else None,
            "sequence_length": len(record["sequence"]),
            "sequence": record["sequence"]
        }

        # For fields that are dynamic (e.g., UTR5, CDS, UTR3), we'll detect them based on their prefixes
        for part in header_parts[7:]:
            if "UTR5" in part:
                transcript_info["UTR5"] = part.split(":")[1]
            elif "UTR3" in part:
                transcript_info["UTR3"] = part.split(":")[1]
            elif "CDS" in part:
                transcript_info["CDS"] = part.split(":")[1]
            # You can extend this with other prefixes if needed

        parsed_records.append(transcript_info)

    return parsed_records

def incompletes_elimination_parse_fasta(file_path):
    """
    Parse a FASTA file, filter out records without both UTRs, 
    and return a dictionary with transcript IDs as keys and their information as values.
    """
    parsed_records = basic_fasta_parser(file_path)
    
    # Filtering records that have both UTR5 and UTR3
    complete_records = {record["transcript_id"]: record for record in parsed_records if "UTR5" in record and "UTR3" in record}
    
    return complete_records

def filling_fasta_parser(file_path):
    """
    Parse a FASTA file and fill missing UTRs using the identify_codons algorithm.
    Returns a list of dictionaries with complete transcript information.
    """
    parsed_records = basic_fasta_parser(file_path)
    
    for record in parsed_records:
        sequence = record["sequence"]
        start_pos, stop_pos = identify_codons(sequence)
        
        # If UTR5 is missing and we have identified a start codon
        if "UTR5" not in record and start_pos is not None:
            record["UTR5"] = f"1-{start_pos-1}"
        
        # If UTR3 is missing and we have identified a stop codon
        if "UTR3" not in record and stop_pos is not None:
            record["UTR3"] = f"{stop_pos+3}-{len(sequence)}"
            
        # If CDS is missing and we have identified both start and stop codons
        if "CDS" not in record and start_pos is not None and stop_pos is not None:
            record["CDS"] = f"{start_pos+1}-{stop_pos+2}"
            
    return parsed_records


# Biopython old stuff
'''
# The below functions use biopython
def parse_fasta(file_path):
    """
    Parse a FASTA file and return a list of dictionaries with transcript information.
    """
    records = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            header_parts = record.description.split("|")

            # Initialize a dictionary with the fixed positions
            transcript_info = {
                "transcript_id": header_parts[0],
                "gene_id": header_parts[1],
                "manual_gene_id": header_parts[2],
                "manual_transcript_id": header_parts[3] if len(header_parts) > 3 else None,
                "gene_symbol_variant": header_parts[4] if len(header_parts) > 4 else None,
                "gene_name": header_parts[5] if len(header_parts) > 5 else None,
                "sequence_length": header_parts[6] if len(header_parts) > 6 else None,
                "sequence": str(record.seq)
            }

            # For fields that are dynamic (e.g., UTR5, CDS, UTR3), we'll detect them based on their prefixes
            for part in header_parts[7:]:
                if "UTR5" in part:
                    transcript_info["UTR5"] = part.split(":")[1]
                elif "UTR3" in part:
                    transcript_info["UTR3"] = part.split(":")[1]
                elif "CDS" in part:
                    transcript_info["CDS"] = part.split(":")[1]
                # You can extend this with other prefixes if needed

            records.append(transcript_info)
    return records

def fill_missing_utrs(transcript_info):
    """Fill in missing UTRs based on the sequence and existing annotations."""
    sequence = transcript_info["sequence"]
    cds = transcript_info.get("CDS", None)

    if cds:
        cds_start, cds_end = [int(x) for x in cds.split('-')]
    else:
        start, stop = identify_codons(sequence)
        if start is not None and stop is not None:
            cds_start, cds_end = start, stop + 2
            transcript_info["CDS"] = f"{cds_start + 1}-{cds_end + 1}"  # +1 for 1-indexed
        else:
            return transcript_info

    if "UTR5" not in transcript_info and cds_start != 0:
        transcript_info["UTR5"] = f"1-{cds_start}"
        transcript_info["UTR5_filled"] = 1
    if "UTR3" not in transcript_info and cds_end != len(sequence) - 1:
        transcript_info["UTR3"] = f"{cds_end + 2}-{len(sequence)}"
        transcript_info["UTR3_filled"] = 1

    return transcript_info

def parse_fasta_fill_UTR(file_path):
    records = []
    with open(file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            header_parts = record.description.split("|")

            # Initialize dictionary
            transcript_info = {
                "transcript_id": header_parts[0],
                "gene_id": header_parts[1],
                "manual_gene_id": header_parts[2],
                "manual_transcript_id": header_parts[3] if len(header_parts) > 3 else None,
                "gene_symbol_variant": header_parts[4] if len(header_parts) > 4 else None,
                "gene_name": header_parts[5] if len(header_parts) > 5 else None,
                "sequence_length": header_parts[6] if len(header_parts) > 6 else None,
                "sequence": str(record.seq)
            }

            # Detect dynamic fields
            for part in header_parts[7:]:
                if "UTR5" in part:
                    transcript_info["UTR5"] = part.split(":")[1]
                elif "UTR3" in part:
                    transcript_info["UTR3"] = part.split(":")[1]
                elif "CDS" in part:
                    transcript_info["CDS"] = part.split(":")[1]

            filled_transcript = fill_missing_utrs(transcript_info)
            records.append(filled_transcript)

    return records

def extract_first_100_records(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        count = 0
        for line in infile:
            if line.startswith(">"):
                if count == 100:
                    break
                count += 1
            outfile.write(line)

# Assuming you have the FASTA file named 'gencode.v44.transcripts.fa'
file_path = '../data/gencode.v44.pc_transcripts.fa'
filled_transcripts = parse_fasta_fill_UTR(file_path)
transcripts = parse_fasta(file_path)

extract_first_100_records(file_path, '../data/gencode_transcripts_100.fa')

#print("Number of transcripts:", len(filled_transcripts))

# Convert to DataFrame
df_filled = pd.DataFrame(filled_transcripts)
df = pd.DataFrame(transcripts)

# Display first few rows
print("Printing head")
print(df_filled.head())

# Check for missing values
print("Checking for isnull().sum()")
print("UTRs filled in, null values summary:\n" + df_filled.isnull().sum().to_string())
print("UTRS NOT FILLED IN, null values summary:\n" + df.isnull().sum().to_string())

# Summary statistics
print("Checking for summary statistics:")
print(df_filled.describe())

# Open df as a csv file
df_filled.to_csv('../data/gencode_formatted_UTR_fill.csv', index=False)
df.to_csv('../data/gencode_formatted.csv', index=False)
'''
# etc.
print("util.py runs successfully.")