import os
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from io import StringIO
from typing import Tuple, List
from itertools import compress
from torch.nn.utils.rnn import pad_sequence
import pytest

# Paths
gencode_source_file_path = '../data/gencode/gencode.v44.pc_transcripts.fa'

# Define bases
bases = ['A', 'T', 'G', 'C', 'N']

# Mapping codons to amino acids, standard capitalised IUPAC codes
# Padded codons (any that include N) are mapped to 'X'
codon_to_aa = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    # Adding the padded codons
    'ANN':'X', 'CNN':'X', 'GNN':'X', 'TNN':'X',
    'AAN':'X', 'CAN':'X', 'GAN':'X', 'TAN':'X',
    'ANA':'X', 'CNA':'X', 'GNA':'X', 'TNA':'X',
    'ANC':'X', 'CNC':'X', 'GNC':'X', 'TNC':'X',
    'ANG':'X', 'CNG':'X', 'GNG':'X', 'TNG':'X',
    'ANT':'X', 'CNT':'X', 'GNT':'X', 'TNT':'X',
    'AGN':'X', 'CGN':'X', 'GGN':'X', 'TGN':'X',
    'ATN':'X', 'CTN':'X', 'GTN':'X', 'TTN':'X',
    'ACN':'X', 'CCN':'X', 'GCN':'X', 'TCN':'X',
    'NAA':'X', 'NAC':'X', 'NAG':'X', 'NAT':'X',
    'NCA':'X', 'NCC':'X', 'NCG':'X', 'NCT':'X',
    'NGA':'X', 'NGC':'X', 'NGG':'X', 'NGT':'X',
    'NTA':'X', 'NTC':'X', 'NTG':'X', 'NTT':'X',
    'NAN':'X', 'NCN':'X', 'NGN':'X', 'NTN':'X',
    'NNN':'X'
}

# Mapping amino acids to integers, 1-20
# Unknown Amino Acid ('X') is mapped to '0'
# Stop codon ('*') is mapped to '21'
aa_to_int = {
    'A':1, 'C':2, 'D':3, 'E':4,
    'F':5, 'G':6, 'H':7, 'I':8,
    'K':9, 'L':10, 'M':11, 'N':12,
    'P':13, 'Q':14, 'R':15, 'S':16,
    'T':17, 'V':18, 'W':19, 'Y':20,
    # Unknown Amino Acid
    'X':0, '*':21
}

# Mapping codons to ints, 1-64
# Padded codons (any that include N) are mapped to '0'
codon_to_int = {
    'ATA':1, 'ATC':2, 'ATT':3, 'ATG':4,
    'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8,
    'AAT':9, 'AAC':10, 'AAA':11, 'AAG':12,
    'AGA':13, 'AGC':14, 'AGG':15, 'AGT':16,
    'CTA':17, 'CTC':18, 'CTT':19, 'CTG':20,
    'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24,
    'CAT':25, 'CAC':26, 'CAA':27, 'CAG':28,
    'CGA':29, 'CGC':30, 'CGG':31, 'CGT':32,
    'GTA':33, 'GTC':34, 'GTT':35, 'GTG':36,
    'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40,
    'GAT':41, 'GAC':42, 'GAA':43, 'GAG':44,
    'GGA':45, 'GGC':46, 'GGG':47, 'GGT':48,
    'TCA':49, 'TCC':50, 'TCT':51, 'TCG':52,
    'TTA':53, 'TTC':54, 'TTT':55, 'TTG':56,
    'TAT':57, 'TAC':58, 'TAA':59, 'TAG':60,
    'TGA':61, 'TGC':62, 'TGG':63, 'TGT':64,
    # Adding the padded codons
    'ANN':0, 'CNN':0, 'GNN':0, 'TNN':0,
    'AAN':0, 'CAN':0, 'GAN':0, 'TAN':0,
    'ANA':0, 'CNA':0, 'GNA':0, 'TNA':0,
    'ANC':0, 'CNC':0, 'GNC':0, 'TNC':0,
    'ANG':0, 'CNG':0, 'GNG':0, 'TNG':0,
    'ANT':0, 'CNT':0, 'GNT':0, 'TNT':0,
    'AGN':0, 'CGN':0, 'GGN':0, 'TGN':0,
    'ATN':0, 'CTN':0, 'GTN':0, 'TTN':0,
    'ACN':0, 'CCN':0, 'GCN':0, 'TCN':0,
    'NAA':0, 'NAC':0, 'NAG':0, 'NAT':0,
    'NCA':0, 'NCC':0, 'NCG':0, 'NCT':0,
    'NGA':0, 'NGC':0, 'NGG':0, 'NGT':0,
    'NTA':0, 'NTC':0, 'NTG':0, 'NTT':0,
    'NAN':0, 'NCN':0, 'NGN':0, 'NTN':0,
    'NNN':0
}


# 0
def load_src_tgt_sequences(source_file: str, max_seq_length: int = 120000) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Load source and target sequences from a FASTA file and encode them into numerical sequences.

    Args:
        source_file (str): Path to the source FASTA file.
        max_seq_length (int): Maximum length of the target sequences in nucleotides.

    Returns:
        Tuple of two numpy arrays:
        - aa_enc: Encoded amino acid sequences.
        - codon_enc: Encoded codon sequences.
    """
    # Input validation
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} not found")  
    
    df = parse_fasta(source_file)

    # Data extraction
    df = extract_cds_columns(df)
    aa_seqs, codon_seqs = extract_sequences(df)

    # Filter sequences based on max_seq_length
    valid_seq_mask = [(len(seq) * 3 <= max_seq_length) for seq in codon_seqs]
    aa_seqs = list(compress(aa_seqs, valid_seq_mask))
    codon_seqs = list(compress(codon_seqs, valid_seq_mask))

    # Sequence encoding
    aa_enc = encode_amino_sequence(aa_seqs) 
    codon_enc = encode_codon_sequence(codon_seqs)

    return aa_enc, codon_enc

# 1
def parse_fasta(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    parsed_records = []
    for record in records:
        header_parts = record.description.split("|")
        transcript_info = {
            "transcript_id": header_parts[0],
            "gene_id": header_parts[1],
            "manual_gene_id": header_parts[2],
            "manual_transcript_id": header_parts[3],
            "gene_symbol_variant": header_parts[4],
            "gene_name": header_parts[5],
            "sequence_length": int(header_parts[6]),
            "UTR5": header_parts[7].split(":")[1] if len(header_parts) > 7 and "UTR5" in header_parts[7] else None,
            "CDS": header_parts[8].split(":")[1] if len(header_parts) > 8 and "CDS" in header_parts[8] else None,
            "UTR3": header_parts[9].split(":")[1] if len(header_parts) > 9 and "UTR3" in header_parts[9] else None,

            "sequence": str(record.seq)
        }
        parsed_records.append(transcript_info)
        
    df = pd.DataFrame(parsed_records)
    return df

# 2
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

# Legacy extract CDS
'''
def extract_cds_columns(df):
    """Extract CDS start/end columns"""
    
    # Split the 'CDS' column once
    cds_splits = df['CDS'].str.split('-')
    
    # Check if all rows have exactly two parts after splitting
    valid_format = cds_splits.apply(lambda x: len(x) == 2 if x else False)
    if not all(valid_format):
        # Find and report problematic rows
        problem_rows = df[~valid_format]
        problem_indices = problem_rows.index.tolist()
        problem_values = problem_rows['CDS'].tolist()
        raise ValueError(f"Error in parsing 'CDS' column. Rows {problem_indices} have problematic values: {problem_values}. Ensure all rows have the format 'start-end'.")
    
    try:
        df['cds_start'] = cds_splits.str[0].astype(int)
        df['cds_end'] = cds_splits.str[1].astype(int)
    except TypeError as e:
        # Catch specific exception and raise with additional information
        raise ValueError(f"Error converting CDS values to integers. Original error: {e}")
    
    valid_rows = (df['cds_start'] > 0) & (df['cds_end'] <= df['sequence'].str.len())
    valid_df = df[valid_rows]
    return valid_df
'''

# 3

# 3.1 Codons -> amino acids
def translate_codons_to_amino_acids(codon_seqs: List[str]) -> List[str]:
    """
    Translate a list of codon sequences to their corresponding amino acid sequences.

    If the codon sequence length isn't a multiple of 3, it will be padded with 'N' 
    to the nearest multiple of 3.

    Parameters:
    - codon_seqs (List[str]): A list of codon sequences. 
                              Each codon is expected to be a triplet of nucleotide bases.

    Returns:
    - List[str]: A list of amino acid sequences corresponding to the input codon sequences.
    
    Raises:
    - ValueError: If a provided codon is not recognized.
    """

    codon_to_aa = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
        # Adding the padded codons
        'ANN':'X', 'CNN':'X', 'GNN':'X', 'TNN':'X',
        'AAN':'X', 'CAN':'X', 'GAN':'X', 'TAN':'X',
        'ANA':'X', 'CNA':'X', 'GNA':'X', 'TNA':'X',
        'ANC':'X', 'CNC':'X', 'GNC':'X', 'TNC':'X',
        'ANG':'X', 'CNG':'X', 'GNG':'X', 'TNG':'X',
        'ANT':'X', 'CNT':'X', 'GNT':'X', 'TNT':'X',
        'AGN':'X', 'CGN':'X', 'GGN':'X', 'TGN':'X',
        'ATN':'X', 'CTN':'X', 'GTN':'X', 'TTN':'X',
        'ACN':'X', 'CCN':'X', 'GCN':'X', 'TCN':'X',
        'NAA':'X', 'NAC':'X', 'NAG':'X', 'NAT':'X',
        'NCA':'X', 'NCC':'X', 'NCG':'X', 'NCT':'X',
        'NGA':'X', 'NGC':'X', 'NGG':'X', 'NGT':'X',
        'NTA':'X', 'NTC':'X', 'NTG':'X', 'NTT':'X',
        'NAN':'X', 'NCN':'X', 'NGN':'X', 'NTN':'X',
        'NNN':'X'
    }   

    result = []
    
    for seq in codon_seqs:
        # Pad with 'N' if not multiple of 3
        while len(seq) % 3 != 0:
            seq += 'N'
            
        amino_acid_seq = ""
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            if codon not in codon_to_aa:
                raise ValueError(f"Unrecognized codon: {codon}")
            amino_acid_seq += codon_to_aa[codon]
        
        result.append(amino_acid_seq)

    return result

# 3.2 Amino acids -> ints
def translate_amino_acids_to_ints(aa_seqs: List[str]) -> List[List[int]]:
    """
    Translate a list of amino acid sequences to their corresponding integer sequences.

    Parameters:
    - aa_seqs (List[str]): A list of amino acid sequences. 
                           Each amino acid is represented as a single character.

    Returns:
    - List[List[int]]: A list of integer sequences corresponding to the input amino acid sequences.
    
    Raises:
    - ValueError: If a provided amino acid is not recognized.
    """
    # Mapping amino acids to integers, 1-20
    # Unknown Amino Acid ('X') is mapped to '0'
    # Stop codon ('*') is mapped to '21'
    aa_to_int = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4,
        'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L':10, 'M':11, 'N':12,
        'P':13, 'Q':14, 'R':15, 'S':16,
        'T':17, 'V':18, 'W':19, 'Y':20,
        # Unknown and stop codon
        'X': 0, '*':21
    }


    result = []
    
    for seq in aa_seqs:
        int_seq = []
        for aa in seq:
            if aa not in aa_to_int:
                raise ValueError(f"Unrecognized amino acid: {aa}")
            int_seq.append(aa_to_int[aa])
        
        result.append(int_seq)

    return result

# 3.3 Codons -> ints
def translate_codons_to_ints(codon_seqs: List[str]) -> List[int]:
    """
    Translate a list of codon sequences to their corresponding integer values.

    If the codon sequence length isn't a multiple of 3, it will be padded with 'N' 
    to the nearest multiple of 3.

    Parameters:
    - codon_seqs (List[str]): A list of codon sequences. 
                              Each codon is expected to be a triplet of nucleotide bases.

    Returns:
    - List[int]: A list of integer values corresponding to the input codon sequences.
    
    Raises:
    - ValueError: If a provided codon is not recognized.
    """

    codon_to_int = {
        'ATA': 1, 'ATC': 2, 'ATT': 3, 'ATG': 4,
        'ACA': 5, 'ACC': 6, 'ACG': 7, 'ACT': 8,
        'AAT': 9, 'AAC':10, 'AAA':11, 'AAG':12,
        'AGA':13, 'AGC':14, 'AGG':15, 'AGT':16,
        'CTA':17, 'CTC':18, 'CTT':19, 'CTG':20,
        'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24,
        'CAT':25, 'CAC':26, 'CAA':27, 'CAG':28,
        'CGA':29, 'CGC':30, 'CGG':31, 'CGT':32,
        'GTA':33, 'GTC':34, 'GTT':35, 'GTG':36,
        'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40,
        'GAT':41, 'GAC':42, 'GAA':43, 'GAG':44,
        'GGA':45, 'GGC':46, 'GGG':47, 'GGT':48,
        'TCA':49, 'TCC':50, 'TCT':51, 'TCG':52,
        'TTA':53, 'TTC':54, 'TTT':55, 'TTG':56,
        'TAT':57, 'TAC':58, 'TAA':59, 'TAG':60,
        'TGA':61, 'TGC':62, 'TGG':63, 'TGT':64,
        # Adding the padded codons
        'ANN':0, 'CNN':0, 'GNN':0, 'TNN':0,
        'AAN':0, 'CAN':0, 'GAN':0, 'TAN':0,
        'ANA':0, 'CNA':0, 'GNA':0, 'TNA':0,
        'ANC':0, 'CNC':0, 'GNC':0, 'TNC':0,
        'ANG':0, 'CNG':0, 'GNG':0, 'TNG':0,
        'ANT':0, 'CNT':0, 'GNT':0, 'TNT':0,
        'AGN':0, 'CGN':0, 'GGN':0, 'TGN':0,
        'ATN':0, 'CTN':0, 'GTN':0, 'TTN':0,
        'ACN':0, 'CCN':0, 'GCN':0, 'TCN':0,
        'NAA':0, 'NAC':0, 'NAG':0, 'NAT':0,
        'NCA':0, 'NCC':0, 'NCG':0, 'NCT':0,
        'NGA':0, 'NGC':0, 'NGG':0, 'NGT':0,
        'NTA':0, 'NTC':0, 'NTG':0, 'NTT':0,
        'NAN':0, 'NCN':0, 'NGN':0, 'NTN':0,
        'NNN':0
    }

    result = []

    for seq in codon_seqs:
        # Pad with 'N' if not multiple of 3
        while len(seq) % 3 != 0:
            seq += 'N'
            
        int_values = []
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            if codon not in codon_to_int:
                raise ValueError(f"Unrecognized codon: {codon}")
            int_values.append(codon_to_int[codon])

        result.append(int_values)

    return result

# 4
def extract_sequences(df) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extracts amino acid and codon sequences from the 'sequence' field in the DataFrame.

    Args:
        df: A pandas DataFrame containing the 'sequence', 'cds_start', and 'cds_end' columns.

    Returns:
        A tuple containing two lists:
        - aa_seqs_int: A list of amino acid sequences as integers.
        - codon_seqs_int: A list of codon sequences as integers.
    """

    aa_seqs_int = []   # For storing amino acid sequences as integers
    codon_seqs_int = []   # For storing codon sequences as integers
    
    for _, row in df.iterrows():
        seq = row['sequence'][row['cds_start']-1:row['cds_end']]  # -1 because Python is 0-based
        
        # Extracting codons
        codons = [seq[i:i+3] for i in range(0, len(seq), 3) if 1 <= len(seq[i:i+3]) <= 3]

        # Getting the amino acid integer sequences
        aa_seqs = translate_codons_to_amino_acids(codons)
        aa_ints = [aa for seq in translate_amino_acids_to_ints(aa_seqs) for aa in seq]
        
        # Getting the codon integer sequences
        codon_ints = [codon for seq in translate_codons_to_ints(codons) for codon in seq]
        
        codon_seqs_int.append(codon_ints)
        aa_seqs_int.append(aa_ints)

    return aa_seqs_int, codon_seqs_int


# 5 Dummy encode functions
def encode_amino_sequence(aa_seqs: List[List[int]]) -> List[List[int]]:
    return aa_seqs

def encode_codon_sequence(codon_seqs: List[List[int]]) -> List[List[int]]:
    return codon_seqs

# Legacy encode functions
'''
# 4
def encode_amino_acids(aa_seqs: List[str]) -> List[List[int]]:
    """Encodes a list of amino acid sequences into their corresponding integer sequences."""
    return translate_amino_acids_to_ints(aa_seqs)

def encode_codons(codon_seqs: List[str]) -> List[List[int]]:
    """Encodes a list of codon sequences into their corresponding integer sequences."""
    return translate_codons_to_ints(codon_seqs)

def encode_amino_sequence(aa_seqs):
    """Integer encode amino acid sequences"""
    unique_amino_acids = sorted(set(codon_to_aa.values()))
    aa_to_int = {aa: i for i, aa in enumerate(unique_amino_acids)}

    if isinstance(aa_seqs, str):
        aa_seqs = [aa_seqs]

    if not all(isinstance(seq, str) for seq in aa_seqs):
        raise TypeError("All sequences should be of type string")

    encoded_aa = []
    for seq in aa_seqs:
        encoded_seq = [aa_to_int.get(aa, len(aa_to_int)) for aa in seq]  # use the next integer for unknown AAs
        encoded_aa.append(encoded_seq)

    return encoded_aa

# 5
def encode_codon_sequence(codon_seqs):
    """Integer encode codon sequences"""
    unique_codons = sorted(set(codon_to_aa.keys()))
    codon_to_int = {codon: i for i, codon in enumerate(unique_codons)}

    if isinstance(codon_seqs, str):
        codon_seqs = [codon_seqs]

    if not all(isinstance(seq, str) for seq in codon_seqs):
        raise TypeError("All sequences should be of type string (Encode Codon error)")

    encoded_codons = []
    for seq in codon_seqs:
        encoded_seq = []
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            encoded_seq.append(codon_to_int.get(codon, len(codon_to_int)))  # use the next integer for unknown codons
        encoded_codons.append(encoded_seq)

    return encoded_codons
'''

# 6
def collate_fn(batch):
    src_sequences, tgt_sequences = zip(*batch)
    # Padding sequences
    src_sequences = pad_sequence(src_sequences, batch_first=True)
    tgt_sequences = pad_sequence(tgt_sequences, batch_first=True)
    return src_sequences, tgt_sequences

if __name__ == "__main__":

    
    exit()

    # Parse the FASTA file
    data_df = parse_fasta(gencode_source_file_path)
    print("\nParsed FASTA Data:")
    print(data_df.head())

    # Extract coding sequences
    cds_sequences = extract_cds_sequences(data_df)
    print("\nCDS Sequences (first 5):")
    print(cds_sequences[:5])

    # Convert to amino acids
    aa_sequences = extract_amino_acids_from_codons(cds_sequences)
    print("\nAmino Acid Sequences (first 5):")
    print(aa_sequences[:5])
