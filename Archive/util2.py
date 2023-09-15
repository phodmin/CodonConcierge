import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import string
import itertools

# Paths
gencode_source_file_path = '../data/gencode/gencode.v44.pc_transcripts.fa'

# Define bases and possible codons
bases = ['A', 'T', 'G', 'C', 'N']
all_combinations = [''.join(codon) for codon in itertools.product(bases, repeat=3)]
codons = [codon for codon in all_combinations if 0 < codon.count('N') < 3]

# Mapping codons to integers
codon_to_int = {codon: i for i, codon in enumerate(sorted(set(codons)))}

# Mapping amino acids to integers
amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
aa_to_int = {aa: i for i, aa in enumerate(sorted(set(amino_acids)))}

# Codon to Amino Acid mapping
codon_to_aa = {codon: aa for codon, aa in zip(codons, amino_acids)}


def load_src_tgt_sequences(source_file=gencode_source_file_path):
    
    # Input validation
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} not found")  

    df = parse_fasta(source_file)

    # Data extraction
    df = extract_cds_columns(df)
    aa_seqs, codon_seqs = extract_sequences(df)

    # Checking for sequences that are lists
    for idx, seq in enumerate(codon_seqs):
        if isinstance(seq, list):
            # Print the index, the first 5 items of the list, and its length
            print(f"Index {idx}: First 5 items {seq[:5]}... (total length: {len(seq)})")
            break  # we break after finding the first one to reduce output

    # Sequence encoding
    aa_enc = encode_amino_sequence(aa_seqs) 
    codon_enc = encode_codon_sequence(codon_seqs)

    return aa_enc, codon_enc

def parse_fasta(fasta_file=gencode_source_file_path):
    """
    Parse a FASTA file and return the parsed data as a DataFrame.
    
    Parameters:
    - fasta_file: Path to the FASTA file to be parsed.
    
    Returns:
    - DataFrame containing parsed FASTA data.
    """
    
    records = []
    with open(fasta_file, "r") as handle:
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

    # Parsing header details
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
        for part in header_parts[6:]:
            if "UTR5" in part:
                transcript_info["UTR5"] = part.split(":")[1]
            elif "UTR3" in part:
                transcript_info["UTR3"] = part.split(":")[1]
            elif "CDS" in part:
                transcript_info["CDS"] = part.split(":")[1]
            # You can extend this with other prefixes if needed

        parsed_records.append(transcript_info)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(parsed_records)

    return df

def extract_cds_columns(df):
    """Extract CDS start/end columns"""
    # Extract start and end positions for CDS
    try:
        df['cds_start'] = df['CDS'].str.split('-').str[0].astype(int)
        df['cds_end'] = df['CDS'].str.split('-').str[1].astype(int)
    except:
        raise ValueError("Error in parsing 'CDS' column. Ensure it has the format 'start-end'.")
    
    # Filter out invalid rows
    valid_rows = (df['cds_start'] > 0) & (df['cds_end'] <= df['sequence'].str.len())
    valid_df = df[valid_rows]

    return valid_df

def extract_sequences(df):
    """Extract AA and codon sequences from the 'sequence' field in the DataFrame"""
    aa_seqs = []
    codon_seqs = []
    for _, row in df.iterrows():
        seq = row['sequence'][row['cds_start']-1:row['cds_end']]  # -1 because Python is 0-based

        # Pad the sequence with 'N' so that its length is a multiple of 3
        while len(seq) % 3 != 0:
            seq += 'N'
        
        aa_seq = [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3]
        aa_seqs.append(''.join([codon_to_aa.get(codon, 'X') for codon in aa_seq]))  # Convert to amino acids, 'X' for unknown
        codon_seqs.append(aa_seq)

    return aa_seqs, codon_seqs


def encode_amino_sequence(aa_seqs):
    """Integer encode amino acid sequences"""

    # Check if input is a single string or a single list of integers
    if isinstance(aa_seqs, str):
        aa_seqs = [aa_seqs]
    elif isinstance(aa_seqs, list) and all(isinstance(i, int) for i in aa_seqs):
        return aa_seqs  # Return already encoded sequence

    # Guard against non-string sequences
    if not all(isinstance(seq, str) for seq in aa_seqs):
        raise TypeError("All sequences should be of type string")

    encoded_aa = []
    for seq in aa_seqs:
        encoded_seq = [aa_to_int[aa] for aa in seq if aa in aa_to_int]
        encoded_aa.append(encoded_seq)

    return encoded_aa

def encode_codon_sequence(codon_seqs):
    """Integer encode codon sequences"""
    codon_to_int = {codon: i for i, codon in enumerate(sorted(set(codons)))}

    # Check if input is a single string or a single list of integers
    if isinstance(codon_seqs, str):
        codon_seqs = [codon_seqs]
    elif isinstance(codon_seqs, list) and all(isinstance(i, int) for i in codon_seqs):
        return codon_seqs  # Return already encoded sequence

    # Guard against non-string sequences
    if not all(isinstance(seq, str) for seq in codon_seqs):
        raise TypeError("All sequences should be of type string (Util2 Encode Codon error)")

    encoded_codons = []
    for seq in codon_seqs:
        encoded_seq = []
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            if codon not in codon_to_int:
                raise ValueError(f"Unknown codon encountered: {codon}. Check your codon_to_int mapping.")
            encoded_seq.append(codon_to_int[codon])
        encoded_codons.append(encoded_seq)

    return encoded_codons





def collate_fn(batch):
    src_sequences, tgt_sequences = zip(*batch)
    # Padding sequences
    src_sequences = pad_sequence(src_sequences, batch_first=True)
    tgt_sequences = pad_sequence(tgt_sequences, batch_first=True)
    return src_sequences, tgt_sequences


def test_encode_codon_sequence():
    # Define a sample codon list
    global codons
    codons = ['ATG', 'CTG', 'TTG', 'TAA', 'TAG', 'TGA']

    # 1. Test for valid single string codon sequences.
    single_codon = 'ATG'
    encoded_single = encode_codon_sequence(single_codon)
    assert encoded_single == [[0]], f"Expected [[0]], got {encoded_single}"
    print("Test 1 passed! (single string codon sequence)")

    # 2. Test for valid list of string codon sequences.
    list_codons = ['ATG', 'CTG', 'TTG']
    encoded_list = encode_codon_sequence(list_codons)
    assert encoded_list == [[0], [1], [2]], f"Expected [[0], [1], [2]], got {encoded_list}"
    print("Test 2 passed! (list of string codon sequences)")

    # 3. Test for a list of already encoded integers.
    encoded_integers = [0, 1, 2]
    output_integers = encode_codon_sequence(encoded_integers)
    assert output_integers == encoded_integers, f"Expected {encoded_integers}, got {output_integers}"
    print("Test 3 passed! (list of already encoded integers)")

    # 4. Test for invalid sequences.
    try:
        invalid_seq = [123, 'ATG']
        encode_codon_sequence(invalid_seq)
    except TypeError as e:
        assert str(e) == "All sequences should be of type string (Util2 Encode Codon error)"
    print("Test 4 passed! (invalid sequences)")

    # 5. Test for unknown codons.
    try:
        unknown_codons = ['AGT']
        encode_codon_sequence(unknown_codons)
    except ValueError as e:
        assert str(e) == "Unknown codon encountered: AGT. Check your codon_to_int mapping."
    print("Test 5 passed! (unknown codons)")

    # 6. Test for padded codons.
    padded_codons = ['AN', 'TN', 'GN', 'CN', 'AAN', 'TNN']
    encoded_padded = encode_codon_sequence(padded_codons)
    expected_encoded_padded = [[codon_to_int[codon] for codon in padded_codons]]
    assert encoded_padded == expected_encoded_padded, f"Expected {expected_encoded_padded}, got {encoded_padded}"
    print("Test 6 passed! (padded codons)")

    print("All tests passed!")


if __name__ == "__main__":

    test_encode_codon_sequence()
    
    exit()
    # Parse the FASTA file
    data_df = parse_fasta(gencode_source_file_path)
    print("\nParsed FASTA Data:")
    print(data_df.head())  # Displaying the first 5 rows for brevity

    # Extract valid rows based on CDS columns
    valid_data_df = extract_cds_columns(data_df)
    print("\nValid Rows After Extracting CDS Columns:")
    print(valid_data_df.head())

    # Extract amino acid and codon sequences
    aa_sequences, codon_sequences = extract_sequences(valid_data_df)
    print("\nExtracted Amino Acid Sequences (first 5):")
    print(aa_sequences[:5])
    print("\nExtracted Codon Sequences (first 5):")
    print(codon_sequences[:5])

    # Encoding the amino acid and codon sequences
    encoded_aa_sequences = encode_amino_sequence(aa_sequences)
    encoded_codon_sequences = encode_codon_sequence(codon_sequences)
    print("\nEncoded Amino Acid Sequences (first 5):")
    print(encoded_aa_sequences[:5])
    print("\nEncoded Codon Sequences (first 5):")
    print(encoded_codon_sequences[:5])

    # Load source and target sequences together
    src_seqs, tgt_seqs = load_src_tgt_sequences()
    print("\nSource Sequences from load_src_tgt_sequences (first 5):")
    print(src_seqs[:5])
    print("\nTarget Sequences from load_src_tgt_sequences (first 5):")
    print(tgt_seqs[:5])

    # Using the collate function (assuming batch size of 5 for simplicity)
    collated_src, collated_tgt = collate_fn(list(zip(src_seqs[:5], tgt_seqs[:5])))
    print("\nCollated Source Sequences (with a simulated batch size of 5):")
    print(collated_src)
    print("\nCollated Target Sequences (with a simulated batch size of 5):")
    print(collated_tgt)
