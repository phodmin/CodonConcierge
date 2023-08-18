from Bio import SeqIO

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