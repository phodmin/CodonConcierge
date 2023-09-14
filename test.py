from util import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bio.Seq import Seq
# from Bio.Alphabet import IUPAC

def parse_amino_acid_fasta(file_path):
    """
    Parse amino acid sequences from a FASTA file.
    """
    records = []
    with open(file_path, "r") as handle:
        sequence = ""
        header = None
        for line in handle:
            line = line.strip()
            if line.startswith(">"):  # Header line
                if header:
                    records.append({
                        "header": header,
                        "sequence": sequence
                    })
                header = line[1:]
                sequence = ""
            else:
                sequence += line
        if header:
            records.append({
                "header": header,
                "sequence": sequence
            })

    parsed_records = {record["header"].split("|")[0]: record["sequence"] for record in records}
    return parsed_records

def get_longest_cds_length_from_parsed_records(records):
    """
    Given a list of parsed records, derive and return the length of the longest CDS sequence.
    """
    longest_length = 0
    longest_record = None
    
    for record in records:
        # Check if the record has CDS information
        if "CDS" in record:
            try:
                cds_start, cds_end = map(int, record["CDS"].split('-'))
                cds_sequence = record["sequence"][cds_start-1:cds_end]  # Adjusting for 0-based index
                cds_length = len(cds_sequence)
                
                if cds_length > longest_length:
                    longest_length = cds_length
                    longest_record = record
            except ValueError:
                print(f"Error parsing CDS range {record['CDS']} for record with transcript ID {record['transcript_id']}")
    
    # If you've found a long sequence, print its details for inspection
    if longest_record:
        print("Transcript ID of the longest CDS record:", longest_record["transcript_id"])
        if "gene_name" in longest_record:
            print("Gene name:", longest_record["gene_name"])
        if "gene_symbol_variant" in longest_record:
            print("Gene symbol variant:", longest_record["gene_symbol_variant"])
    
    return longest_length

def count_sequences_longer_than(records, threshold_length=10000):
    """
    Given a list of parsed records, count and return the number of sequences
    with a CDS length greater than the specified threshold_length.
    """
    count = 0
    
    for record in records:
        # Check if the record has CDS information
        if "CDS" in record:
            try:
                cds_start, cds_end = map(int, record["CDS"].split('-'))
                cds_sequence = record["sequence"][cds_start-1:cds_end]  # Adjusting for 0-based index
                cds_length = len(cds_sequence)
                
                if cds_length > threshold_length:
                    count += 1
                    
            except ValueError:
                print(f"Error parsing CDS range {record['CDS']} for record with transcript ID {record['transcript_id']}")
    
    return count

def get_top_100_longest_transcripts(file_path):
    parsed_records = basic_fasta_parser(file_path)

    # Sorting the records based on sequence length in descending order
    sorted_records = sorted(parsed_records, key=lambda x: x["sequence_length"], reverse=True)

    # Extracting the required details from the top 100 records
    top_100_details = []
    for record in sorted_records[:100]:
        details = {
            "gene_name": record.get("gene_name", "N/A"),
            "gene_id": record.get("gene_id", "N/A"),
            "sequence_length": record.get("sequence_length")
        }
        top_100_details.append(details)
    
    return top_100_details

def check_divisibility_by_three(records):
    divisible_by_three = 0
    not_divisible_by_three = 0

    for record in records:
        # Check if the record has CDS information
        if "CDS" in record:
            try:
                cds_start, cds_end = map(int, record["CDS"].split('-'))
                cds_sequence = record["sequence"][cds_start-1:cds_end]  # Adjusting for 0-based index
                cds_length = len(cds_sequence)
                
                if cds_length % 3 == 0:
                    divisible_by_three += 1
                else:
                    not_divisible_by_three += 1
                    
            except ValueError:
                print(f"Error parsing CDS range {record['CDS']} for record with transcript ID {record['transcript_id']}")
    
    print(f"Number of CDS sequences divisible by 3: {divisible_by_three}")
    print(f"Number of CDS sequences NOT divisible by 3: {not_divisible_by_three}")

def plot_sequence_length_distribution(records):
    lengths = []

    for record in records:
        if "sequence" in record:
            lengths.append(len(record["sequence"]))
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Sequences')
    plt.grid(axis='y', linestyle='--')
    plt.show()

# Plot only up to X length
def plot_sequence_length_distribution_v2(records, max_length):
    lengths = []

    for record in records:
        if "sequence" in record:
            lengths.append(len(record["sequence"]))
    
    # Set the range of x-axis
    x_range = (1, max_length)
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='blue', alpha=0.7, range=x_range)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Sequences')
    plt.grid(axis='y', linestyle='--')
    plt.show()


def plot_log_sequence_length_distribution(records):
    lengths = []

    for record in records:
        if "sequence" in record:
            lengths.append(len(record["sequence"]))
    
    # Generating logarithmically spaced bins
    min_length = min(lengths)
    max_length = max(lengths)
    bins = np.logspace(np.log10(min_length), np.log10(max_length), num=50)  # 50 logarithmically spaced bins

    # Plotting the histogram
    plt.figure(figsize=(12, 7))
    plt.hist(lengths, bins=bins, color='blue', alpha=0.7)
    plt.title('Logarithmic Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length (Log scale)')
    plt.ylabel('Number of Sequences')
    plt.xscale('log')
    plt.grid(axis='y', linestyle='--')

    # Annotating the specified sequence lengths
    annotation_points = [100, 500, 1000, 2000, 5000, 8000, 16000]
    for point in annotation_points:
        plt.annotate(f"{point}", (point, 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

    plt.show()

def print_long_sequences_count(parsed_records, start, end, step, specific_lengths=None):
    """
    Given a range (start, end, step) and a list of parsed records, 
    print the number of CDS sequences longer than lengths in that range.
    If specific_lengths is provided, it will also check for those specific lengths.
    """
    for threshold in range(start, end + 1, step):
        count_long_sequences = count_sequences_longer_than(parsed_records, threshold)
        print(f"Number of CDS sequences longer than {threshold:,} nucleotides: {count_long_sequences}")
    
    if specific_lengths:
        for length in specific_lengths:
            count_long_sequences = count_sequences_longer_than(parsed_records, length)
            print(f"Number of CDS sequences longer than {length:,} nucleotides: {count_long_sequences}")
    
    print(f"Total number of sequences in the dataset: {len(parsed_records)}")

def validate_sequences(source_file, aa_source_file):
    """
    Validate nucleotide sequences against amino acid sequences by checking translation and length.
    
    Parameters:
    - source_file (str): Path to the nucleotide sequence file.
    - aa_source_file (str): Path to the amino acid sequence file.
    
    Returns:
    - count_no_aa (int): Count of nucleotide sequences for which no corresponding amino acid sequence was found.
    """
    
    parsed_records = basic_fasta_parser(source_file)
    aa_sequences = parse_amino_acid_fasta(aa_source_file)
    
    count_no_aa = 0

    for record in parsed_records:
        transcript_id = record["transcript_id"]
        nucleotide_sequence = Seq(record["sequence"])
        
        try:
            translated_sequence = nucleotide_sequence.translate(to_stop=True)
        except:
            # Handle any exceptions in translation (like unknown nucleotides)
            print(f"Error translating sequence for {transcript_id}")
            continue

        # If the corresponding amino acid sequence exists
        if transcript_id in aa_sequences:
            given_aa_sequence = aa_sequences[transcript_id]
            
            # Compare the lengths
            if len(translated_sequence) != len(given_aa_sequence):
                print(f"Length mismatch for {transcript_id}: Expected {len(given_aa_sequence)} but got {len(translated_sequence)}")

                # Check for incomplete codons
                remainder = len(nucleotide_sequence) % 3
                if remainder:
                    incomplete_codon = nucleotide_sequence[-remainder:]
                    print(f"Incomplete codon for {transcript_id}: {incomplete_codon}")

            # You can also compare sequences directly and report discrepancies
            # If the sequences do not match:
            if translated_sequence != given_aa_sequence:
                print(f"Sequence mismatch for {transcript_id}")
                
        else:
            count_no_aa += 1

    return count_no_aa

if __name__ == "__main__":
    source_file = '../data/gencode/gencode.v44.pc_transcripts.fa'
    aa_source_file = '../data/gencode/gencode.v44.pc_translations.fa'
    
    count_missing_aa = validate_sequences(source_file, aa_source_file)
    print(f"Number of nucleotide sequences with no corresponding amino acid sequence: {count_missing_aa}")

    parsed_records = basic_fasta_parser(source_file)

    # unique_characters = set()
    # for record in parsed_records:
    #     sequence = record["sequence"]
    #     unique_characters.update(sequence)
    
    # print("Unique characters in the dataset:", sorted(unique_characters))

    # longest_cds_length = get_longest_cds_length_from_parsed_records(parsed_records)
    # print("Length of the Longest CDS Sequence:", longest_cds_length)
    # total_sequences = len(parsed_records)
    # top_100_records = get_top_100_longest_transcripts(source_file)
    # for index, record in enumerate(top_100_records, 1):
    #     print(f"{index}. Gene Name: {record['gene_name']}, Gene ID: {record['gene_id']}, Length: {record['sequence_length']}")
    # check_divisibility_by_three(parsed_records)  # Check divisibility
    
    #plot_sequence_length_distribution(parsed_records)
    
    #-------
    
    #plot_sequence_length_distribution_v2(parsed_records,10000)

    #-------

    print_long_sequences_count(parsed_records, 0000,1000, 100)
    #plot_log_sequence_length_distribution(parsed_records)
