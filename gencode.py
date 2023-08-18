from Bio import SeqIO
import pandas as pd

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

# Assuming you have the FASTA file named 'gencode.v44.transcripts.fa'
file_path = '../data/gencode.v44.pc_transcripts.fa'
filled_transcripts = parse_fasta_fill_UTR(file_path)
transcripts = parse_fasta(file_path)

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
df_filled.to_csv('data/gencode_formatted_UTR_fill.csv', index=False)
df.to_csv('data/gencode_formatted.csv', index=False)
