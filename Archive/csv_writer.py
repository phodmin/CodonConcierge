from Bio import SeqIO
import pandas as pd
import csv
from util import *

# 1. Simple Parsing
simple_parsed_records = basic_fasta_parser(gencode_source_file_path)
with open(simple_output_path, 'w', newline='') as csvfile:
    fieldnames = list(simple_parsed_records[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for record in simple_parsed_records:
        writer.writerow(record)
# Read the CSV
df = pd.read_csv('../data/1_simple_parsed_records.csv')

# View the first few rows
print(df.head())

# 2. Elimination of Records with Incomplete UTRs
elimination_parsed_records = incompletes_elimination_parse_fasta(gencode_source_file_path)
with open(elimination_output_path, 'w', newline='') as csvfile:
    fieldnames = list(elimination_parsed_records.values())[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for record in elimination_parsed_records.values():
        writer.writerow(record)

# 3. UTR Filling
filled_parsed_records = filling_fasta_parser(gencode_source_file_path)
with open(filled_output_path, 'w', newline='') as csvfile:
    fieldnames = list(filled_parsed_records[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for record in filled_parsed_records:
        writer.writerow(record)

