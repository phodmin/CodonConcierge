from Bio import Entrez, SeqIO
from io import StringIO
import csv
import time
import http.client

# Set your email (required by NCBI)
Entrez.email = "miso.karlubik@gmail.com"

# Define a function to fetch mRNA sequences with pagination
def fetch_mRNA_sequences(batch_size=10000):  # Reduced batch size
    sequences = []
    retries = 3
    delay = 5  # Added delay variable

    # Initial search to get total count of records
    handle = Entrez.esearch(db="nucleotide", term="Homo sapiens[orgn] AND mRNA[filter]")
    record = Entrez.read(handle)
    total_count = int(record["Count"])
    handle.close()
    
    for start in range(0, total_count, batch_size):
        for _ in range(retries):
            try:
                handle = Entrez.esearch(db="nucleotide", term="Homo sapiens[orgn] AND mRNA[filter]", retstart=start, retmax=batch_size)
                record = Entrez.read(handle)
                id_list = record["IdList"]
                handle.close()
                
                # Fetch sequences for this batch
                handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="gb", retmode="text")
                data = handle.read()  # Read the entire response into a string
                handle.close()
                batch_records = list(SeqIO.parse(StringIO(data), "genbank"))
                sequences.extend(batch_records)
                break  # If successful, break out of the retry loop
            except http.client.IncompleteRead:
                print(f"Error fetching records {start}-{start+batch_size}. Retrying in {delay} seconds...")
                time.sleep(delay)
    
    return sequences

# Fetch sequences
sequences = fetch_mRNA_sequences()

# Save to CSV
with open("../data/NCBI_mRNA_sequences.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "Sequence"])
    for seq_record in sequences:
        writer.writerow([seq_record.id, str(seq_record.seq)])
