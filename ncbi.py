from Bio import Entrez

# Set your email (required by NCBI)
Entrez.email = "miso.karlubik@gmail.com"

# Query for human mRNA sequences in RefSeq
query = "Homo sapiens[Organism] AND refseq[filter] AND mRNA[filter]"
handle = Entrez.esearch(db="nucleotide", term=query, retmax=10)
record = Entrez.read(handle)
handle.close()

#txid9606[organism:exp] AND biomol_mrna[prop]

# Get the count of sequences
count = int(record["Count"])
print(f"Number of human mRNA sequences in RefSeq: {count}")

# Print the first 10 unique identifiers
print("First 10 unique identifiers:")
for unique_id in record['IdList']:
    print(unique_id)
