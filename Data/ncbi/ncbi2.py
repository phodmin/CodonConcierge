from Bio import Entrez, SeqIO

def fetch_sequences(num_records=110000):
    # Set your email for NCBI
    Entrez.email = "miso.karlubik@gmail.com"
    
    # Query NCBI for human mRNA sequences from RefSeq that are protein coding
    query = "Homo sapiens[ORGN] AND RefSeq[DB]"
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=num_records)
    record = Entrez.read(handle)

    handle.close()
    
    print(record)

    # Print number of results found
    print(f"Number of results found: {record['Count']}")

    # Get the sequence IDs
    id_list = record["IdList"]

    if not id_list:
        print("No sequences found for the query.")
        return []

    # Fetch the sequences based on IDs
    handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="gb", retmode="text")
    records = list(SeqIO.parse(handle, "genbank"))
    handle.close()
    
    return records

# Use the function
sequences = fetch_sequences()
counter = 0

# Print the sequences
for seq_record in sequences:
    
    print("sequence #"+str(counter))
    print(f"ID: {seq_record.id}")
    print(f"Description: {seq_record.description}")
    print(f"Sequence: {seq_record.seq}\n")
    counter += 1

