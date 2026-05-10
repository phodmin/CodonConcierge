import pandas as pd
import requests

def get_uniprot_ids():
    url = "https://www.uniprot.org/uniprot/?query=organism:9606&format=list"
    response = requests.get(url)
    ids = response.text.split()
    return ids

def get_ena_id_from_uniprot(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
    response = requests.get(url)
    # Extract the ENA ID from the XML. This is a simplification; in reality, you might need to parse the XML.
    ena_id = "EXTRACTED_FROM_XML"
    return ena_id

def get_nucleotide_sequence_from_ena(ena_id):
    url = f"https://www.ebi.ac.uk/ena/data/view/{ena_id}&display=fasta"
    response = requests.get(url)
    sequence = response.text.split("\n", 1)[1]
    return sequence

def main():
    uniprot_ids = get_uniprot_ids()
    data = []

    for uid in uniprot_ids:
        ena_id = get_ena_id_from_uniprot(uid)
        nucleotide_seq = get_nucleotide_sequence_from_ena(ena_id)
        data.append({'uniprot_id': uid, 'ena_id': ena_id, 'nucleotide_sequence': nucleotide_seq})

    df = pd.DataFrame(data)
    print(df)

if __name__ == "__main__":
    main()