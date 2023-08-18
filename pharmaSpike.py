from codonCalculator import *
from Bio import SeqIO

# Define the Spike Protein Covid-19 vaccine sequence source
spike = '../data/spikecov19_pfizer_moderna.fasta'
sequences = list(SeqIO.parse(spike, 'fasta'))
bion = sequences[0]
mrna = sequences[1]

# bion_cds = find_coding_sequences(bion)
# mrna_cds = find_coding_sequences(mrna)

# print(bion_cds)
# print(mrna_cds)



# Define the filename (for the GENCODE transcript sequences database)
filename = 'data/gencode.v44.pc_transcripts.fa'

# Parse the fasta file and convert it to a list
sequences = list(SeqIO.parse(filename, 'fasta'))
for record in sequences:
    example = record
    break

print(example)

CDS = example.seq[60:1041]

print(CDS)

unique = countUniqueSequences(filename)
nonunique = countNonUniqueSequences(filename)
print("Number of unique sequences:" , unique)
print("Number of non-unique sequences:" , nonunique)

if unique == nonunique:
    print("The number of unique sequences is equal to the number of non-unique sequences.")
else:
    print("The number of unique sequences is NOT equal to the number of non-unique sequences.")



