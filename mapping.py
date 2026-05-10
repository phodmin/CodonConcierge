# Mapping codons to amino acids, standard capitalised IUPAC codes
# Padded codons (any that include N) are mapped to 'X'
codon_to_aa = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    # Adding the padded codons
    'ANN':'X', 'CNN':'X', 'GNN':'X', 'TNN':'X',
    'AAN':'X', 'CAN':'X', 'GAN':'X', 'TAN':'X',
    'ANA':'X', 'CNA':'X', 'GNA':'X', 'TNA':'X',
    'ANC':'X', 'CNC':'X', 'GNC':'X', 'TNC':'X',
    'ANG':'X', 'CNG':'X', 'GNG':'X', 'TNG':'X',
    'ANT':'X', 'CNT':'X', 'GNT':'X', 'TNT':'X',
    'AGN':'X', 'CGN':'X', 'GGN':'X', 'TGN':'X',
    'ATN':'X', 'CTN':'X', 'GTN':'X', 'TTN':'X',
    'ACN':'X', 'CCN':'X', 'GCN':'X', 'TCN':'X',
    'NAA':'X', 'NAC':'X', 'NAG':'X', 'NAT':'X',
    'NCA':'X', 'NCC':'X', 'NCG':'X', 'NCT':'X',
    'NGA':'X', 'NGC':'X', 'NGG':'X', 'NGT':'X',
    'NTA':'X', 'NTC':'X', 'NTG':'X', 'NTT':'X',
    'NAN':'X', 'NCN':'X', 'NGN':'X', 'NTN':'X',
    'NNN':'X'
}

# Mapping amino acids to integers, 1-20
# Unknown Amino Acid ('X') is mapped to '0'
# Stop codon ('*') is mapped to '21'
aa_to_int = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'K': 9, 'L':10, 'M':11, 'N':12,
    'P':13, 'Q':14, 'R':15, 'S':16,
    'T':17, 'V':18, 'W':19, 'Y':20,
    # Unknown Amino Acid
    'X': 0, '*':21
}

# Mapping codons to ints, 1-64
# Padded codons (any that include N) are mapped to '0'
codon_to_int = {
    'ATA':1, 'ATC':2, 'ATT':3, 'ATG':4,
    'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8,
    'AAT':9, 'AAC':10, 'AAA':11, 'AAG':12,
    'AGA':13, 'AGC':14, 'AGG':15, 'AGT':16,
    'CTA':17, 'CTC':18, 'CTT':19, 'CTG':20,
    'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24,
    'CAT':25, 'CAC':26, 'CAA':27, 'CAG':28,
    'CGA':29, 'CGC':30, 'CGG':31, 'CGT':32,
    'GTA':33, 'GTC':34, 'GTT':35, 'GTG':36,
    'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40,
    'GAT':41, 'GAC':42, 'GAA':43, 'GAG':44,
    'GGA':45, 'GGC':46, 'GGG':47, 'GGT':48,
    'TCA':49, 'TCC':50, 'TCT':51, 'TCG':52,
    'TTA':53, 'TTC':54, 'TTT':55, 'TTG':56,
    'TAT':57, 'TAC':58, 'TAA':59, 'TAG':60,
    'TGA':61, 'TGC':62, 'TGG':63, 'TGT':64,
    # Adding the padded codons
    'ANN':0, 'CNN':0, 'GNN':0, 'TNN':0,
    'AAN':0, 'CAN':0, 'GAN':0, 'TAN':0,
    'ANA':0, 'CNA':0, 'GNA':0, 'TNA':0,
    'ANC':0, 'CNC':0, 'GNC':0, 'TNC':0,
    'ANG':0, 'CNG':0, 'GNG':0, 'TNG':0,
    'ANT':0, 'CNT':0, 'GNT':0, 'TNT':0,
    'AGN':0, 'CGN':0, 'GGN':0, 'TGN':0,
    'ATN':0, 'CTN':0, 'GTN':0, 'TTN':0,
    'ACN':0, 'CCN':0, 'GCN':0, 'TCN':0,
    'NAA':0, 'NAC':0, 'NAG':0, 'NAT':0,
    'NCA':0, 'NCC':0, 'NCG':0, 'NCT':0,
    'NGA':0, 'NGC':0, 'NGG':0, 'NGT':0,
    'NTA':0, 'NTC':0, 'NTG':0, 'NTT':0,
    'NAN':0, 'NCN':0, 'NGN':0, 'NTN':0,
    'NNN':0
}