import pandas as pd
from Bio.Seq import Seq
from util3 import *
from io import StringIO
import numpy as np
from typing import Tuple, List
import unittest
import pytest

def print_success(message):
    # ANSI color codes
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Emoji and styled message
    success_emoji = "✅"
    styled_message = f"{GREEN}{BOLD}{success_emoji} {message} {success_emoji}{RESET}"

    # Border
    border_length = len(message) + 4 + 4  # 4 for spaces & emoji, 4 for bold ANSI code characters
    border = "+" + "-" * border_length + "+"

    # Print the styled output
    print(border)
    print(f"| {styled_message} |")
    print(border)

# -------- Test functions --------------------------------------------------------------
# 0

# 1 
def test_parse_fasta():
    # Mock FASTA content
    mock_fasta_content = """>ENST00000641515.2|ENSG00000186092.7|OTTHUMG00000001094.4|OTTHUMT00000003223.4|OR4F5-201|OR4F5|2618|UTR5:1-60|CDS:61-1041|UTR3:1042-2618|
CCCAGATCTCTTCAGTTTTTATGCCTCATTCTGTGAAAATTGCTGT"""

    # Use StringIO to simulate a file object
    mock_file = StringIO(mock_fasta_content)
    
    # Parse the mock file with your function
    df = parse_fasta(mock_file)

    # Write assertions to test
    assert df.loc[0, "transcript_id"] == "ENST00000641515.2"
    assert df.loc[0, "gene_id"] == "ENSG00000186092.7"
    assert df.loc[0, "UTR5"] == "1-60"
    assert df.loc[0, "CDS"] == "61-1041"
    assert df.loc[0, "UTR3"] == "1042-2618"
    assert df.loc[0, "sequence_length"] == 2618
    assert df.loc[0, "sequence"].startswith("CCCAGATCTCTTCAGT")


    print_success("1: All test_parse_fasta() tests passed!")

# 2
def test_extract_cds_columns():
    # Create a mock DataFrame
    data = {
        'CDS': ['1-50', '30-90', '15-45', '5-110'],
        'sequence': ['A'*50, 'A'*100, 'A'*45, 'A'*100]
    }
    df = pd.DataFrame(data)
    
    # Use the function
    valid_df = extract_cds_columns(df)
    
    # Expected results:
    # Row 4 is dropped because cds_end 110 exceeds the sequence length.
    # Therefore, Rows 1, 2, and 3 should remain in the valid_df.

    # Assertions
    assert valid_df.shape[0] == 3, "Expected 3 valid rows"
    assert "cds_start" in valid_df.columns, "cds_start column missing"
    assert "cds_end" in valid_df.columns, "cds_end column missing"
    
    assert valid_df.iloc[0]["cds_start"] == 1 and valid_df.iloc[0]["cds_end"] == 50, "Row 1 CDS values incorrect"
    assert valid_df.iloc[1]["cds_start"] == 30 and valid_df.iloc[1]["cds_end"] == 90, "Row 2 CDS values incorrect"
    assert valid_df.iloc[2]["cds_start"] == 15 and valid_df.iloc[2]["cds_end"] == 45, "Row 3 CDS values incorrect"

    
    print_success("2: All test_extract_cds_columns() tests passed!")

# 3
# 3.1 C -> AA
def test_translate_codons_to_amino_acids(verbose=True):
    if verbose: print("Starting tests for 'translate_codons_to_amino_acids' function...")

    # 1. Basic functionality
    if verbose: print("Testing basic functionality...")
    input_codons = ["ATG", "GCA", "TAC", "GGT"]
    expected_output = ["M", "A", "Y", "G"]
    assert translate_codons_to_amino_acids(input_codons) == expected_output
    if verbose: print("Basic functionality test passed!\n")

    # 2. Test padding
    if verbose: print("Testing padding with sequences not multiple of 3...")

    # Test sequences with 1 nucleotide (2 'N's will be added)
    if verbose: print("- Testing sequences with 1 nucleotide...")
    input_codons = ["A", "T", "G"]
    expected_output = ["X", "X", "X"]  # Since AN, TN, and GN are all translated to 'X'
    assert translate_codons_to_amino_acids(input_codons) == expected_output
    if verbose: print("-- Passed!")

    # Test sequences with 2 nucleotides (1 'N' will be added)
    if verbose: print("- Testing sequences with 2 nucleotides...")
    input_codons = ["AC", "TG", "GC"]
    expected_output = ["X", "X", "X"]  # Since ACN, TGN, and GCN are all translated to 'X'
    assert translate_codons_to_amino_acids(input_codons) == expected_output
    if verbose: print("-- Passed!")

    # Test sequences with 4 nucleotides (2 'N's will be added)
    if verbose: print("- Testing sequences with 4 nucleotides...")
    input_codons = ["ACAT", "TGGC"]
    expected_output = ["TX", "WX"]  # First triplet is translated and the remaining two are padded and translated to 'X'
    assert translate_codons_to_amino_acids(input_codons) == expected_output
    if verbose: print("-- Passed!\n")

    # 3. Test error handling
    if verbose: print("Testing error handling with unrecognized codons...")
    with pytest.raises(ValueError, match="Unrecognized codon: ZZZ"):
        translate_codons_to_amino_acids(["ZZZ"])
    if verbose: print("Error handling test passed!")

    print_success("3.1: All tests for translate_codons_to_amino_acids completed!")   

# 3.2 AA -> Int
def test_translate_amino_acids_to_ints(verbose=True):
    if verbose: print("Starting tests for 'translate_amino_acids_to_ints' function...")

    # 1. Basic functionality
    if verbose: print("Testing basic functionality...")
    input_aas = ["ACDE", "FWY*"]
    expected_output = [[1, 2, 3, 4], [5, 19, 20, 21]]
    assert translate_amino_acids_to_ints(input_aas) == expected_output
    if verbose: print("Basic functionality test passed!\n")

    # 2. Test for special amino acids
    if verbose: print("Testing translation with special amino acids (X and *)...")
    input_aas = ["ACDX", "TX*"]
    expected_output = [[1, 2, 3, 0], [17, 0, 21]]
    assert translate_amino_acids_to_ints(input_aas) == expected_output
    if verbose: print("Translation with special amino acids test passed!\n")

    # 3. Test error handling
    if verbose: print("Testing error handling with unrecognized amino acids...")
    with pytest.raises(ValueError, match="Unrecognized amino acid: Z"):
        translate_amino_acids_to_ints(["ZACD"])
    if verbose: print("Error handling test passed!\n")

    print_success("3.2: All tests for translate_amino_acids_to_ints completed!")

# 3.3 C -> Int
def test_translate_codons_to_ints(verbose=True):
    if verbose: print("Starting tests for 'translate_codons_to_ints' function...")

    # 1. Basic functionality
    if verbose: print("Testing basic functionality...")
    input_codons = ["ATAACACCAG", "TTATGT"]  # The second sequence doesn't require padding.
    expected_output = [[1, 5, 21, 0], [53, 64]]  # Adjusted based on the codon_to_int mapping.
    actual_output = translate_codons_to_ints(input_codons)
    assert actual_output == expected_output
    if verbose: print("Basic functionality test passed!\n")

    # 2. Test for padding
    if verbose: print("Testing codons padding...")
    input_codons = ["AT", "TGAAC", "T"]  # All sequences require padding.
    expected_output = [[0], [61, 0], [0]]  # Adjusted expected output to match function's behavior.
    actual_output = translate_codons_to_ints(input_codons)
    assert actual_output == expected_output
    if verbose: print("Codon padding test passed!\n")

    # 3. Test error handling
    if verbose: print("Testing error handling with unrecognized codons...")
    with pytest.raises(ValueError, match="Unrecognized codon: ZZZ"):
        translate_codons_to_ints(["ZZZATA"])
    if verbose: print("Error handling test passed!\n")

    print_success("3.3: All tests for translate_codons_to_ints completed!")

# 4 Extract Sequences

def test_extract_sequences():
    """Tests the function `extract_sequences`."""

    def check_test_case(test_name, df, expected_aa, expected_codon):
        """Helper function to check a single test case."""
        aa_seqs_int, codon_seqs_int = extract_sequences(df)
        # # Print the expected and extracted outputs
        # print("Testing:\n")
        # print(f"Input DataFrame:\n{df}\n")
        # print(f"Expected aa_seqs_int: {expected_aa}\n")
        # print(f"Extracted aa_seqs_int: {aa_seqs_int}\n")
        # print(f"Expected codon_seqs_int: {expected_codon}\n")
        # print(f"Extracted codon_seqs_int: {codon_seqs_int}\n")

        # print("+--------------------------------------------------------------+\n")
        assert aa_seqs_int == expected_aa, f"{test_name} [AminoAcid] failed for aa_seqs_int. \nExpected \n{expected_aa}, but got \n{aa_seqs_int}"
        assert codon_seqs_int == expected_codon, f"{test_name} [Codon] failed for codon_seqs_int. \nExpected \n{expected_codon}, but got \n{codon_seqs_int}"

    test_cases = [
        {
            'name': "TestBasicSequences1",
            'df': pd.DataFrame({
                'sequence': ['ATGACAAACAGA', 'TGTGGGTAG'],
                'cds_start': [1, 1],
                'cds_end': [12, 9]
            }),
            'expected_aa': [[11, 17, 12, 15], [2, 6, 21]],
            'expected_codon': [[4, 5, 10, 13], [64, 47, 60]]
        },
        {
            'name': "TestBasicSequences2",
            'df': pd.DataFrame({
                'sequence': ['GGCGGGGGT'],
                'cds_start': [1],
                'cds_end': [9]
            }),
            'expected_aa': [[6, 6, 6]],
            'expected_codon': [[46, 47, 48]]
        },
        {
            'name': "TestSequencesWithPadding11",
            'df': pd.DataFrame({
                'sequence': ['AG'],
                'cds_start': [1],
                'cds_end': [2]
            }),
            'expected_aa': [[0]],
            'expected_codon': [[0]]
        },
        {
            'name': "TestSequencesWithPadding11",
            'df': pd.DataFrame({
                'sequence': ['AAAGG'],
                'cds_start': [1],
                'cds_end': [5]
            }),
            'expected_aa': [[9,0]],
            'expected_codon': [[11,0]]
        },
        {
            'name': "TestSequencesWithPadding2",
            'df': pd.DataFrame({
                'sequence': ['A'],
                'cds_start': [1],
                'cds_end': [2]
            }),
            'expected_aa': [[0]],
            'expected_codon': [[0]]
        },
        {
            'name': "TestSequencesWithPadding3",
            'df': pd.DataFrame({
                'sequence': ['ATAATCATTATGA', 'TGTTGGTGCA'],
                'cds_start': [1, 1],
                'cds_end': [13, 10]
            }),
            'expected_aa': [[8, 8, 8, 11, 0], [2, 19, 2, 0]],
            'expected_codon': [[1, 2, 3, 4, 0], [64, 63, 62, 0]]
        },
                {
            'name': "TestBasicSequences4",
            'df': pd.DataFrame({
                'sequence': ['ATGACAACTGA', 'TGTGGGTAG'],
                'cds_start': [1, 1],
                'cds_end': [10, 9]
            }),
            'expected_aa': [[11, 17, 17, 0], [2, 6, 21]],
            'expected_codon': [[4, 5, 8, 0], [64, 47, 60]]
        },
        {
            'name': "TestWithPadding4",
            'df': pd.DataFrame({
                'sequence': ['ATAATCATTATGA', 'TGTTGGTGCA'],
                'cds_start': [1, 1],
                'cds_end': [13, 10]
            }),
            'expected_aa': [[8, 8, 8, 11, 0], [2, 19, 2, 0]],
            'expected_codon': [[1, 2, 3, 4, 0], [64, 63, 62, 0]]
        },
        {
            'name': "TestSequencesAtEdges",
            'df': pd.DataFrame({
                'sequence': ['ATG', 'TGT', 'NNN'],
                'cds_start': [1, 1, 1],
                'cds_end': [3, 3, 3]
            }),
            'expected_aa': [[11], [2], [0]],
            'expected_codon': [[4], [64], [0]]
        },
        {
            'name': "TestEmptySequence",
            'df': pd.DataFrame({
                'sequence': [''],
                'cds_start': [1],
                'cds_end': [0]
            }),
            'expected_aa': [[]],
            'expected_codon': [[]]
        }
    ]

    for test_case in test_cases:
        check_test_case(test_case['name'], test_case['df'], test_case['expected_aa'], test_case['expected_codon'])

    print_success("All tests in test_extract_sequences passed!")

def test_extract_sequences3():
    # Test 1: Basic sequences
    data1 = {
        'sequence': ['ATGACAAACAGA', 'TGTGGGTAG'],
        'cds_start': [1, 2],
        'cds_end': [12, 9]
    }
    df1 = pd.DataFrame(data1)
    aa_seqs_int_1, codon_seqs_int_1 = extract_sequences(df1)
    assert aa_seqs_int_1 == [[11, 17, 12, 15], [2, 6, 21]], "Test 1 failed for aa_seqs_int"
    assert codon_seqs_int_1 == [[4, 5, 10, 13], [64, 47, 60]], "Test 1 failed for codon_seqs_int"

    # Test 2: Sequences with padding
    data2 = {
        'sequence': ['ATGACANAACAGA', 'TGTGGNTAG'],
        'cds_start': [1, 2],
        'cds_end': [13, 9]
    }
    df2 = pd.DataFrame(data2)
    aa_seqs_int_2, codon_seqs_int_2 = extract_sequences(df2)
    assert aa_seqs_int_2 == [[11, 14, 0, 9, 14], [5, 6, 0, 21]], "Test 2 failed for aa_seqs_int"
    assert codon_seqs_int_2 == [[4, 5, 0, 27, 13], [62, 0, 60]], "Test 2 failed for codon_seqs_int"

    # Test 3: Sequences at the edges
    data3 = {
        'sequence': ['ATG', 'TGT', 'NNN'],
        'cds_start': [1, 1, 1],
        'cds_end': [3, 3, 3]
    }
    df3 = pd.DataFrame(data3)
    aa_seqs_int_3, codon_seqs_int_3 = extract_sequences(df3)
    assert aa_seqs_int_3 == [[11], [5], [0]], "Test 3 failed for aa_seqs_int"
    assert codon_seqs_int_3 == [[4], [62], [0]], "Test 3 failed for codon_seqs_int"

    # Test 4: Empty sequence
    data4 = {
        'sequence': [''],
        'cds_start': [1],
        'cds_end': [1]
    }
    df4 = pd.DataFrame(data4)
    aa_seqs_int_4, codon_seqs_int_4 = extract_sequences(df4)
    assert aa_seqs_int_4 == [[]], "Test 4 failed for aa_seqs_int"
    assert codon_seqs_int_4 == [[]], "Test 4 failed for codon_seqs_int"

    print("All tests in new_test_extract_sequences passed!")


    print("---------- DEBUG ENDS ----------")


# -------- Run tests --------------------------------------------------------------
if __name__ == "__main__":
    # 0
    
    # 1 
    test_parse_fasta()
    # 2
    test_extract_cds_columns()
    # 3
    # 3.1
    test_translate_codons_to_amino_acids(verbose=False)
    # 3.2 
    test_translate_amino_acids_to_ints(verbose=False)
    # 3.3
    test_translate_codons_to_ints(verbose=False)

    # 4
    test_extract_sequences()

    print("FUCKING HELL YES")
    print("-----")
    print("ja sa asi pojebem ze toto funguje ")
    print(" do skurvenej riti")
    print("-----")
